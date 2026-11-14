//! `MistralRecipe`: Architecture recipe for `Mistral`, `Mistral3` (Nemo / Small), and `Mixtral` models.
//!
//! Implements [`ArchRecipe`] supporting dynamic GGUF hydration, `Norm` `(2i, 2i+1)` rotary
//! embeddings on both `Q` and `K`, Sliding Window Attention (`SWA`) via [`KvCacheView`],
//! Sparse Top-`k` Mixture-of-Experts (`MoE`) routing over packed 3D or per-expert 2D weights,
//! and multi-quant parity (`F32`, `F16`, `BF16`, `Q4_0`, `Q4_K_M`, `Q5_K`, `Q6_K`, `Q8_0`)
//! per ADR-023 and ADR-024.

use jero_core::{
    ArchRecipe, ExecutionContext, GgufArray, GgufFile, GgufValue, K3Error, KvCacheView, KvGeometry,
    ModelMetadata, ModelWeights, QuantType, TensorDescriptor, TensorView,
};

use crate::admission::validate_token_id_in_vocab;
pub use crate::kernels_compat::RopeScalingConfig;
use crate::kernels_compat::{
    dequantize_embedding_row, gqa_attention_compat, gqa_causal_attention_prefill_compat,
    matmul_tensor, matvec_tensor, resolve_2d_dims, rmsnorm_compat, rope_inv_freq_compat,
    swiglu_compat, tensor_bounds_err,
};
use crate::repack_cache::{
    CachedDenseProjection, MistralActScratch, MistralLayerCache, with_shared_model_cache,
};

#[derive(Debug, Clone, Copy)]
struct MistralGeometry {
    num_layers: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    total_q: usize,
    total_kv: usize,
    vocab_size: usize,
    rms_eps: f32,
    rope_theta: f32,
    sliding_window: Option<usize>,
    moe: Option<(usize, usize)>,
}

/// Selects the top-`k` experts from router logits and computes normalized softmax gating weights.
///
/// # Errors
///
/// Fails closed with [`K3Error::CapabilityMismatch`] if `top_k == 0`, `router_logits.len() < top_k`,
/// or any router logit is non-finite (`NaN` or `Inf`).
pub fn route_topk_experts(
    router_logits: &[f32],
    top_k: usize,
) -> Result<Vec<(usize, f32)>, K3Error> {
    if top_k == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "mixtral".to_owned(),
            reason: "top_k expert count must be at least 1".to_owned(),
        });
    }
    if router_logits.len() < top_k {
        return Err(K3Error::CapabilityMismatch {
            architecture: "mixtral".to_owned(),
            reason: format!(
                "expected at least {top_k} expert router logits, got {}",
                router_logits.len()
            ),
        });
    }

    for (i, &logit) in router_logits.iter().enumerate() {
        if !logit.is_finite() {
            return Err(K3Error::CapabilityMismatch {
                architecture: "mixtral".to_owned(),
                reason: format!("non-finite router logit {logit} encountered at expert index {i}"),
            });
        }
    }

    let mut ranked: Vec<(usize, f32)> = router_logits.iter().copied().enumerate().collect();
    // Sort descending by logit, breaking exact ties by ascending expert index for determinism.
    ranked.sort_by(|(idx_a, val_a), (idx_b, val_b)| {
        val_b.total_cmp(val_a).then_with(|| idx_a.cmp(idx_b))
    });
    ranked.truncate(top_k);

    let max_val = ranked[0].1;
    let mut sum = 0.0_f32;
    let mut exps = Vec::with_capacity(top_k);
    for &(idx, val) in &ranked {
        let e = (val - max_val).exp();
        sum += e;
        exps.push((idx, e));
    }
    for (_, weight) in &mut exps {
        *weight /= sum;
    }
    Ok(exps)
}

/// Selects the top-2 experts from router logits and computes normalized softmax gating weights.
///
/// # Errors
///
/// Fails closed with [`K3Error::CapabilityMismatch`] if any router logit is non-finite (`NaN` or `Inf`)
/// or if fewer than 2 expert logits are provided.
pub fn route_top2_experts(router_logits: &[f32]) -> Result<[(usize, f32); 2], K3Error> {
    let selected = route_topk_experts(router_logits, 2)?;
    Ok([selected[0], selected[1]])
}

/// Applies GGML `Norm`-style interleaved rotary embeddings `(2i, 2i + 1)` in place to all
/// attention heads in `qk`.
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if `head_dim` is zero or odd, `qk` is not a multiple
/// of `head_dim`, or `inv_freq.len() != head_dim / 2`.
pub(crate) fn apply_norm_rope_inplace(
    qk: &mut [f32],
    head_dim: usize,
    pos: usize,
    inv_freq: &[f32],
) -> Result<(), K3Error> {
    if head_dim == 0
        || head_dim % 2 != 0
        || qk.len() % head_dim != 0
        || inv_freq.len() != head_dim / 2
    {
        return Err(K3Error::CapabilityMismatch {
            architecture: "mistral".to_owned(),
            reason: format!(
                "invalid Norm RoPE geometry: qk_len={}, head_dim={head_dim}, inv_freq_len={}",
                qk.len(),
                inv_freq.len()
            ),
        });
    }

    if pos == 0 {
        return Ok(());
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "sequence position fits cleanly in f32"
    )]
    let pos_f32 = pos as f32;
    for head in qk.chunks_exact_mut(head_dim) {
        for (i, &freq) in inv_freq.iter().enumerate() {
            let (sin_a, cos_a) = (pos_f32 * freq).sin_cos();
            let even = head[2 * i];
            let odd = head[2 * i + 1];
            head[2 * i] = even * cos_a - odd * sin_a;
            head[2 * i + 1] = even * sin_a + odd * cos_a;
        }
    }
    Ok(())
}

/// Mistral, Mistral3, and Mixtral architecture recipe implementation.
#[derive(Debug, Clone)]
pub struct MistralRecipe {
    /// Exact architecture string (`"mistral"`, `"mistral3"`, or `"mixtral"`).
    pub architecture: &'static str,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Model hidden dimension ($d_{\text{model}}$).
    pub hidden_dim: usize,
    /// Feed-forward intermediate dimension ($d_{\text{ffn}}$).
    pub ffn_dim: usize,
    /// Number of query attention heads ($`H_q`$).
    pub n_heads_q: usize,
    /// Number of key-value attention heads ($`H_{kv}`$).
    pub n_heads_kv: usize,
    /// Attention head dimension ($d_{\text{head}}$).
    pub head_dim: usize,
    /// Vocabulary dimension.
    pub vocab_size: usize,
    /// RMS normalization epsilon constant.
    pub rms_eps: f32,
    /// Base rotary theta parameter ($\theta$).
    pub rope_theta: f32,
    /// Optional rotary position embedding scaling configuration.
    pub rope_scaling: Option<RopeScalingConfig>,
    /// Sliding window attention size ($W$, e.g. `Some(4096)` for Mistral-7B-v0.1, `None` when disabled).
    pub sliding_window: Option<usize>,
    /// Total number of routed experts (e.g. 8 for Mixtral-8x7B, `None` for dense Mistral).
    pub num_experts: Option<usize>,
    /// Number of active experts routed per token (e.g. 2 for Mixtral).
    pub num_experts_per_tok: Option<usize>,
}

impl Default for MistralRecipe {
    fn default() -> Self {
        Self::new()
    }
}

impl MistralRecipe {
    /// Creates a new default dense Mistral recipe (`"mistral"` with 4096-token SWA).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            architecture: "mistral",
            num_layers: 32,
            hidden_dim: 4096,
            ffn_dim: 14_336,
            n_heads_q: 32,
            n_heads_kv: 8,
            head_dim: 128,
            vocab_size: 32_000,
            rms_eps: 1e-5,
            rope_theta: 10_000.0,
            rope_scaling: None,
            sliding_window: Some(4096),
            num_experts: None,
            num_experts_per_tok: None,
        }
    }

    /// Creates a new `Mistral3` recipe (`"mistral3"` defaults for Mistral-Nemo / Mistral-Small-3.x).
    #[must_use]
    pub const fn new_mistral3() -> Self {
        Self {
            architecture: "mistral3",
            num_layers: 40,
            hidden_dim: 5120,
            ffn_dim: 14_336,
            n_heads_q: 32,
            n_heads_kv: 8,
            head_dim: 128,
            vocab_size: 131_072,
            rms_eps: 1e-5,
            rope_theta: 1_000_000.0,
            rope_scaling: None,
            sliding_window: None,
            num_experts: None,
            num_experts_per_tok: None,
        }
    }

    /// Creates a new Sparse `MoE` Mixtral recipe (`"mixtral"` with top-2 of 8 routing and 4096-token SWA).
    #[must_use]
    pub const fn new_mixtral() -> Self {
        Self {
            architecture: "mixtral",
            num_layers: 32,
            hidden_dim: 4096,
            ffn_dim: 14_336,
            n_heads_q: 32,
            n_heads_kv: 8,
            head_dim: 128,
            vocab_size: 32_000,
            rms_eps: 1e-5,
            rope_theta: 1_000_000.0,
            rope_scaling: None,
            sliding_window: Some(4096),
            num_experts: Some(8),
            num_experts_per_tok: Some(2),
        }
    }

    /// Configures the sliding window attention length.
    #[must_use]
    pub const fn with_sliding_window(mut self, window: Option<usize>) -> Self {
        self.sliding_window = window;
        self
    }

    /// Configures the Sparse `MoE` expert routing parameters.
    #[must_use]
    pub const fn with_experts(mut self, num_experts: usize, num_experts_per_tok: usize) -> Self {
        self.num_experts = Some(num_experts);
        self.num_experts_per_tok = Some(num_experts_per_tok);
        self
    }

    /// Configures the `RoPE` scaling parameters (`Linear`, `Llama3Smooth`, or `LongRope`).
    #[must_use]
    pub fn with_rope_scaling(mut self, scaling: Option<RopeScalingConfig>) -> Self {
        self.rope_scaling = scaling;
        self
    }

    /// Hydrates a [`MistralRecipe`] from parsed GGUF [`ModelMetadata`] and tensor descriptors.
    #[must_use]
    pub fn from_metadata(meta: &ModelMetadata) -> Self {
        let mut recipe = match meta.architecture() {
            "mistral3" => Self::new_mistral3(),
            "mixtral" => Self::new_mixtral(),
            _ => Self::new(),
        };

        recipe.hidden_dim = infer_hidden_dim(meta, recipe.hidden_dim);

        let probed_layers = probe_descriptor_layers(meta.tensors());
        if meta.layer_count() > 0 {
            recipe.num_layers = meta.layer_count();
        } else if probed_layers > 0 {
            recipe.num_layers = probed_layers;
        }

        if meta.head_count() > 0 {
            recipe.n_heads_q = meta.head_count();
        }
        if meta.head_count_kv() > 0 {
            recipe.n_heads_kv = meta.head_count_kv();
        }

        hydrate_attention_dims(&mut recipe, meta);
        hydrate_moe_and_ffn(&mut recipe, meta);

        if let Some(vocab) = meta
            .vocab_size()
            .filter(|&v| v > 0)
            .or_else(|| infer_vocab_rows(meta, recipe.hidden_dim))
        {
            recipe.vocab_size = vocab;
        }

        if let Some(eps) = meta.rms_norm_eps().filter(|v| v.is_finite() && *v > 0.0) {
            recipe.rms_eps = eps;
        }
        if let Some(theta) = meta.rope_freq_base().filter(|v| v.is_finite() && *v > 0.0) {
            recipe.rope_theta = theta;
        }

        // GGUF omits `{arch}.attention.sliding_window` or encodes `0` when SWA is disabled.
        recipe.sliding_window = meta.sliding_window().filter(|&w| w > 0);

        recipe
    }

    /// Hydrates a [`MistralRecipe`] from a parsed [`GgufFile`], reading `RoPE` scaling metadata
    /// (`linear`, `yarn`, `llama3`) alongside GGUF hyperparameters and tensor descriptors.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] if `gguf.architecture()` is not `"mistral"`,
    /// `"mistral3"`, or `"mixtral"`, or if `rope.scaling.*` metadata is malformed.
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Result<Self, K3Error> {
        let arch = gguf.architecture();
        if !matches!(arch, "mistral" | "mistral3" | "mixtral") {
            return Err(K3Error::CapabilityMismatch {
                architecture: arch.to_owned(),
                reason: format!(
                    "MistralRecipe supports 'mistral', 'mistral3', and 'mixtral', got '{arch}'"
                ),
            });
        }

        let mut recipe = Self::from_metadata(gguf.metadata());
        if let Some(vocab) = gguf_fallback_vocab(gguf, &recipe) {
            recipe.vocab_size = vocab;
        }
        recipe.rope_scaling = parse_gguf_rope_scaling(gguf, arch)?;

        if recipe.vocab_size > 0 {
            for (label, key) in [
                ("bos_token_id", "tokenizer.ggml.bos_token_id"),
                ("eos_token_id", "tokenizer.ggml.eos_token_id"),
                ("eot_token_id", "tokenizer.ggml.eot_token_id"),
                ("padding_token_id", "tokenizer.ggml.padding_token_id"),
            ] {
                if let Some(tok_id) = gguf.metadata_value(key).and_then(GgufValue::as_u32) {
                    validate_token_id_in_vocab(arch, label, tok_id, recipe.vocab_size)?;
                }
            }
        }

        Ok(recipe)
    }

    /// Pre-decodes 1D `RMSNorm` scales (`attn_norm`, `ffn_norm`, `output_norm`) and populates
    /// `128-byte`-aligned dense projections (`Q/K/V/O`, dense `Gate/Up/Down`, or `MoE` router)
    /// in `cache` while keeping stacked `3D` `_exps` tensors zero-copy.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error`] if geometry is invalid or any required layer tensor fails verification.
    pub fn warm_layer_cache(
        &self,
        weights: &dyn ModelWeights,
        cache: &mut MistralLayerCache,
    ) -> Result<(), K3Error> {
        let geom = self.effective_geometry()?;
        cache.bind_weights(weights)?;
        let _ = cache.get_or_decode_norm(weights, "output_norm.weight", geom.hidden_dim)?;

        for layer in 0..geom.num_layers {
            let _ = cache.get_or_decode_norm(
                weights,
                &format!("blk.{layer}.attn_norm.weight"),
                geom.hidden_dim,
            )?;
            let _ = cache.get_or_repack_dense(
                weights,
                &format!("blk.{layer}.attn_q.weight"),
                geom.total_q,
                geom.hidden_dim,
            )?;
            let _ = cache.get_or_repack_dense(
                weights,
                &format!("blk.{layer}.attn_k.weight"),
                geom.total_kv,
                geom.hidden_dim,
            )?;
            let _ = cache.get_or_repack_dense(
                weights,
                &format!("blk.{layer}.attn_v.weight"),
                geom.total_kv,
                geom.hidden_dim,
            )?;
            let _ = cache.get_or_repack_dense(
                weights,
                &format!("blk.{layer}.attn_output.weight"),
                geom.hidden_dim,
                geom.total_q,
            )?;
            let _ = cache.get_or_decode_norm(
                weights,
                &format!("blk.{layer}.ffn_norm.weight"),
                geom.hidden_dim,
            )?;

            if let Some((n_exp, _top_k)) = geom.moe {
                let _ = cache.get_or_repack_dense(
                    weights,
                    &format!("blk.{layer}.ffn_gate_inp.weight"),
                    n_exp,
                    geom.hidden_dim,
                )?;
            } else {
                let _ = cache.get_or_repack_dense(
                    weights,
                    &format!("blk.{layer}.ffn_gate.weight"),
                    geom.ffn_dim,
                    geom.hidden_dim,
                )?;
                let _ = cache.get_or_repack_dense(
                    weights,
                    &format!("blk.{layer}.ffn_up.weight"),
                    geom.ffn_dim,
                    geom.hidden_dim,
                )?;
                let _ = cache.get_or_repack_dense(
                    weights,
                    &format!("blk.{layer}.ffn_down.weight"),
                    geom.hidden_dim,
                    geom.ffn_dim,
                )?;
            }
        }
        Ok(())
    }

    /// Executes prefill forward pass using a caller-owned [`MistralLayerCache`].
    ///
    /// # Errors
    ///
    /// Returns [`K3Error`] on invalid context bounds, missing tensors, or kernel errors.
    pub fn prefill_step_with_cache(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
        cache: &mut MistralLayerCache,
    ) -> Result<(), K3Error> {
        let prompt_tokens = ctx.prompt_tokens().to_vec();
        let prompt_len = prompt_tokens.len();
        let start_pos = ctx.start_position();
        if weights.tensor_count() == 0 {
            return self.prefill_step(ctx, weights);
        }
        if prompt_tokens.is_empty() {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture_id().to_owned(),
                reason: "prefill_step requires at least one prompt token".to_owned(),
            });
        }
        let seq_len = start_pos
            .checked_add(prompt_len)
            .ok_or(K3Error::ContextLengthExceeded {
                requested_tokens: usize::MAX,
                max_context_tokens: ctx.n_ctx(),
            })?;
        if seq_len > ctx.n_ctx() {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: seq_len,
                max_context_tokens: ctx.n_ctx(),
            });
        }
        let geom = self.effective_geometry()?;
        let kv_geom = ctx.kv_geometry();
        let compute_logits = ctx.compute_logits();
        let mut act_scratch = MistralActScratch::new();
        let last_hidden = self.execute_real_prefill_batch_with_cache(
            PrefillBatchRequest {
                geom,
                tokens: &prompt_tokens,
                start_pos,
                seq_len,
                compute_logits,
            },
            ctx.kv_cache_mut(),
            kv_geom,
            weights,
            cache,
            &mut act_scratch,
        )?;
        if compute_logits {
            let final_hidden =
                last_hidden
                    .as_deref()
                    .ok_or_else(|| K3Error::CapabilityMismatch {
                        architecture: self.architecture_id().to_owned(),
                        reason: "prefill missing final token hidden state for logit projection"
                            .to_owned(),
                    })?;
            project_lm_head(final_hidden, geom, ctx.logits_mut(), weights)?;
        }
        ctx.advance_position(prompt_len)?;
        Ok(())
    }

    /// Executes single-token decode step using a caller-owned [`MistralLayerCache`].
    ///
    /// # Errors
    ///
    /// Returns [`K3Error`] if no decode token is staged, context bounds are exceeded, or a tensor
    /// fails validation.
    pub fn decode_step_with_cache(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
        cache: &mut MistralLayerCache,
    ) -> Result<(), K3Error> {
        if weights.tensor_count() == 0 {
            return self.decode_step(ctx, weights);
        }
        let token = ctx
            .decode_token()
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: self.architecture_id().to_owned(),
                reason: "decode_step called without a staged decode token".to_owned(),
            })?;
        let pos = ctx.position();
        let seq_len = pos + 1;
        if seq_len > ctx.n_ctx() {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: seq_len,
                max_context_tokens: ctx.n_ctx(),
            });
        }
        let geom = self.effective_geometry()?;
        let kv_geom = ctx.kv_geometry();
        let mut act_scratch = MistralActScratch::new();
        let hidden = self.execute_real_token_step_with_cache(
            geom,
            token,
            pos,
            seq_len,
            ctx.kv_cache_mut(),
            kv_geom,
            weights,
            cache,
            &mut act_scratch,
        )?;
        project_lm_head(&hidden, geom, ctx.logits_mut(), weights)?;
        ctx.advance_position(1)?;
        Ok(())
    }

    fn effective_geometry(&self) -> Result<MistralGeometry, K3Error> {
        if self.num_layers == 0
            || self.hidden_dim == 0
            || self.ffn_dim == 0
            || self.n_heads_q == 0
            || self.n_heads_kv == 0
            || self.head_dim == 0
            || self.head_dim % 2 != 0
            || self.n_heads_q % self.n_heads_kv != 0
            || !self.rms_eps.is_finite()
            || self.rms_eps <= 0.0
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture_id().to_owned(),
                reason: format!(
                    "invalid Mistral geometry: layers={}, hidden={}, ffn={}, nq={}, nkv={}, head_dim={}",
                    self.num_layers,
                    self.hidden_dim,
                    self.ffn_dim,
                    self.n_heads_q,
                    self.n_heads_kv,
                    self.head_dim
                ),
            });
        }
        if self.sliding_window == Some(0) {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture_id().to_owned(),
                reason: "sliding_window must be None or >= 1".to_owned(),
            });
        }
        let moe = match self.num_experts {
            None => None,
            Some(0) => {
                return Err(K3Error::CapabilityMismatch {
                    architecture: self.architecture_id().to_owned(),
                    reason: "num_experts must be >= 1 when enabled".to_owned(),
                });
            }
            Some(n_exp) => {
                let top_k = self.num_experts_per_tok.unwrap_or(2);
                if top_k == 0 || top_k > n_exp {
                    return Err(K3Error::CapabilityMismatch {
                        architecture: self.architecture_id().to_owned(),
                        reason: format!("num_experts_per_tok ({top_k}) must be in 1..={n_exp}"),
                    });
                }
                Some((n_exp, top_k))
            }
        };

        Ok(MistralGeometry {
            num_layers: self.num_layers,
            hidden_dim: self.hidden_dim,
            ffn_dim: self.ffn_dim,
            n_heads_q: self.n_heads_q,
            n_heads_kv: self.n_heads_kv,
            head_dim: self.head_dim,
            total_q: self.n_heads_q * self.head_dim,
            total_kv: self.n_heads_kv * self.head_dim,
            vocab_size: self.vocab_size,
            rms_eps: self.rms_eps,
            rope_theta: self.rope_theta,
            sliding_window: self.sliding_window,
            moe,
        })
    }

    fn execute_real_prefill_batch_with_cache(
        &self,
        req: PrefillBatchRequest<'_>,
        kv_cache: &mut [f32],
        kv_geom_override: Option<KvGeometry>,
        weights: &dyn ModelWeights,
        cache: &mut MistralLayerCache,
        act_scratch: &mut MistralActScratch,
    ) -> Result<Option<Vec<f32>>, K3Error> {
        let m = req.tokens.len();
        let geom = req.geom;
        let mut hidden =
            embed_real_tokens_batch(req.tokens, geom.hidden_dim, geom.vocab_size, weights)?;
        let inv_freq = rope_inv_freq_compat(
            geom.head_dim,
            geom.rope_theta,
            self.rope_scaling.as_ref(),
            req.seq_len,
        )?;
        let mut kv_view = build_real_kv_view(geom, req.seq_len, kv_cache, kv_geom_override)?;
        let mut bufs = LayerPrefillBuffers::new(geom, m, req.seq_len)?;

        #[expect(clippy::cast_precision_loss, reason = "head_dim fits cleanly in f32")]
        let attn_scale = 1.0_f32 / (geom.head_dim as f32).sqrt();
        let batch_ctx = PrefillBatchCtx {
            geom,
            m,
            start_pos: req.start_pos,
            seq_len: req.seq_len,
            attn_scale,
            inv_freq: &inv_freq,
            weights,
        };

        for layer in 0..geom.num_layers {
            execute_layer_attention_prefill_cached(
                &batch_ctx,
                layer,
                &mut kv_view,
                &mut bufs,
                &mut hidden,
                cache,
                act_scratch,
            )?;

            apply_layer_rmsnorm_batch_cached(
                &format!("blk.{layer}.ffn_norm.weight"),
                &hidden,
                &mut bufs.ffn_norm,
                m,
                geom.hidden_dim,
                geom.rms_eps,
                weights,
                cache,
            )?;

            if let Some((n_exp, top_k)) = geom.moe {
                execute_moe_ffn_prefill_cached(
                    &batch_ctx,
                    layer,
                    n_exp,
                    top_k,
                    &bufs.ffn_norm,
                    &mut hidden,
                    cache,
                    act_scratch,
                )?;
            } else {
                execute_dense_ffn_prefill_cached(
                    &batch_ctx,
                    layer,
                    &bufs.ffn_norm,
                    &mut hidden,
                    cache,
                    act_scratch,
                )?;
            }
        }

        if !req.compute_logits {
            return Ok(None);
        }

        let mut final_hidden = vec![0.0_f32; geom.hidden_dim];
        let last_row = &hidden[(m - 1) * geom.hidden_dim..m * geom.hidden_dim];
        apply_layer_rmsnorm_cached(
            "output_norm.weight",
            last_row,
            &mut final_hidden,
            geom.rms_eps,
            weights,
            cache,
        )?;
        Ok(Some(final_hidden))
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "internal single-token step coordinates geometry, KV cache, weights, and layer cache"
    )]
    fn execute_real_token_step_with_cache(
        &self,
        geom: MistralGeometry,
        token_id: u32,
        pos: usize,
        seq_len: usize,
        kv_cache: &mut [f32],
        kv_geom_override: Option<KvGeometry>,
        weights: &dyn ModelWeights,
        cache: &mut MistralLayerCache,
        act_scratch: &mut MistralActScratch,
    ) -> Result<Vec<f32>, K3Error> {
        let mut hidden = embed_real_token(token_id, geom.hidden_dim, geom.vocab_size, weights)?;
        let inv_freq = rope_inv_freq_compat(
            geom.head_dim,
            geom.rope_theta,
            self.rope_scaling.as_ref(),
            seq_len,
        )?;

        let mut kv_view = build_real_kv_view(geom, seq_len, kv_cache, kv_geom_override)?;
        let mut bufs = LayerStepBuffers::new(geom);

        #[expect(clippy::cast_precision_loss, reason = "head_dim fits cleanly in f32")]
        let attn_scale = 1.0_f32 / (geom.head_dim as f32).sqrt();
        let step_ctx = CachedLayerStepCtx {
            geom,
            pos,
            attn_scale,
            inv_freq: &inv_freq,
            weights,
        };

        for layer in 0..geom.num_layers {
            execute_layer_attention_cached(
                &step_ctx,
                layer,
                &mut kv_view,
                &mut bufs,
                &mut hidden,
                cache,
                act_scratch,
            )?;

            apply_layer_rmsnorm_cached(
                &format!("blk.{layer}.ffn_norm.weight"),
                &hidden,
                &mut bufs.ffn_norm,
                geom.rms_eps,
                weights,
                cache,
            )?;

            if let Some((n_exp, top_k)) = geom.moe {
                execute_moe_ffn_cached(
                    layer,
                    geom,
                    n_exp,
                    top_k,
                    &bufs.ffn_norm,
                    &mut hidden,
                    weights,
                    cache,
                )?;
            } else {
                execute_dense_ffn_cached(
                    layer,
                    geom.ffn_dim,
                    &bufs.ffn_norm,
                    &mut hidden,
                    weights,
                    cache,
                    act_scratch,
                )?;
            }
        }

        let mut final_hidden = vec![0.0_f32; geom.hidden_dim];
        apply_layer_rmsnorm_cached(
            "output_norm.weight",
            &hidden,
            &mut final_hidden,
            geom.rms_eps,
            weights,
            cache,
        )?;

        Ok(final_hidden)
    }

    /// Executes deterministic synthetic token step ONLY when `weights.tensor_count() == 0`
    /// (preserves unit test mocks without weight backing buffers).
    fn execute_synthetic_token_step(
        &self,
        token_id: u32,
        pos: usize,
        kv_cache: &mut [f32],
    ) -> Result<Vec<f32>, K3Error> {
        let hidden_dim = self.hidden_dim;
        #[expect(clippy::cast_precision_loss, reason = "token_id modulo 17 fits in f32")]
        let base_val = ((token_id % 17) as f32) * 0.1 + 0.5;
        let mut hidden = vec![base_val; hidden_dim];
        let total_kv = self.n_heads_kv * self.head_dim;
        let inv_freq = rope_inv_freq_compat(
            self.head_dim,
            self.rope_theta,
            self.rope_scaling.as_ref(),
            pos + 1,
        )?;
        let mut k_vec = vec![0.01_f32; total_kv];
        let v_vec = vec![0.01_f32; total_kv];
        apply_norm_rope_inplace(&mut k_vec, self.head_dim, pos, &inv_freq)?;

        let eff_window = self.sliding_window.unwrap_or(pos + 1).max(1);
        let cache_len = kv_cache.len();
        if cache_len >= 2 * total_kv {
            let slot = (pos % eff_window) % (cache_len / (2 * total_kv));
            let k_off = slot * total_kv;
            let v_off = k_off + total_kv;
            if v_off + total_kv <= cache_len {
                kv_cache[k_off..k_off + total_kv].copy_from_slice(&k_vec);
                kv_cache[v_off..v_off + total_kv].copy_from_slice(&v_vec);
            }
        }
        if !hidden.is_empty() {
            hidden[0] += 0.01;
        }
        Ok(hidden)
    }
}

impl ArchRecipe for MistralRecipe {
    fn architecture_id(&self) -> &'static str {
        self.architecture
    }

    fn target_layouts(&self) -> &'static [&'static str] {
        &["native", "ime_tile_8x16"]
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        matches!(
            quant,
            QuantType::F32
                | QuantType::F16
                | QuantType::BF16
                | QuantType::Q4_0
                | QuantType::Q4_K_M
                | QuantType::Q5_K
                | QuantType::Q6_K
                | QuantType::Q8_0
        )
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        if weights.tensor_count() > 0 {
            return with_shared_model_cache(weights, |cache| {
                self.prefill_step_with_cache(ctx, weights, cache)
            });
        }

        let prompt_tokens = ctx.prompt_tokens().to_vec();
        let prompt_len = prompt_tokens.len();
        let start_pos = ctx.position();
        let mut last_hidden = Vec::new();
        {
            let kv_cache = ctx.kv_cache_mut();
            for (i, &tok) in prompt_tokens.iter().enumerate() {
                last_hidden = self.execute_synthetic_token_step(tok, start_pos + i, kv_cache)?;
            }
        }
        let compute_logits = ctx.compute_logits();
        ctx.advance_position(prompt_len)?;
        if compute_logits {
            let logits = ctx.logits_mut();
            if !logits.is_empty() {
                logits.fill(0.0);
                let bias = (last_hidden.first().copied().unwrap_or(0.0).abs() % 0.25) * 0.01;
                let target_slot = if logits.len() > 9 { 9 } else { 0 };
                logits[target_slot] = 1.0 + bias;
            }
        }
        Ok(())
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        if weights.tensor_count() > 0 {
            return with_shared_model_cache(weights, |cache| {
                self.decode_step_with_cache(ctx, weights, cache)
            });
        }

        let token = ctx
            .decode_token()
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: self.architecture_id().to_owned(),
                reason: "decode_step called without a staged decode token".to_owned(),
            })?;

        let pos = ctx.position();
        let hidden = self.execute_synthetic_token_step(token, pos, ctx.kv_cache_mut())?;
        ctx.advance_position(1)?;
        let logits = ctx.logits_mut();
        if !logits.is_empty() {
            logits.fill(0.0);
            let bias = (hidden.first().copied().unwrap_or(0.0).abs() % 0.25) * 0.01;
            let next_slot = (usize::try_from(token).unwrap_or(0) + 1) % logits.len();
            logits[next_slot] = 2.0 + bias;
        }
        Ok(())
    }
}

fn infer_hidden_dim(meta: &ModelMetadata, default_hidden: usize) -> usize {
    if meta.embedding_dim() > 0 {
        return meta.embedding_dim();
    }
    meta.find_tensor("blk.0.attn_norm.weight")
        .or_else(|| meta.find_tensor("output_norm.weight"))
        .map_or(default_hidden, TensorDescriptor::num_elements)
}

fn hydrate_attention_dims(recipe: &mut MistralRecipe, meta: &ModelMetadata) {
    if let Some(hd) = meta
        .key_length()
        .or_else(|| meta.rope_dimension_count())
        .filter(|&d| d >= 2)
    {
        recipe.head_dim = hd;
    } else if let Some(q_desc) = meta.find_tensor("blk.0.attn_q.weight")
        && let Some(q_rows) = infer_descriptor_rows(q_desc, recipe.hidden_dim)
        && recipe.n_heads_q > 0
        && q_rows % recipe.n_heads_q == 0
    {
        recipe.head_dim = q_rows / recipe.n_heads_q;
    } else if recipe.n_heads_q > 0 && recipe.hidden_dim % recipe.n_heads_q == 0 {
        recipe.head_dim = recipe.hidden_dim / recipe.n_heads_q;
    }

    if meta.head_count_kv() == 0
        && let Some(k_desc) = meta.find_tensor("blk.0.attn_k.weight")
        && let Some(k_rows) = infer_descriptor_rows(k_desc, recipe.hidden_dim)
        && recipe.head_dim > 0
        && k_rows % recipe.head_dim == 0
    {
        recipe.n_heads_kv = k_rows / recipe.head_dim;
    }
}

fn hydrate_moe_and_ffn(recipe: &mut MistralRecipe, meta: &ModelMetadata) {
    if let Some(ec) = meta.expert_count() {
        recipe.num_experts = (ec > 0).then_some(ec);
    }
    if let Some(used) = meta.expert_used_count() {
        recipe.num_experts_per_tok = (used > 0).then_some(used);
    }
    if recipe.num_experts.is_none() {
        recipe.num_experts_per_tok = None;
    } else if recipe.num_experts_per_tok.is_none() {
        recipe.num_experts_per_tok = Some(2);
    }

    let explicit_ffn = if recipe.num_experts.is_some() {
        meta.expert_feed_forward_length()
            .or_else(|| meta.feed_forward_length())
    } else {
        meta.feed_forward_length()
    };
    if let Some(ffn) = explicit_ffn.filter(|&f| f > 0) {
        recipe.ffn_dim = ffn;
        return;
    }

    let inferred = meta
        .find_tensor("blk.0.ffn_gate.weight")
        .or_else(|| meta.find_tensor("blk.0.ffn_up.weight"))
        .or_else(|| meta.find_tensor("blk.0.ffn_down.weight"))
        .or_else(|| meta.find_tensor("blk.0.ffn_gate.0.weight"))
        .or_else(|| meta.find_tensor("blk.0.ffn_gate_exps.0.weight"))
        .and_then(|d| infer_descriptor_rows(d, recipe.hidden_dim))
        .or_else(|| {
            meta.find_tensor("blk.0.ffn_gate_exps.weight")
                .and_then(|d| {
                    infer_3d_expert_ffn_dim(d.shape(), recipe.hidden_dim, recipe.num_experts)
                })
        });
    if let Some(ffn) = inferred.filter(|&f| f > 0) {
        recipe.ffn_dim = ffn;
    }
}

fn infer_3d_expert_ffn_dim(
    shape: &[usize],
    hidden_dim: usize,
    num_experts: Option<usize>,
) -> Option<usize> {
    if shape.len() != 3 {
        return None;
    }
    let n_exp = num_experts?;
    if (shape[2] == n_exp && shape[0] == hidden_dim)
        || (shape[0] == n_exp && shape[2] == hidden_dim)
    {
        Some(shape[1])
    } else {
        None
    }
}

fn infer_vocab_rows(meta: &ModelMetadata, hidden_dim: usize) -> Option<usize> {
    meta.find_tensor("output.weight")
        .or_else(|| meta.find_tensor("token_embd.weight"))
        .and_then(|desc| infer_descriptor_rows(desc, hidden_dim))
}

fn gguf_fallback_vocab(gguf: &GgufFile<'_>, recipe: &MistralRecipe) -> Option<usize> {
    let arch = gguf.architecture();
    let vocab_key = format!("{arch}.vocab_size");
    if let Some(v) = gguf
        .metadata_value(&vocab_key)
        .and_then(GgufValue::as_usize)
        .filter(|&v| v > 0)
    {
        return Some(v);
    }
    if gguf.metadata().vocab_size().is_none()
        && infer_vocab_rows(gguf.metadata(), recipe.hidden_dim).is_none()
        && let Some(GgufValue::Array(arr)) = gguf.metadata_value("tokenizer.ggml.tokens")
    {
        let len = gguf_array_len(arr);
        if len > 0 {
            return Some(len);
        }
    }
    None
}

fn parse_gguf_rope_scaling(
    gguf: &GgufFile<'_>,
    arch: &str,
) -> Result<Option<RopeScalingConfig>, K3Error> {
    let type_key = format!("{arch}.rope.scaling.type");
    let Some(scale_type) = gguf.metadata_value(&type_key).and_then(GgufValue::as_str) else {
        return Ok(None);
    };

    let factor_key = format!("{arch}.rope.scaling.factor");
    let factor = gguf
        .metadata_value(&factor_key)
        .and_then(GgufValue::as_f32)
        .filter(|f| f.is_finite() && *f > 0.0)
        .ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: format!("'{type_key}' requires positive finite '{factor_key}'"),
        })?;

    match scale_type {
        "linear" => Ok(Some(RopeScalingConfig::Linear { factor })),
        "llama3" => {
            let low = read_positive_f32(gguf, &format!("{arch}.rope.scaling.low_freq_factor"))
                .unwrap_or(1.0);
            let high = read_positive_f32(gguf, &format!("{arch}.rope.scaling.high_freq_factor"))
                .unwrap_or(4.0);
            let orig_ctx = gguf
                .metadata_value(&format!("{arch}.rope.scaling.original_context_length"))
                .and_then(GgufValue::as_usize)
                .filter(|&c| c > 0)
                .unwrap_or(8192);
            Ok(Some(RopeScalingConfig::Llama3Smooth {
                factor,
                low_freq_factor: low,
                high_freq_factor: high,
                original_max_position_embeddings: orig_ctx,
            }))
        }
        "yarn" => {
            let orig_ctx = gguf
                .metadata_value(&format!("{arch}.rope.scaling.original_context_length"))
                .and_then(GgufValue::as_usize)
                .filter(|&c| c > 0);
            if let Some(ctx) = orig_ctx {
                let beta_slow = read_positive_f32(gguf, &format!("{arch}.rope.scaling.beta_slow"))
                    .unwrap_or(1.0);
                let beta_fast = read_positive_f32(gguf, &format!("{arch}.rope.scaling.beta_fast"))
                    .unwrap_or(32.0);
                Ok(Some(RopeScalingConfig::Llama3Smooth {
                    factor,
                    low_freq_factor: beta_slow,
                    high_freq_factor: beta_fast,
                    original_max_position_embeddings: ctx,
                }))
            } else {
                Ok(Some(RopeScalingConfig::Linear { factor }))
            }
        }
        other => Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: format!("unsupported rope.scaling.type '{other}'"),
        }),
    }
}

fn read_positive_f32(gguf: &GgufFile<'_>, key: &str) -> Option<f32> {
    gguf.metadata_value(key)
        .and_then(GgufValue::as_f32)
        .filter(|v| v.is_finite() && *v > 0.0)
}

fn embed_real_tokens_batch(
    token_ids: &[u32],
    hidden_dim: usize,
    vocab_size: usize,
    weights: &dyn ModelWeights,
) -> Result<Vec<f32>, K3Error> {
    let emb_tensor = weights.tensor("token_embd.weight")?;
    let _ = resolve_2d_dims(&emb_tensor, vocab_size, hidden_dim)?;
    let total_elems = token_ids.len().checked_mul(hidden_dim).ok_or_else(|| {
        tensor_bounds_err(&emb_tensor, 0, "batched embedding allocation size overflow")
    })?;
    let mut hidden = vec![0.0_f32; total_elems];
    for (t, &token_id) in token_ids.iter().enumerate() {
        let row_idx = usize::try_from(token_id).map_err(|_| {
            tensor_bounds_err(
                &emb_tensor,
                0,
                &format!("token_id {token_id} overflows usize"),
            )
        })?;
        let dst = &mut hidden[t * hidden_dim..(t + 1) * hidden_dim];
        dequantize_embedding_row(&emb_tensor, row_idx, dst)?;
    }
    Ok(hidden)
}

fn embed_real_token(
    token_id: u32,
    hidden_dim: usize,
    vocab_size: usize,
    weights: &dyn ModelWeights,
) -> Result<Vec<f32>, K3Error> {
    embed_real_tokens_batch(&[token_id], hidden_dim, vocab_size, weights)
}

fn build_real_kv_view(
    geom: MistralGeometry,
    seq_len: usize,
    kv_cache: &mut [f32],
    kv_geom_override: Option<KvGeometry>,
) -> Result<KvCacheView<'_>, K3Error> {
    let per_tok_all_layers = 2_usize
        .checked_mul(geom.num_layers)
        .and_then(|v| v.checked_mul(geom.total_kv))
        .ok_or(K3Error::ContextLengthExceeded {
            requested_tokens: seq_len,
            max_context_tokens: 0,
        })?;

    let kv_geom = if let Some(ctx_geom) = kv_geom_override {
        if ctx_geom.num_layers != geom.num_layers
            || ctx_geom.n_heads_kv != geom.n_heads_kv
            || ctx_geom.head_dim != geom.head_dim
            || (ctx_geom.sliding_window.is_some() && ctx_geom.sliding_window != geom.sliding_window)
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: "mistral".to_owned(),
                reason: format!(
                    "ExecutionContext KvGeometry mismatch with MistralRecipe (ctx={ctx_geom:?}, expected layers={}, nkv={}, head_dim={}, swa={:?})",
                    geom.num_layers, geom.n_heads_kv, geom.head_dim, geom.sliding_window
                ),
            });
        }
        KvGeometry::new(
            geom.num_layers,
            geom.n_heads_kv,
            geom.head_dim,
            ctx_geom.seq_capacity,
            ctx_geom.sliding_window.or(geom.sliding_window),
        )?
    } else {
        let seq_cap = kv_cache.len() / per_tok_all_layers.max(1);
        if seq_cap < seq_len {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: seq_len,
                max_context_tokens: seq_cap,
            });
        }
        KvGeometry::new(
            geom.num_layers,
            geom.n_heads_kv,
            geom.head_dim,
            seq_cap,
            geom.sliding_window,
        )?
    };

    KvCacheView::new(kv_cache, kv_geom)
}

#[expect(
    clippy::too_many_arguments,
    reason = "batched RMSNorm over [m, hidden_dim] reuses one decoded norm weight lookup per layer"
)]
fn apply_layer_rmsnorm_batch_cached(
    tensor_name: &str,
    input: &[f32],
    out: &mut [f32],
    m: usize,
    hidden_dim: usize,
    eps: f32,
    weights: &dyn ModelWeights,
    cache: &mut MistralLayerCache,
) -> Result<(), K3Error> {
    let expected = m
        .checked_mul(hidden_dim)
        .ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: "mistral".to_owned(),
            reason: format!("batched RMSNorm size overflow for '{tensor_name}'"),
        })?;
    if input.len() != expected || out.len() != expected {
        return Err(K3Error::CapabilityMismatch {
            architecture: "mistral".to_owned(),
            reason: format!(
                "batched RMSNorm buffer length mismatch for '{tensor_name}': in={}, out={}, expected={expected}",
                input.len(),
                out.len()
            ),
        });
    }
    let norm_weights = cache.get_or_decode_norm(weights, tensor_name, hidden_dim)?;
    for t in 0..m {
        let range = t * hidden_dim..(t + 1) * hidden_dim;
        rmsnorm_compat(&mut out[range.clone()], &input[range], &norm_weights, eps)?;
    }
    Ok(())
}

fn apply_layer_rmsnorm_cached(
    tensor_name: &str,
    input: &[f32],
    out: &mut [f32],
    eps: f32,
    weights: &dyn ModelWeights,
    cache: &mut MistralLayerCache,
) -> Result<(), K3Error> {
    apply_layer_rmsnorm_batch_cached(tensor_name, input, out, 1, input.len(), eps, weights, cache)
}

fn execute_dense_ffn_cached(
    layer: usize,
    ffn_dim: usize,
    ffn_norm: &[f32],
    hidden: &mut [f32],
    weights: &dyn ModelWeights,
    cache: &mut MistralLayerCache,
    act_scratch: &mut MistralActScratch,
) -> Result<(), K3Error> {
    let mut gate = vec![0.0_f32; ffn_dim];
    let mut up = vec![0.0_f32; ffn_dim];
    let mut act = vec![0.0_f32; ffn_dim];
    let mut down = vec![0.0_f32; hidden.len()];

    act_scratch.prepare_row(ffn_norm)?;
    cache.matvec_dense_cached(
        weights,
        &format!("blk.{layer}.ffn_gate.weight"),
        &mut gate,
        ffn_norm,
        Some(act_scratch),
    )?;
    cache.matvec_dense_cached(
        weights,
        &format!("blk.{layer}.ffn_up.weight"),
        &mut up,
        ffn_norm,
        Some(act_scratch),
    )?;
    swiglu_compat(&mut act, &gate, &up)?;
    cache.matvec_dense_cached(
        weights,
        &format!("blk.{layer}.ffn_down.weight"),
        &mut down,
        &act,
        None,
    )?;

    for (h, &d) in hidden.iter_mut().zip(&down) {
        *h += d;
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "coordinates layer geometry, top-k expert slicing, and cached router dispatch"
)]
fn execute_moe_ffn_cached(
    layer: usize,
    geom: MistralGeometry,
    n_exp: usize,
    top_k: usize,
    ffn_norm: &[f32],
    hidden: &mut [f32],
    weights: &dyn ModelWeights,
    cache: &mut MistralLayerCache,
) -> Result<(), K3Error> {
    let mut router_logits = vec![0.0_f32; n_exp];
    cache.matvec_dense_cached(
        weights,
        &format!("blk.{layer}.ffn_gate_inp.weight"),
        &mut router_logits,
        ffn_norm,
        None,
    )?;
    let selected = route_topk_experts(&router_logits, top_k)?;

    let mut gate = vec![0.0_f32; geom.ffn_dim];
    let mut up = vec![0.0_f32; geom.ffn_dim];
    let mut act = vec![0.0_f32; geom.ffn_dim];
    let mut down = vec![0.0_f32; geom.hidden_dim];

    for (expert_idx, gate_weight) in selected {
        let g_w = cache.slice_expert_projection(
            weights,
            layer,
            "gate",
            expert_idx,
            n_exp,
            geom.ffn_dim,
            geom.hidden_dim,
        )?;
        let u_w = cache.slice_expert_projection(
            weights,
            layer,
            "up",
            expert_idx,
            n_exp,
            geom.ffn_dim,
            geom.hidden_dim,
        )?;
        let d_w = cache.slice_expert_projection(
            weights,
            layer,
            "down",
            expert_idx,
            n_exp,
            geom.hidden_dim,
            geom.ffn_dim,
        )?;

        matvec_tensor(&mut gate, &g_w, ffn_norm)?;
        matvec_tensor(&mut up, &u_w, ffn_norm)?;
        swiglu_compat(&mut act, &gate, &up)?;
        matvec_tensor(&mut down, &d_w, &act)?;

        for (h, &delta) in hidden.iter_mut().zip(&down) {
            *h += gate_weight * delta;
        }
    }
    Ok(())
}

fn project_lm_head(
    final_hidden: &[f32],
    geom: MistralGeometry,
    logits: &mut [f32],
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    let head = if weights.contains_tensor("output.weight") {
        weights.tensor("output.weight")?
    } else {
        weights.tensor("token_embd.weight")?
    };

    let vocab = if geom.vocab_size > 0 {
        geom.vocab_size
    } else if geom.hidden_dim > 0 && head.num_elements() % geom.hidden_dim == 0 {
        head.num_elements() / geom.hidden_dim
    } else {
        return Err(tensor_bounds_err(
            &head,
            0,
            "unable to determine LM head vocabulary rows",
        ));
    };

    if logits.len() < vocab {
        return Err(K3Error::TensorBoundsError {
            tensor_name: head.name().to_owned(),
            offset: 0,
            byte_len: logits.len(),
            buffer_len: vocab,
            reason: format!(
                "logits buffer length {} is smaller than vocabulary size {vocab}",
                logits.len()
            ),
        });
    }

    matvec_tensor(&mut logits[..vocab], &head, final_hidden)?;
    if logits.len() > vocab {
        logits[vocab..].fill(0.0);
    }
    Ok(())
}

fn probe_descriptor_layers(tensors: &[TensorDescriptor]) -> usize {
    let mut max_layer = 0_usize;
    for desc in tensors {
        if let Some(rest) = desc.name().strip_prefix("blk.")
            && let Some((idx_str, _)) = rest.split_once('.')
            && let Ok(idx) = idx_str.parse::<usize>()
        {
            max_layer = max_layer.max(idx + 1);
        }
    }
    max_layer
}

fn infer_descriptor_rows(desc: &TensorDescriptor, expected_cols: usize) -> Option<usize> {
    let shape = desc.shape();
    if shape.len() != 2 || expected_cols == 0 {
        return None;
    }
    if shape[0] == expected_cols {
        Some(shape[1])
    } else if shape[1] == expected_cols {
        Some(shape[0])
    } else {
        None
    }
}

struct LayerStepBuffers {
    attn_norm: Vec<f32>,
    q_vec: Vec<f32>,
    k_vec: Vec<f32>,
    v_vec: Vec<f32>,
    attn_out: Vec<f32>,
    o_proj: Vec<f32>,
    ffn_norm: Vec<f32>,
}

impl LayerStepBuffers {
    fn new(geom: MistralGeometry) -> Self {
        Self {
            attn_norm: vec![0.0_f32; geom.hidden_dim],
            q_vec: vec![0.0_f32; geom.total_q],
            k_vec: vec![0.0_f32; geom.total_kv],
            v_vec: vec![0.0_f32; geom.total_kv],
            attn_out: vec![0.0_f32; geom.total_q],
            o_proj: vec![0.0_f32; geom.hidden_dim],
            ffn_norm: vec![0.0_f32; geom.hidden_dim],
        }
    }
}

struct CachedLayerStepCtx<'a> {
    geom: MistralGeometry,
    pos: usize,
    attn_scale: f32,
    inv_freq: &'a [f32],
    weights: &'a dyn ModelWeights,
}

fn execute_layer_attention_cached(
    ctx: &CachedLayerStepCtx<'_>,
    layer: usize,
    kv_view: &mut KvCacheView<'_>,
    bufs: &mut LayerStepBuffers,
    hidden: &mut [f32],
    cache: &mut MistralLayerCache,
    act_scratch: &mut MistralActScratch,
) -> Result<(), K3Error> {
    apply_layer_rmsnorm_cached(
        &format!("blk.{layer}.attn_norm.weight"),
        hidden,
        &mut bufs.attn_norm,
        ctx.geom.rms_eps,
        ctx.weights,
        cache,
    )?;

    act_scratch.prepare_row(&bufs.attn_norm)?;
    cache.matvec_dense_cached(
        ctx.weights,
        &format!("blk.{layer}.attn_q.weight"),
        &mut bufs.q_vec,
        &bufs.attn_norm,
        Some(act_scratch),
    )?;
    cache.matvec_dense_cached(
        ctx.weights,
        &format!("blk.{layer}.attn_k.weight"),
        &mut bufs.k_vec,
        &bufs.attn_norm,
        Some(act_scratch),
    )?;
    cache.matvec_dense_cached(
        ctx.weights,
        &format!("blk.{layer}.attn_v.weight"),
        &mut bufs.v_vec,
        &bufs.attn_norm,
        Some(act_scratch),
    )?;

    apply_norm_rope_inplace(&mut bufs.q_vec, ctx.geom.head_dim, ctx.pos, ctx.inv_freq)?;
    apply_norm_rope_inplace(&mut bufs.k_vec, ctx.geom.head_dim, ctx.pos, ctx.inv_freq)?;

    {
        let (k_slot, v_slot) = kv_view.write_pos_range(layer, ctx.pos)?;
        k_slot.copy_from_slice(&bufs.k_vec);
        v_slot.copy_from_slice(&bufs.v_vec);
    }
    let valid_range = kv_view.valid_token_range(ctx.pos + 1)?;
    let valid_len = valid_range.len();
    let (k_valid, v_valid) = kv_view.valid_kv_range(layer, ctx.pos + 1)?;
    gqa_attention_compat(
        &mut bufs.attn_out,
        &bufs.q_vec,
        k_valid,
        v_valid,
        ctx.geom.n_heads_q,
        ctx.geom.n_heads_kv,
        ctx.geom.head_dim,
        valid_len,
        ctx.attn_scale,
    )?;

    cache.matvec_dense_cached(
        ctx.weights,
        &format!("blk.{layer}.attn_output.weight"),
        &mut bufs.o_proj,
        &bufs.attn_out,
        None,
    )?;
    for (h, &delta) in hidden.iter_mut().zip(&bufs.o_proj) {
        *h += delta;
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct PrefillBatchRequest<'a> {
    geom: MistralGeometry,
    tokens: &'a [u32],
    start_pos: usize,
    seq_len: usize,
    compute_logits: bool,
}

struct PrefillBatchCtx<'a> {
    geom: MistralGeometry,
    m: usize,
    start_pos: usize,
    seq_len: usize,
    attn_scale: f32,
    inv_freq: &'a [f32],
    weights: &'a dyn ModelWeights,
}

struct LayerPrefillBuffers {
    attn_norm: Vec<f32>,
    q_batch: Vec<f32>,
    k_batch: Vec<f32>,
    v_batch: Vec<f32>,
    attn_out: Vec<f32>,
    o_proj: Vec<f32>,
    ffn_norm: Vec<f32>,
    scores_scratch: Vec<f32>,
}

impl LayerPrefillBuffers {
    fn new(geom: MistralGeometry, m: usize, seq_len: usize) -> Result<Self, K3Error> {
        let checked_mul = |dim: usize| {
            m.checked_mul(dim).ok_or(K3Error::ContextLengthExceeded {
                requested_tokens: seq_len,
                max_context_tokens: usize::MAX,
            })
        };
        let hidden_elems = checked_mul(geom.hidden_dim)?;
        let q_elems = checked_mul(geom.total_q)?;
        let kv_elems = checked_mul(geom.total_kv)?;
        let max_scores = geom
            .sliding_window
            .map_or(seq_len, |w| w.min(seq_len))
            .max(1);
        Ok(Self {
            attn_norm: vec![0.0_f32; hidden_elems],
            q_batch: vec![0.0_f32; q_elems],
            k_batch: vec![0.0_f32; kv_elems],
            v_batch: vec![0.0_f32; kv_elems],
            attn_out: vec![0.0_f32; q_elems],
            o_proj: vec![0.0_f32; hidden_elems],
            ffn_norm: vec![0.0_f32; hidden_elems],
            scores_scratch: vec![0.0_f32; max_scores],
        })
    }
}

fn execute_layer_attention_prefill_cached(
    ctx: &PrefillBatchCtx<'_>,
    layer: usize,
    kv_view: &mut KvCacheView<'_>,
    bufs: &mut LayerPrefillBuffers,
    hidden: &mut [f32],
    cache: &mut MistralLayerCache,
    act_scratch: &mut MistralActScratch,
) -> Result<(), K3Error> {
    apply_layer_rmsnorm_batch_cached(
        &format!("blk.{layer}.attn_norm.weight"),
        hidden,
        &mut bufs.attn_norm,
        ctx.m,
        ctx.geom.hidden_dim,
        ctx.geom.rms_eps,
        ctx.weights,
        cache,
    )?;

    act_scratch.prepare_batch(&bufs.attn_norm, ctx.m, ctx.geom.hidden_dim)?;
    cache.matmul_dense_cached(
        ctx.weights,
        &format!("blk.{layer}.attn_q.weight"),
        &mut bufs.q_batch,
        ctx.m,
        &bufs.attn_norm,
        Some(act_scratch),
    )?;
    cache.matmul_dense_cached(
        ctx.weights,
        &format!("blk.{layer}.attn_k.weight"),
        &mut bufs.k_batch,
        ctx.m,
        &bufs.attn_norm,
        Some(act_scratch),
    )?;
    cache.matmul_dense_cached(
        ctx.weights,
        &format!("blk.{layer}.attn_v.weight"),
        &mut bufs.v_batch,
        ctx.m,
        &bufs.attn_norm,
        Some(act_scratch),
    )?;

    for t in 0..ctx.m {
        let pos = ctx.start_pos + t;
        let q_row = &mut bufs.q_batch[t * ctx.geom.total_q..(t + 1) * ctx.geom.total_q];
        let k_row = &mut bufs.k_batch[t * ctx.geom.total_kv..(t + 1) * ctx.geom.total_kv];
        apply_norm_rope_inplace(q_row, ctx.geom.head_dim, pos, ctx.inv_freq)?;
        apply_norm_rope_inplace(k_row, ctx.geom.head_dim, pos, ctx.inv_freq)?;
    }

    let kv_start = ctx.start_pos * ctx.geom.total_kv;
    let kv_end = ctx.seq_len * ctx.geom.total_kv;
    {
        let seq_cap = kv_view.geometry().seq_capacity;
        let (k_plane, v_plane) = kv_view.layer_kv_mut(layer)?;
        if kv_end > k_plane.len() || kv_end > v_plane.len() {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: ctx.seq_len,
                max_context_tokens: seq_cap,
            });
        }
        k_plane[kv_start..kv_end].copy_from_slice(&bufs.k_batch);
        v_plane[kv_start..kv_end].copy_from_slice(&bufs.v_batch);
    }

    let (k_plane, v_plane) = kv_view.layer_kv(layer)?;
    gqa_causal_attention_prefill_compat(
        &mut bufs.attn_out,
        &bufs.q_batch,
        &k_plane[..kv_end],
        &v_plane[..kv_end],
        ctx.m,
        ctx.geom.n_heads_q,
        ctx.geom.n_heads_kv,
        ctx.geom.head_dim,
        ctx.seq_len,
        ctx.start_pos,
        ctx.geom.sliding_window,
        ctx.attn_scale,
        &mut bufs.scores_scratch,
    )?;

    act_scratch.prepare_batch(&bufs.attn_out, ctx.m, ctx.geom.total_q)?;
    cache.matmul_dense_cached(
        ctx.weights,
        &format!("blk.{layer}.attn_output.weight"),
        &mut bufs.o_proj,
        ctx.m,
        &bufs.attn_out,
        Some(act_scratch),
    )?;
    for (h, &delta) in hidden.iter_mut().zip(&bufs.o_proj) {
        *h += delta;
    }
    Ok(())
}

fn execute_dense_ffn_prefill_cached(
    ctx: &PrefillBatchCtx<'_>,
    layer: usize,
    ffn_norm: &[f32],
    hidden: &mut [f32],
    cache: &mut MistralLayerCache,
    act_scratch: &mut MistralActScratch,
) -> Result<(), K3Error> {
    let ffn_elems = ctx.m * ctx.geom.ffn_dim;
    let mut gate = vec![0.0_f32; ffn_elems];
    let mut up = vec![0.0_f32; ffn_elems];
    let mut act = vec![0.0_f32; ffn_elems];
    let mut down = vec![0.0_f32; hidden.len()];

    act_scratch.prepare_batch(ffn_norm, ctx.m, ctx.geom.hidden_dim)?;
    cache.matmul_dense_cached(
        ctx.weights,
        &format!("blk.{layer}.ffn_gate.weight"),
        &mut gate,
        ctx.m,
        ffn_norm,
        Some(act_scratch),
    )?;
    cache.matmul_dense_cached(
        ctx.weights,
        &format!("blk.{layer}.ffn_up.weight"),
        &mut up,
        ctx.m,
        ffn_norm,
        Some(act_scratch),
    )?;
    swiglu_compat(&mut act, &gate, &up)?;

    act_scratch.prepare_batch(&act, ctx.m, ctx.geom.ffn_dim)?;
    cache.matmul_dense_cached(
        ctx.weights,
        &format!("blk.{layer}.ffn_down.weight"),
        &mut down,
        ctx.m,
        &act,
        Some(act_scratch),
    )?;
    for (h, &d) in hidden.iter_mut().zip(&down) {
        *h += d;
    }
    Ok(())
}

struct ExpertBucket {
    token_indices: Vec<usize>,
    gate_weights: Vec<f32>,
}

#[derive(Clone, Copy)]
struct ActiveExpertSpec {
    layer: usize,
    expert_idx: usize,
    n_exp: usize,
}

struct MoeBatchIo<'a> {
    ffn_norm: &'a [f32],
    hidden: &'a mut [f32],
}

struct MoeExpertWorkspace {
    expert_in: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    act: Vec<f32>,
    down: Vec<f32>,
}

impl MoeExpertWorkspace {
    fn with_max_bucket(max_bucket: usize, hidden_dim: usize, ffn_dim: usize) -> Self {
        Self {
            expert_in: vec![0.0_f32; max_bucket * hidden_dim],
            gate: vec![0.0_f32; max_bucket * ffn_dim],
            up: vec![0.0_f32; max_bucket * ffn_dim],
            act: vec![0.0_f32; max_bucket * ffn_dim],
            down: vec![0.0_f32; max_bucket * hidden_dim],
        }
    }
}

fn bucket_moe_tokens(
    router_logits: &[f32],
    m: usize,
    n_exp: usize,
    top_k: usize,
) -> Result<Vec<ExpertBucket>, K3Error> {
    let mut buckets: Vec<ExpertBucket> = (0..n_exp)
        .map(|_| ExpertBucket {
            token_indices: Vec::new(),
            gate_weights: Vec::new(),
        })
        .collect();
    for t in 0..m {
        let row = &router_logits[t * n_exp..(t + 1) * n_exp];
        for (expert_idx, gate_weight) in route_topk_experts(row, top_k)? {
            let bucket =
                buckets
                    .get_mut(expert_idx)
                    .ok_or_else(|| K3Error::CapabilityMismatch {
                        architecture: "mixtral".to_owned(),
                        reason: format!(
                            "routed expert index {expert_idx} out of bounds for {n_exp} experts"
                        ),
                    })?;
            bucket.token_indices.push(t);
            bucket.gate_weights.push(gate_weight);
        }
    }
    Ok(buckets)
}

fn run_transient_proj(
    tensor: &TensorView<'_>,
    repacked: Option<&CachedDenseProjection>,
    m_e: usize,
    shape: (usize, usize),
    input: &[f32],
    out: &mut [f32],
    act_scratch: &MistralActScratch,
) -> Result<(), K3Error> {
    match repacked {
        Some(proj) => proj.gemm_with_scratch(out, act_scratch),
        None => matmul_tensor(out, tensor, m_e, input, shape.0, shape.1),
    }
}

fn execute_active_expert_batch(
    ctx: &PrefillBatchCtx<'_>,
    exp: ActiveExpertSpec,
    bucket: &ExpertBucket,
    io: &mut MoeBatchIo<'_>,
    ws: &mut MoeExpertWorkspace,
    cache: &mut MistralLayerCache,
    act_scratch: &mut MistralActScratch,
) -> Result<(), K3Error> {
    let m_e = bucket.token_indices.len();
    let hidden_dim = ctx.geom.hidden_dim;
    let ffn_dim = ctx.geom.ffn_dim;
    let hidden_len = m_e * hidden_dim;
    let ffn_len = m_e * ffn_dim;

    for (row_idx, &tok_idx) in bucket.token_indices.iter().enumerate() {
        let src = &io.ffn_norm[tok_idx * hidden_dim..(tok_idx + 1) * hidden_dim];
        ws.expert_in[row_idx * hidden_dim..(row_idx + 1) * hidden_dim].copy_from_slice(src);
    }

    let in_slice = &ws.expert_in[..hidden_len];
    let gate_slice = &mut ws.gate[..ffn_len];
    let up_slice = &mut ws.up[..ffn_len];
    let act_slice = &mut ws.act[..ffn_len];
    let down_slice = &mut ws.down[..hidden_len];

    let g_w = cache.slice_expert_projection(
        ctx.weights,
        exp.layer,
        "gate",
        exp.expert_idx,
        exp.n_exp,
        ffn_dim,
        hidden_dim,
    )?;
    let u_w = cache.slice_expert_projection(
        ctx.weights,
        exp.layer,
        "up",
        exp.expert_idx,
        exp.n_exp,
        ffn_dim,
        hidden_dim,
    )?;
    let g_proj = cache.repack_transient_projection(&g_w, ffn_dim, hidden_dim)?;
    let u_proj = cache.repack_transient_projection(&u_w, ffn_dim, hidden_dim)?;
    if g_proj.is_some() || u_proj.is_some() {
        act_scratch.prepare_batch(in_slice, m_e, hidden_dim)?;
    }
    run_transient_proj(
        &g_w,
        g_proj.as_ref(),
        m_e,
        (ffn_dim, hidden_dim),
        in_slice,
        gate_slice,
        act_scratch,
    )?;
    run_transient_proj(
        &u_w,
        u_proj.as_ref(),
        m_e,
        (ffn_dim, hidden_dim),
        in_slice,
        up_slice,
        act_scratch,
    )?;
    drop((g_proj, u_proj));

    swiglu_compat(act_slice, gate_slice, up_slice)?;

    let d_w = cache.slice_expert_projection(
        ctx.weights,
        exp.layer,
        "down",
        exp.expert_idx,
        exp.n_exp,
        hidden_dim,
        ffn_dim,
    )?;
    let d_proj = cache.repack_transient_projection(&d_w, hidden_dim, ffn_dim)?;
    if d_proj.is_some() {
        act_scratch.prepare_batch(act_slice, m_e, ffn_dim)?;
    }
    run_transient_proj(
        &d_w,
        d_proj.as_ref(),
        m_e,
        (hidden_dim, ffn_dim),
        act_slice,
        down_slice,
        act_scratch,
    )?;

    for (row_idx, (&tok_idx, &gate_weight)) in bucket
        .token_indices
        .iter()
        .zip(&bucket.gate_weights)
        .enumerate()
    {
        let h_row = &mut io.hidden[tok_idx * hidden_dim..(tok_idx + 1) * hidden_dim];
        let d_row = &down_slice[row_idx * hidden_dim..(row_idx + 1) * hidden_dim];
        for (h, &delta) in h_row.iter_mut().zip(d_row) {
            *h += gate_weight * delta;
        }
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "coordinates batched Mixtral router logits, token-bucketed active expert GEMMs, and scatter-add"
)]
fn execute_moe_ffn_prefill_cached(
    ctx: &PrefillBatchCtx<'_>,
    layer: usize,
    n_exp: usize,
    top_k: usize,
    ffn_norm: &[f32],
    hidden: &mut [f32],
    cache: &mut MistralLayerCache,
    act_scratch: &mut MistralActScratch,
) -> Result<(), K3Error> {
    let mut router_logits = vec![0.0_f32; ctx.m * n_exp];
    act_scratch.prepare_batch(ffn_norm, ctx.m, ctx.geom.hidden_dim)?;
    cache.matmul_dense_cached(
        ctx.weights,
        &format!("blk.{layer}.ffn_gate_inp.weight"),
        &mut router_logits,
        ctx.m,
        ffn_norm,
        Some(act_scratch),
    )?;

    let buckets = bucket_moe_tokens(&router_logits, ctx.m, n_exp, top_k)?;
    let max_bucket = buckets
        .iter()
        .map(|b| b.token_indices.len())
        .max()
        .unwrap_or(0);
    if max_bucket == 0 {
        return Ok(());
    }

    let mut ws =
        MoeExpertWorkspace::with_max_bucket(max_bucket, ctx.geom.hidden_dim, ctx.geom.ffn_dim);
    let mut io = MoeBatchIo { ffn_norm, hidden };
    for (expert_idx, bucket) in buckets.iter().enumerate() {
        if bucket.token_indices.is_empty() {
            continue;
        }
        execute_active_expert_batch(
            ctx,
            ActiveExpertSpec {
                layer,
                expert_idx,
                n_exp,
            },
            bucket,
            &mut io,
            &mut ws,
            cache,
            act_scratch,
        )?;
    }
    Ok(())
}

fn gguf_array_len(arr: &GgufArray<'_>) -> usize {
    match arr {
        GgufArray::Uint8(v) => v.len(),
        GgufArray::Int8(v) => v.len(),
        GgufArray::Uint16(v) => v.len(),
        GgufArray::Int16(v) => v.len(),
        GgufArray::Uint32(v) => v.len(),
        GgufArray::Int32(v) => v.len(),
        GgufArray::Float32(v) => v.len(),
        GgufArray::Bool(v) => v.len(),
        GgufArray::String(v) => v.len(),
        GgufArray::Uint64(v) => v.len(),
        GgufArray::Int64(v) => v.len(),
        GgufArray::Float64(v) => v.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    struct StoredTestTensor {
        name: String,
        quant: QuantType,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    }

    #[derive(Default)]
    struct RecipeTestWeights {
        tensors: HashMap<String, StoredTestTensor>,
        tensor_calls: Mutex<HashMap<String, usize>>,
    }

    fn encode_f16_bits(val: f32) -> u16 {
        let bits = val.to_bits();
        let sign = u16::try_from((bits >> 16) & 0x8000).unwrap_or(0);
        let exp = i32::try_from((bits >> 23) & 0xFF).unwrap_or(0) - 127 + 15;
        let mant = u16::try_from((bits >> 13) & 0x03FF).unwrap_or(0);
        if exp <= 0 {
            sign
        } else if exp >= 31 {
            sign | 0x7C00
        } else {
            sign | (u16::try_from(exp).unwrap_or(0) << 10) | mant
        }
    }

    impl RecipeTestWeights {
        fn insert_f32(&mut self, name: &str, shape: Vec<usize>, vals: &[f32]) {
            let mut bytes = Vec::with_capacity(vals.len() * 4);
            for &v in vals {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            self.tensors.insert(
                name.to_owned(),
                StoredTestTensor {
                    name: name.to_owned(),
                    quant: QuantType::F32,
                    shape,
                    bytes,
                },
            );
        }

        fn insert_f16_from_f32(&mut self, name: &str, shape: Vec<usize>, vals: &[f32]) {
            let mut bytes = Vec::with_capacity(vals.len() * 2);
            for &v in vals {
                bytes.extend_from_slice(&encode_f16_bits(v).to_le_bytes());
            }
            self.tensors.insert(
                name.to_owned(),
                StoredTestTensor {
                    name: name.to_owned(),
                    quant: QuantType::F16,
                    shape,
                    bytes,
                },
            );
        }

        fn reset_calls(&self) {
            self.tensor_calls.lock().unwrap().clear();
        }

        fn call_count(&self, name: &str) -> usize {
            self.tensor_calls
                .lock()
                .unwrap()
                .get(name)
                .copied()
                .unwrap_or(0)
        }
    }

    impl ModelWeights for RecipeTestWeights {
        fn contains_tensor(&self, name: &str) -> bool {
            self.tensors.contains_key(name)
        }

        fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
            *self
                .tensor_calls
                .lock()
                .unwrap()
                .entry(name.to_owned())
                .or_insert(0) += 1;
            let t = self
                .tensors
                .get(name)
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: name.to_owned(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: 0,
                    reason: format!("missing tensor '{name}'"),
                })?;
            TensorView::from_bytes(&t.name, t.quant, &t.shape, 0, &t.bytes)
        }

        fn tensor_count(&self) -> usize {
            self.tensors.len()
        }

        fn total_weight_bytes(&self) -> usize {
            self.tensors.values().map(|t| t.bytes.len()).sum()
        }
    }

    #[expect(clippy::cast_precision_loss, reason = "small synthetic test weights")]
    fn pseudo_matrix(rows: usize, cols: usize, seed: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|idx| {
                let centered = (((idx * 17 + seed * 13 + 5) % 29) as f32) - 14.0;
                centered * 0.025
            })
            .collect()
    }

    fn build_compact_dense_weights(recipe: &MistralRecipe, use_f16: bool) -> RecipeTestWeights {
        let mut w = RecipeTestWeights::default();
        let h = recipe.hidden_dim;
        let q = recipe.n_heads_q * recipe.head_dim;
        let kv = recipe.n_heads_kv * recipe.head_dim;
        let insert_mat =
            |weights: &mut RecipeTestWeights, name: &str, rows: usize, cols: usize, seed: usize| {
                let vals = pseudo_matrix(rows, cols, seed);
                if use_f16 {
                    weights.insert_f16_from_f32(name, vec![cols, rows], &vals);
                } else {
                    weights.insert_f32(name, vec![cols, rows], &vals);
                }
            };

        insert_mat(&mut w, "token_embd.weight", recipe.vocab_size, h, 3);
        w.insert_f32("blk.0.attn_norm.weight", vec![h], &vec![1.0_f32; h]);
        insert_mat(&mut w, "blk.0.attn_q.weight", q, h, 7);
        insert_mat(&mut w, "blk.0.attn_k.weight", kv, h, 11);
        insert_mat(&mut w, "blk.0.attn_v.weight", kv, h, 13);
        insert_mat(&mut w, "blk.0.attn_output.weight", h, q, 17);
        w.insert_f32("blk.0.ffn_norm.weight", vec![h], &vec![1.0_f32; h]);
        insert_mat(&mut w, "blk.0.ffn_gate.weight", recipe.ffn_dim, h, 19);
        insert_mat(&mut w, "blk.0.ffn_up.weight", recipe.ffn_dim, h, 23);
        insert_mat(&mut w, "blk.0.ffn_down.weight", h, recipe.ffn_dim, 29);
        w.insert_f32("output_norm.weight", vec![h], &vec![1.0_f32; h]);
        insert_mat(&mut w, "output.weight", recipe.vocab_size, h, 31);
        w
    }

    #[test]
    fn test_batched_dense_prefill_chunked_and_compute_logits_false_miri() {
        let mut recipe = MistralRecipe::new().with_sliding_window(Some(2));
        recipe.num_layers = 1;
        recipe.hidden_dim = 4;
        recipe.ffn_dim = 4;
        recipe.n_heads_q = 2;
        recipe.n_heads_kv = 1;
        recipe.head_dim = 2;
        recipe.vocab_size = 3;

        let weights = build_compact_dense_weights(&recipe, false);
        let n_ctx = 4;
        let kv_len = 2 * n_ctx * recipe.n_heads_kv * recipe.head_dim;
        let prompt = [0_u32, 1, 2];

        // 1. Single batched prefill over M=3 tokens (> SWA window 2).
        let mut kv_batched = vec![0.0_f32; kv_len];
        let mut logits_batched = vec![0.0_f32; recipe.vocab_size];
        let mut cache_batched = MistralLayerCache::new();
        let mut ctx_batched =
            ExecutionContext::new(&prompt, n_ctx, &mut logits_batched, &mut kv_batched).unwrap();
        recipe
            .prefill_step_with_cache(&mut ctx_batched, &weights, &mut cache_batched)
            .expect("batched prefill");

        // 2. Chunked prefill: first chunk omits logits (`compute_logits=false`), second projects row M-1.
        weights.reset_calls();
        let mut kv_chunked = vec![0.0_f32; kv_len];
        let mut cache_chunked = MistralLayerCache::new();
        let mut empty_logits: [f32; 0] = [];
        let mut ctx_c0 = ExecutionContext::new_prefill_chunk(
            &prompt[..2],
            n_ctx,
            0,
            false,
            &mut empty_logits,
            &mut kv_chunked,
        )
        .unwrap();
        recipe
            .prefill_step_with_cache(&mut ctx_c0, &weights, &mut cache_chunked)
            .expect("chunk 0 prefill");
        assert_eq!(weights.call_count("output_norm.weight"), 0);
        assert_eq!(weights.call_count("output.weight"), 0);

        let mut logits_chunked = vec![0.0_f32; recipe.vocab_size];
        let mut ctx_c1 = ExecutionContext::new_prefill_chunk(
            &prompt[2..],
            n_ctx,
            2,
            true,
            &mut logits_chunked,
            &mut kv_chunked,
        )
        .unwrap();
        recipe
            .prefill_step_with_cache(&mut ctx_c1, &weights, &mut cache_chunked)
            .expect("chunk 1 prefill");

        // 3. Strictly sequential single-token reference (reusing one layer cache).
        let mut kv_seq = vec![0.0_f32; kv_len];
        let mut logits_seq = vec![0.0_f32; recipe.vocab_size];
        let mut cache_seq = MistralLayerCache::new();
        let mut ctx_seq =
            ExecutionContext::new(&prompt[..1], n_ctx, &mut logits_seq, &mut kv_seq).unwrap();
        recipe
            .prefill_step_with_cache(&mut ctx_seq, &weights, &mut cache_seq)
            .expect("seq prefill[0]");
        for &tok in &prompt[1..] {
            ctx_seq.prepare_decode(tok).unwrap();
            recipe
                .decode_step_with_cache(&mut ctx_seq, &weights, &mut cache_seq)
                .expect("seq decode");
        }

        for ((&lb, &lc), &ls) in logits_batched.iter().zip(&logits_chunked).zip(&logits_seq) {
            assert!(lb.is_finite() && lc.is_finite() && ls.is_finite());
            assert!((lb - ls).abs() < 1e-4, "batch={lb} seq={ls}");
            assert!((lc - ls).abs() < 1e-4, "chunk={lc} seq={ls}");
        }
    }

    #[test]
    fn test_mixtral_token_bucketed_batched_prefill_miri() {
        let mut recipe = MistralRecipe::new_mixtral()
            .with_sliding_window(Some(2))
            .with_experts(4, 2);
        recipe.num_layers = 1;
        recipe.hidden_dim = 4;
        recipe.ffn_dim = 4;
        recipe.n_heads_q = 2;
        recipe.n_heads_kv = 1;
        recipe.head_dim = 2;
        recipe.vocab_size = 3;

        let mut weights = build_compact_dense_weights(&recipe, false);
        weights.tensors.remove("blk.0.ffn_gate.weight");
        weights.tensors.remove("blk.0.ffn_up.weight");
        weights.tensors.remove("blk.0.ffn_down.weight");

        // Route tokens [0, 1, 2] exclusively across experts {0, 1, 2}, leaving expert 3 cold (M_3 = 0).
        let h = recipe.hidden_dim;
        let mut emb = vec![0.02_f32; 3 * h];
        emb[0] = 10.0; // token 0 -> col 0 dominant -> experts (0, 1)
        emb[h + 1] = 10.0; // token 1 -> col 1 dominant -> experts (2, 0)
        emb[2 * h] = 10.0; // token 2 -> col 0 dominant (shares expert bucket with token 0)
        weights.insert_f32("token_embd.weight", vec![h, 3], &emb);

        let mut gate_inp = vec![0.0_f32; 4 * h];
        gate_inp[0] = 6.0;
        gate_inp[1] = 4.0;
        gate_inp[h] = 4.0;
        gate_inp[h + 1] = 2.0;
        gate_inp[2 * h] = 2.0;
        gate_inp[2 * h + 1] = 6.0;
        // Expert 3 stays at 0.0 across all dimensions so its logit is always 0.0 << active logits.
        weights.insert_f32("blk.0.ffn_gate_inp.weight", vec![h, 4], &gate_inp);

        for e in 0..4 {
            weights.insert_f32(
                &format!("blk.0.ffn_gate.{e}.weight"),
                vec![h, recipe.ffn_dim],
                &pseudo_matrix(recipe.ffn_dim, h, 40 + e * 3),
            );
            weights.insert_f32(
                &format!("blk.0.ffn_up.{e}.weight"),
                vec![h, recipe.ffn_dim],
                &pseudo_matrix(recipe.ffn_dim, h, 41 + e * 3),
            );
            weights.insert_f32(
                &format!("blk.0.ffn_down.{e}.weight"),
                vec![recipe.ffn_dim, h],
                &pseudo_matrix(h, recipe.ffn_dim, 42 + e * 3),
            );
        }

        let prompt = [0_u32, 1, 2];
        let n_ctx = 4;
        let kv_len = 2 * n_ctx * recipe.n_heads_kv * recipe.head_dim;

        // Batched token-bucketed prefill with transient expert repacking.
        weights.reset_calls();
        let mut kv_batch = vec![0.0_f32; kv_len];
        let mut logits_batch = vec![0.0_f32; recipe.vocab_size];
        let mut cache_batch = MistralLayerCache::new();
        let mut ctx_batch =
            ExecutionContext::new(&prompt, n_ctx, &mut logits_batch, &mut kv_batch).unwrap();
        recipe
            .prefill_step_with_cache(&mut ctx_batch, &weights, &mut cache_batch)
            .expect("mixtral batched prefill");

        // Each active expert (0, 1, 2) is loaded once per projection; cold expert 3 is never loaded.
        for e in 0..3 {
            assert_eq!(weights.call_count(&format!("blk.0.ffn_gate.{e}.weight")), 1);
            assert_eq!(weights.call_count(&format!("blk.0.ffn_up.{e}.weight")), 1);
            assert_eq!(weights.call_count(&format!("blk.0.ffn_down.{e}.weight")), 1);
        }
        assert_eq!(weights.call_count("blk.0.ffn_gate.3.weight"), 0);
        assert_eq!(weights.call_count("blk.0.ffn_up.3.weight"), 0);
        assert_eq!(weights.call_count("blk.0.ffn_down.3.weight"), 0);
        assert_eq!(cache_batch.stats().expert_zero_copy_slices, 9);

        // Zero-budget DirectFallback batched prefill (repack_transient_projection -> None).
        let mut kv_fb = vec![0.0_f32; kv_len];
        let mut logits_fb = vec![0.0_f32; recipe.vocab_size];
        let mut cache_fb = MistralLayerCache::new().with_max_repacked_bytes(0);
        let mut ctx_fb = ExecutionContext::new(&prompt, n_ctx, &mut logits_fb, &mut kv_fb).unwrap();
        recipe
            .prefill_step_with_cache(&mut ctx_fb, &weights, &mut cache_fb)
            .expect("mixtral fallback prefill");
        assert_eq!(cache_fb.resident_repacked_bytes(), 0);

        // Sequential single-token steps ending with `decode_step` on the final token.
        let mut kv_seq = vec![0.0_f32; kv_len];
        let mut logits_seq = vec![0.0_f32; recipe.vocab_size];
        let mut cache_seq = MistralLayerCache::new();
        let mut first_ctx =
            ExecutionContext::new(&prompt[..1], n_ctx, &mut logits_seq, &mut kv_seq).unwrap();
        recipe
            .prefill_step_with_cache(&mut first_ctx, &weights, &mut cache_seq)
            .expect("mixtral seq prefill[0]");
        for &tok in &prompt[1..] {
            first_ctx.prepare_decode(tok).unwrap();
            recipe
                .decode_step_with_cache(&mut first_ctx, &weights, &mut cache_seq)
                .expect("mixtral seq decode");
        }

        for ((&lb, &lfb), &ls) in logits_batch.iter().zip(&logits_fb).zip(&logits_seq) {
            assert!((lb - ls).abs() < 1e-4, "batch={lb} seq={ls}");
            assert!((lfb - ls).abs() < 1e-4, "fallback={lfb} seq={ls}");
        }
    }
}
