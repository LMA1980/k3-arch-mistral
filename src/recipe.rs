//! `MistralRecipe`: Architecture recipe for `Mistral`, `Mistral3` (Nemo / Small), and `Mixtral` models.
//!
//! Implements [`ArchRecipe`] supporting dynamic GGUF hydration, `Norm` `(2i, 2i+1)` rotary
//! embeddings on both `Q` and `K`, Sliding Window Attention (`SWA`) via [`KvCacheView`],
//! Sparse Top-`k` Mixture-of-Experts (`MoE`) routing over packed 3D or per-expert 2D weights,
//! and multi-quant parity (`F32`, `F16`, `BF16`, `Q4_0`, `Q4_K_M`, `Q5_K`, `Q6_K`, `Q8_0`)
//! per ADR-023 and ADR-024.

use jero_core::{
    ArchRecipe, ExecutionContext, GgufArray, GgufFile, GgufValue, K3Error, KvCacheView, KvGeometry,
    ModelMetadata, ModelWeights, QuantType, TensorDescriptor,
};

use crate::admission::validate_token_id_in_vocab;
pub use crate::kernels_compat::RopeScalingConfig;
use crate::kernels_compat::{
    dequantize_embedding_row, gqa_attention_compat, matvec_tensor, resolve_2d_dims, rmsnorm_compat,
    rope_inv_freq_compat, swiglu_compat, tensor_bounds_err,
};
use crate::repack_cache::{MistralActScratch, MistralLayerCache, with_shared_model_cache};

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
        let start_pos = ctx.position();
        let seq_len = start_pos + prompt_len;
        if weights.tensor_count() == 0 {
            return self.prefill_step(ctx, weights);
        }
        if prompt_tokens.is_empty() {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture_id().to_owned(),
                reason: "prefill_step requires at least one prompt token".to_owned(),
            });
        }
        if seq_len > ctx.n_ctx() {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: seq_len,
                max_context_tokens: ctx.n_ctx(),
            });
        }
        let geom = self.effective_geometry()?;
        let kv_geom = ctx.kv_geometry();
        let mut act_scratch = MistralActScratch::new();
        let mut last_hidden = Vec::new();
        {
            let kv_cache = ctx.kv_cache_mut();
            for (i, &tok) in prompt_tokens.iter().enumerate() {
                last_hidden = self.execute_real_token_step_with_cache(
                    geom,
                    tok,
                    start_pos + i,
                    seq_len,
                    kv_cache,
                    kv_geom,
                    weights,
                    cache,
                    &mut act_scratch,
                )?;
            }
        }
        project_lm_head(&last_hidden, geom, ctx.logits_mut(), weights)?;
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
        ctx.advance_position(prompt_len)?;
        let logits = ctx.logits_mut();
        if !logits.is_empty() {
            logits.fill(0.0);
            let bias = (last_hidden.first().copied().unwrap_or(0.0).abs() % 0.25) * 0.01;
            let target_slot = if logits.len() > 9 { 9 } else { 0 };
            logits[target_slot] = 1.0 + bias;
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

fn embed_real_token(
    token_id: u32,
    hidden_dim: usize,
    vocab_size: usize,
    weights: &dyn ModelWeights,
) -> Result<Vec<f32>, K3Error> {
    let emb_tensor = weights.tensor("token_embd.weight")?;
    let _ = resolve_2d_dims(&emb_tensor, vocab_size, hidden_dim)?;
    let row_idx = usize::try_from(token_id).map_err(|_| {
        tensor_bounds_err(
            &emb_tensor,
            0,
            &format!("token_id {token_id} overflows usize"),
        )
    })?;
    let mut hidden = vec![0.0_f32; hidden_dim];
    dequantize_embedding_row(&emb_tensor, row_idx, &mut hidden)?;
    Ok(hidden)
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

fn apply_layer_rmsnorm_cached(
    tensor_name: &str,
    input: &[f32],
    out: &mut [f32],
    eps: f32,
    weights: &dyn ModelWeights,
    cache: &mut MistralLayerCache,
) -> Result<(), K3Error> {
    let norm_weights = cache.get_or_decode_norm(weights, tensor_name, input.len())?;
    rmsnorm_compat(out, input, &norm_weights, eps)
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
