//! Memory admission control, Tekken/SentencePiece vocabulary validation, and `[INST]` contract
//! checks for `Mistral`, `Mistral3` (`12B`/`24B`), and `Mixtral` models under `RLIMIT_MEMLOCK`.
//!
//! Enforces the bounded 28 GiB physical memory ceiling (`29,360,128 KB` per
//! `/etc/security/limits.d/99-modelrunner.conf`) on the `SpacemiT` K3 target platform, preventing
//! unprivileged kernel OOM lockups, guiding `Mixtral` deployment toward `Q4_0` or compact `MoE`
//! architectures per ADR-037, and verifying `Mistral-Nemo-12B` / `Mistral-Small-3.2-24B` and Tekken
//! `131,072`-vocab LM-head/stop-token contracts prior to page locking.

use jero_core::{GgufArray, GgufFile, GgufValue, K3Error, ModelManifest, TensorDescriptor};

use crate::recipe::MistralRecipe;

/// Standard structured advisory recommended when Mixtral exceeds the 28 GiB memlock ceiling.
pub const COMPACT_MOE_ADVISORY: &str = "Mixtral-8x7B in Q4_K_M exceeds 28 GiB memlock headroom; use Q4_0 or deploy compact MoEs: Qwen3-30B-A3B, DeepSeek-V2-Lite-16B, Gemma-4-26B-A4B";

/// Default target hardware lockable memory ceiling: 28 GiB (`29,360,128 KB` = `30,064,771,072` bytes).
pub const DEFAULT_MEMLOCK_LIMIT_BYTES: u64 = 29_360_128 * 1024;

/// Mixtral-8x7B KV cache bytes per token: 32 layers * 2 (K and V) * 8 KV heads * 128 `head_dim` * 2 bytes (F16/BF16).
pub const MIXTRAL_KV_BYTES_PER_TOKEN: u64 = 32 * 2 * 8 * 128 * 2;

/// Mistral-7B KV cache bytes per token: 32 layers * 2 (K and V) * 8 KV heads * 128 `head_dim` * 2 bytes (F16/BF16).
pub const MISTRAL_7B_KV_BYTES_PER_TOKEN: u64 = 32 * 2 * 8 * 128 * 2;

/// `Mistral-Nemo-12B` and `Mistral-Small-3.2-24B` (`mistral3`) KV cache bytes per token:
/// 40 layers * 2 (K and V) * 8 KV heads * 128 `head_dim` * 2 bytes (F16/BF16).
pub const MISTRAL3_KV_BYTES_PER_TOKEN: u64 = 40 * 2 * 8 * 128 * 2;

/// Conservative default F32/Q8 activation and Tekken logits scratch budget (`16 MiB`).
pub const DEFAULT_MISTRAL_SCRATCH_BYTES: u64 = 16 * 1024 * 1024;

/// Legacy Mistral-7B / Mixtral-8x7B `SentencePiece` vocabulary size (`32,000`).
pub const SENTENCEPIECE_VOCAB_32K: usize = 32_000;

/// Extended `SentencePiece` vocabulary size used by Mistral-7B-v0.3 / Mixtral-8x22B (`32,768`).
pub const SENTENCEPIECE_VOCAB_32K_EXT: usize = 32_768;

/// Tekken BPE vocabulary size used by `Mistral-Nemo-12B` and `Mistral-Small-3.2-24B` (`131,072`).
pub const TEKKEN_VOCAB_SIZE: usize = 131_072;

/// Canonical model size profiles covered by [`verify_mixtral_admission`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MistralModelProfile {
    /// Mixtral-8x7B Sparse `MoE` (~46.7B params).
    Mixtral8x7B,
    /// Dense Mistral-7B (~7.2B params).
    Mistral7B,
    /// Dense `Mistral-Nemo-12B` (~12.2B params).
    MistralNemo12B,
    /// Dense `Mistral-Small-3.2-24B` (~24B params).
    MistralSmall24B,
}

/// Verified resident physical memory breakdown returned by admission checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemlockBudgetReport {
    /// Page-aligned model weights footprint in bytes.
    pub weight_bytes: u64,
    /// Page-aligned KV cache footprint for the requested context length in bytes.
    pub kv_bytes: u64,
    /// Page-aligned activation and logits scratch workspace footprint in bytes.
    pub scratch_bytes: u64,
    /// Total lockable memory required (`weight_bytes + kv_bytes + scratch_bytes`).
    pub total_required_bytes: u64,
    /// Target `RLIMIT_MEMLOCK` ceiling applied.
    pub limit_bytes: u64,
    /// Remaining lockable memory headroom (`limit_bytes - total_required_bytes`).
    pub headroom_bytes: u64,
}

/// Tokenizer family classification across `SentencePiece` (`32,000`/`32,768`) and `Tekken` (`131,072`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MistralTokenizerFamily {
    /// Legacy `SentencePiece` (`32,000` tokens, Mistral-7B-v0.1/v0.2 & Mixtral-8x7B).
    SentencePiece32k,
    /// Extended `SentencePiece` (`32,768` tokens, Mistral-7B-v0.3 & Mixtral-8x22B).
    SentencePiece32kExt,
    /// `Tekken` byte-level BPE (`131,072` tokens, Mistral-Nemo-12B & Mistral-Small-3.2-24B).
    Tekken131k,
    /// Non-standard or downscaled synthetic test vocabulary size.
    Custom {
        /// Declared vocabulary size (`> 0`).
        vocab_size: usize,
    },
}

impl MistralTokenizerFamily {
    /// Returns the vocabulary size represented by this tokenizer family.
    #[must_use]
    pub const fn vocab_size(self) -> usize {
        match self {
            Self::SentencePiece32k => SENTENCEPIECE_VOCAB_32K,
            Self::SentencePiece32kExt => SENTENCEPIECE_VOCAB_32K_EXT,
            Self::Tekken131k => TEKKEN_VOCAB_SIZE,
            Self::Custom { vocab_size } => vocab_size,
        }
    }

    /// Returns acceptable `tokenizer.ggml.model` identifiers for standard families,
    /// or an empty slice for [`Self::Custom`].
    #[must_use]
    pub const fn expected_ggml_models(self) -> &'static [&'static str] {
        match self {
            Self::SentencePiece32k | Self::SentencePiece32kExt => &["llama", "spm"],
            Self::Tekken131k => &["gpt2", "tekken"],
            Self::Custom { .. } => &[],
        }
    }
}

/// Validated vocabulary, stop-token, `[INST]` chat-template, and LM-head logits buffer contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MistralVocabContract {
    /// Architecture identifier (`"mistral"`, `"mistral3"`, or `"mixtral"`).
    pub architecture: String,
    /// Classified tokenizer family.
    pub family: MistralTokenizerFamily,
    /// Validated vocabulary size (`> 0`).
    pub vocab_size: usize,
    /// Optional `tokenizer.ggml.model` value declared in metadata.
    pub tokenizer_model: Option<String>,
    /// Beginning-of-sequence token ID (`< vocab_size`).
    pub bos_token_id: Option<u32>,
    /// End-of-sequence token ID (`< vocab_size`).
    pub eos_token_id: Option<u32>,
    /// End-of-turn token ID (`< vocab_size`).
    pub eot_token_id: Option<u32>,
    /// Padding token ID (`< vocab_size`).
    pub padding_token_id: Option<u32>,
    /// Deduplicated stop token IDs verified within `0..vocab_size`.
    pub stop_tokens: Vec<u32>,
    /// Whether GGUF chat template metadata declares the Mistral `[INST]` turn framing.
    pub has_inst_template: bool,
    /// Minimum bytes needed for one output logits vector (`vocab_size * 4`).
    pub logit_buffer_bytes: usize,
}

/// Helper safely rounding `val` up to the next multiple of `align` without integer overflow.
///
/// If `align` is zero, returns `val`. If rounding up would overflow `u64`, saturates to `u64::MAX`.
const fn align_up_u64(val: u64, align: u64) -> u64 {
    if align == 0 {
        return val;
    }
    let rem = val % align;
    if rem == 0 {
        val
    } else {
        val.saturating_add(align - rem)
    }
}

const fn normalize_memlock_limit(limit_memlock_bytes: u64) -> u64 {
    if limit_memlock_bytes == 0 {
        DEFAULT_MEMLOCK_LIMIT_BYTES
    } else {
        limit_memlock_bytes
    }
}

fn check_memlock_budget(
    weight_bytes_raw: u64,
    kv_bytes_raw: u64,
    scratch_bytes_raw: u64,
    limit: u64,
) -> Result<MemlockBudgetReport, K3Error> {
    let weight_bytes = align_up_u64(weight_bytes_raw, 4096);
    let kv_bytes = align_up_u64(kv_bytes_raw, 4096);
    let scratch_bytes = align_up_u64(scratch_bytes_raw, 4096);
    let total_required_bytes = weight_bytes
        .saturating_add(kv_bytes)
        .saturating_add(scratch_bytes);

    if total_required_bytes > limit {
        return Err(K3Error::InsufficientLockableMemory {
            required_bytes: total_required_bytes,
            weight_bytes,
            kv_bytes,
            available_bytes: limit,
            rlimit_memlock_bytes: limit,
        });
    }

    Ok(MemlockBudgetReport {
        weight_bytes,
        kv_bytes,
        scratch_bytes,
        total_required_bytes,
        limit_bytes: limit,
        headroom_bytes: limit - total_required_bytes,
    })
}

fn manifest_mentions_any(manifest: &ModelManifest, needles: &[&str]) -> bool {
    let haystacks = [
        manifest.weight_digest.as_str(),
        manifest.origin.as_deref().unwrap_or(""),
        manifest.license.as_deref().unwrap_or(""),
    ];
    for haystack in haystacks {
        let lower = haystack.to_ascii_lowercase();
        if needles.iter().any(|needle| lower.contains(needle)) {
            return true;
        }
    }
    false
}

/// Infers the default [`MistralModelProfile`] from a [`ModelManifest`].
#[must_use]
pub fn classify_mistral_manifest_profile(manifest: &ModelManifest) -> MistralModelProfile {
    if manifest.architecture_id.eq_ignore_ascii_case("mixtral") {
        return MistralModelProfile::Mixtral8x7B;
    }
    if manifest.architecture_id.eq_ignore_ascii_case("mistral3") {
        if manifest_mentions_any(manifest, &["nemo", "12b"]) {
            return MistralModelProfile::MistralNemo12B;
        }
        return MistralModelProfile::MistralSmall24B;
    }
    if manifest_mentions_any(manifest, &["nemo", "12b"]) {
        MistralModelProfile::MistralNemo12B
    } else if manifest_mentions_any(manifest, &["small", "24b"]) {
        MistralModelProfile::MistralSmall24B
    } else {
        MistralModelProfile::Mistral7B
    }
}

fn estimate_profile_weight_bytes(profile: MistralModelProfile, quantization: &str) -> u64 {
    let quant = quantization.to_ascii_uppercase();
    match profile {
        MistralModelProfile::Mixtral8x7B => match quant.as_str() {
            "Q4_K_M" => 28_990_000_000,
            "Q5_K" => 34_800_000_000,
            "Q6_K" => 40_800_000_000,
            "Q8_0" => 53_150_000_000,
            "F16" | "BF16" => 100_200_000_000,
            "F32" => 200_400_000_000,
            _ => 26_300_000_000,
        },
        MistralModelProfile::MistralNemo12B => match quant.as_str() {
            "Q4_K_M" => 7_480_000_000,
            "Q5_K" => 8_700_000_000,
            "Q6_K" => 10_050_000_000,
            "Q8_0" => 13_050_000_000,
            "F16" | "BF16" => 24_500_000_000,
            "F32" => 49_000_000_000,
            _ => 6_900_000_000,
        },
        MistralModelProfile::MistralSmall24B => match quant.as_str() {
            "Q4_K_M" => 14_330_000_000,
            "Q5_K" => 16_800_000_000,
            "Q6_K" => 19_400_000_000,
            "Q8_0" => 25_400_000_000,
            "F16" | "BF16" => 47_200_000_000,
            "F32" => 94_400_000_000,
            _ => 13_500_000_000,
        },
        MistralModelProfile::Mistral7B => match quant.as_str() {
            "Q4_K_M" => 4_370_000_000,
            "Q5_K" => 5_130_000_000,
            "Q6_K" => 5_940_000_000,
            "Q8_0" => 7_700_000_000,
            "F16" | "BF16" => 14_500_000_000,
            "F32" => 29_000_000_000,
            _ => 3_830_000_000,
        },
    }
}

const fn profile_kv_bytes_per_token(profile: MistralModelProfile) -> u64 {
    match profile {
        MistralModelProfile::Mixtral8x7B => MIXTRAL_KV_BYTES_PER_TOKEN,
        MistralModelProfile::Mistral7B => MISTRAL_7B_KV_BYTES_PER_TOKEN,
        MistralModelProfile::MistralNemo12B | MistralModelProfile::MistralSmall24B => {
            MISTRAL3_KV_BYTES_PER_TOKEN
        }
    }
}

/// Evaluates pre-flight physical memory admission for `MoE` models (specifically Mixtral-8x7B)
/// against the bounded `limit_memlock_bytes` ceiling.
///
/// Computes total lockable memory requirement:
/// $$\text{TotalRequired} = \text{AlignedWeightBytes} + \text{AlignedKvBytes}$$
///
/// If $\text{TotalRequired} > \text{Limit}$, admission is refused with [`K3Error::InsufficientLockableMemory`].
///
/// # Errors
///
/// Returns [`K3Error::InsufficientLockableMemory`] if model weights and KV cache exceed
/// `limit_memlock_bytes`.
pub fn evaluate_moe_memlock_admission(
    manifest: &ModelManifest,
    limit_memlock_bytes: u64,
) -> Result<(), K3Error> {
    if !manifest.architecture_id.eq_ignore_ascii_case("mixtral") {
        return Ok(());
    }

    let limit = normalize_memlock_limit(limit_memlock_bytes);
    let weight_bytes_raw = if manifest.weight_size_bytes > 0 {
        manifest.weight_size_bytes
    } else {
        match manifest.quantization.to_ascii_uppercase().as_str() {
            "Q4_K_M" => 28_990_000_000,
            "Q6_K" => 40_800_000_000,
            "Q8_0" => 53_150_000_000,
            "F16" | "BF16" => 100_200_000_000,
            _ => 26_300_000_000,
        }
    };
    let kv_bytes_raw = manifest
        .context_length
        .saturating_mul(MIXTRAL_KV_BYTES_PER_TOKEN);

    check_memlock_budget(weight_bytes_raw, kv_bytes_raw, 0, limit).map(|_| ())
}

/// Estimates runtime scratch workspace bytes (`f32` hidden/FFN activations, `Q8` staging, and
/// logits buffer) for `Mistral` / `Mistral3` / `Mixtral`.
#[must_use]
pub fn estimate_mistral_scratch_bytes(
    hidden_dim: usize,
    ffn_dim: usize,
    vocab_size: usize,
    batch_tokens: usize,
) -> u64 {
    let batch = u64::try_from(batch_tokens.max(1)).unwrap_or(u64::MAX);
    let hidden = u64::try_from(hidden_dim).unwrap_or(u64::MAX);
    let ffn = u64::try_from(ffn_dim).unwrap_or(u64::MAX);
    let vocab = u64::try_from(vocab_size).unwrap_or(u64::MAX);

    let per_tok_activations = hidden
        .saturating_mul(16)
        .saturating_add(ffn.saturating_mul(12));
    let activation_bytes = batch.saturating_mul(per_tok_activations);
    let logits_bytes = vocab.saturating_mul(4);
    align_up_u64(
        activation_bytes.saturating_add(logits_bytes).max(4096),
        4096,
    )
}

/// Verifies pre-flight physical memory admission (`weights + KV + scratch <= ceiling`) across
/// `Mixtral-8x7B`, `Mistral-7B`, `Mistral-Nemo-12B`, and `Mistral-Small-3.2-24B` (`mistral3`).
///
/// Uses [`DEFAULT_MISTRAL_SCRATCH_BYTES`] as the scratch estimate; callers with an exact scratch
/// budget or hydrated [`MistralRecipe`] can use [`verify_mistral_admission_with_scratch`] or
/// [`verify_mistral_recipe_admission`]. Non-Mistral manifests pass through without error.
///
/// # Errors
///
/// Returns [`K3Error::InsufficientLockableMemory`] deterministically before page locking if
/// `weight + kv + scratch` exceeds `limit_memlock_bytes` (or [`DEFAULT_MEMLOCK_LIMIT_BYTES`] when `0`).
pub fn verify_mixtral_admission(
    manifest: &ModelManifest,
    limit_memlock_bytes: u64,
) -> Result<MemlockBudgetReport, K3Error> {
    verify_mistral_admission_with_scratch(
        manifest,
        limit_memlock_bytes,
        DEFAULT_MISTRAL_SCRATCH_BYTES,
    )
}

/// Verifies pre-flight physical memory admission (`weights + KV + scratch <= ceiling`) with an
/// explicit `scratch_bytes` allowance for `Mixtral`, `Mistral3` (`12B`/`24B`), and `Mistral`.
///
/// # Errors
///
/// Returns [`K3Error::InsufficientLockableMemory`] if required bytes exceed the memlock ceiling.
pub fn verify_mistral_admission_with_scratch(
    manifest: &ModelManifest,
    limit_memlock_bytes: u64,
    scratch_bytes: u64,
) -> Result<MemlockBudgetReport, K3Error> {
    let limit = normalize_memlock_limit(limit_memlock_bytes);
    if !matches!(
        manifest.architecture_id.to_ascii_lowercase().as_str(),
        "mixtral" | "mistral3" | "mistral"
    ) {
        return Ok(MemlockBudgetReport {
            weight_bytes: 0,
            kv_bytes: 0,
            scratch_bytes: 0,
            total_required_bytes: 0,
            limit_bytes: limit,
            headroom_bytes: limit,
        });
    }

    let profile = classify_mistral_manifest_profile(manifest);
    let weight_bytes_raw = if manifest.weight_size_bytes > 0 {
        manifest.weight_size_bytes
    } else {
        estimate_profile_weight_bytes(profile, &manifest.quantization)
    };
    let kv_bytes_raw = manifest
        .context_length
        .saturating_mul(profile_kv_bytes_per_token(profile));

    check_memlock_budget(weight_bytes_raw, kv_bytes_raw, scratch_bytes, limit)
}

/// Verifies pre-flight physical memory admission using exact hydrated [`MistralRecipe`] dimensions
/// (`num_layers`, `n_heads_kv`, `head_dim`, `hidden_dim`, `ffn_dim`, `vocab_size`).
///
/// # Errors
///
/// Returns [`K3Error::InsufficientLockableMemory`] if exact `weights + KV + scratch > ceiling`.
pub fn verify_mistral_recipe_admission(
    manifest: &ModelManifest,
    recipe: &MistralRecipe,
    limit_memlock_bytes: u64,
    batch_tokens: usize,
) -> Result<MemlockBudgetReport, K3Error> {
    let limit = normalize_memlock_limit(limit_memlock_bytes);
    let profile = classify_mistral_manifest_profile(manifest);
    let weight_bytes_raw = if manifest.weight_size_bytes > 0 {
        manifest.weight_size_bytes
    } else {
        estimate_profile_weight_bytes(profile, &manifest.quantization)
    };

    let kv_per_token = u64::try_from(recipe.num_layers)
        .unwrap_or(u64::MAX)
        .saturating_mul(2)
        .saturating_mul(u64::try_from(recipe.n_heads_kv).unwrap_or(u64::MAX))
        .saturating_mul(u64::try_from(recipe.head_dim).unwrap_or(u64::MAX))
        .saturating_mul(2);
    let kv_bytes_raw = manifest.context_length.saturating_mul(kv_per_token);
    let scratch_bytes_raw = estimate_mistral_scratch_bytes(
        recipe.hidden_dim,
        recipe.ffn_dim,
        recipe.vocab_size,
        batch_tokens,
    );

    check_memlock_budget(weight_bytes_raw, kv_bytes_raw, scratch_bytes_raw, limit)
}

/// Classifies `vocab_size` into a canonical Mistral tokenizer family (`32,000`, `32,768`,
/// `Tekken 131,072`, or custom).
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if `vocab_size == 0`.
pub fn classify_mistral_vocab_family(vocab_size: usize) -> Result<MistralTokenizerFamily, K3Error> {
    match vocab_size {
        0 => Err(K3Error::CapabilityMismatch {
            architecture: "mistral".to_owned(),
            reason: "vocabulary size must be greater than zero".to_owned(),
        }),
        SENTENCEPIECE_VOCAB_32K => Ok(MistralTokenizerFamily::SentencePiece32k),
        SENTENCEPIECE_VOCAB_32K_EXT => Ok(MistralTokenizerFamily::SentencePiece32kExt),
        TEKKEN_VOCAB_SIZE => Ok(MistralTokenizerFamily::Tekken131k),
        other => Ok(MistralTokenizerFamily::Custom { vocab_size: other }),
    }
}

/// Validates vocabulary size, BOS/EOS/EOT/padding/stop token IDs, and caller `logits_len`
/// against LM-head bounds (`id < vocab_size <= logits_len`).
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if `vocab_size == 0` or any special/stop token ID is
/// out of range (`>= vocab_size`), or [`K3Error::TensorBoundsError`] if `logits_len < vocab_size`.
#[expect(
    clippy::too_many_arguments,
    reason = "explicit scalar contract parameters without allocating temporary builder structs"
)]
pub fn validate_vocab_stop_contract(
    architecture: &str,
    vocab_size: usize,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    end_of_turn_token_id: Option<u32>,
    padding_token_id: Option<u32>,
    extra_stop_tokens: &[u32],
    logits_len: usize,
) -> Result<MistralVocabContract, K3Error> {
    let family = classify_mistral_vocab_family(vocab_size)?;
    let logit_buffer_bytes =
        vocab_size
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: "output.weight".to_owned(),
                offset: 0,
                byte_len: usize::MAX,
                buffer_len: logits_len,
                reason: format!("vocab_size {vocab_size} overflows logits buffer byte length"),
            })?;

    if logits_len < vocab_size {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "output.weight".to_owned(),
            offset: 0,
            byte_len: logits_len,
            buffer_len: vocab_size,
            reason: format!(
                "logits buffer length {logits_len} is smaller than vocabulary size {vocab_size}"
            ),
        });
    }

    for (label, maybe_id) in [
        ("bos_token_id", bos_token_id),
        ("eos_token_id", eos_token_id),
        ("eot_token_id", end_of_turn_token_id),
        ("padding_token_id", padding_token_id),
    ] {
        if let Some(id) = maybe_id {
            validate_token_id_in_vocab(architecture, label, id, vocab_size)?;
        }
    }

    let mut stop_tokens = Vec::new();
    for id in [eos_token_id, end_of_turn_token_id].into_iter().flatten() {
        if !stop_tokens.contains(&id) {
            stop_tokens.push(id);
        }
    }
    for &id in extra_stop_tokens {
        validate_token_id_in_vocab(architecture, "stop_token_id", id, vocab_size)?;
        if !stop_tokens.contains(&id) {
            stop_tokens.push(id);
        }
    }

    Ok(MistralVocabContract {
        architecture: architecture.to_owned(),
        family,
        vocab_size,
        tokenizer_model: None,
        bos_token_id,
        eos_token_id,
        eot_token_id: end_of_turn_token_id,
        padding_token_id,
        stop_tokens,
        has_inst_template: false,
        logit_buffer_bytes,
    })
}

/// Formats a single user prompt into the canonical Mistral `[INST] ... [/INST]` wrapper.
#[must_use]
pub fn format_mistral_inst_prompt(user_prompt: &str) -> String {
    format!("[INST] {user_prompt} [/INST]")
}

fn resolve_and_verify_gguf_vocab(gguf: &GgufFile<'_>, arch: &str) -> Result<usize, K3Error> {
    let meta = gguf.metadata();
    let hidden_dim = meta.embedding_dim();
    let meta_vocab = gguf
        .metadata_value(&format!("{arch}.vocab_size"))
        .and_then(GgufValue::as_usize)
        .filter(|&v| v > 0)
        .or_else(|| meta.vocab_size());

    let tokens_len = gguf
        .metadata_value("tokenizer.ggml.tokens")
        .and_then(|v| match v {
            GgufValue::Array(arr) => Some(gguf_array_len(arr)),
            _ => None,
        })
        .filter(|&len| len > 0);

    let lm_head_desc = meta
        .find_tensor("output.weight")
        .or_else(|| meta.find_tensor("token_embd.weight"));
    let tensor_vocab = lm_head_desc.and_then(|desc| infer_lm_head_vocab_rows(desc, hidden_dim));

    let vocab_size =
        meta_vocab
            .or(tensor_vocab)
            .or(tokens_len)
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: arch.to_owned(),
                reason: "unable to determine GGUF vocabulary size from metadata or LM head"
                    .to_owned(),
            })?;

    if let Some(t_len) = tokens_len
        && t_len != vocab_size
    {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "tokenizer.ggml.tokens".to_owned(),
            offset: 0,
            byte_len: t_len,
            buffer_len: vocab_size,
            reason: format!(
                "tokenizer.ggml.tokens length {t_len} does not match declared vocab_size {vocab_size}"
            ),
        });
    }
    if let Some(lm_rows) = tensor_vocab
        && lm_rows != vocab_size
    {
        let name = lm_head_desc.map_or("output.weight", TensorDescriptor::name);
        return Err(K3Error::TensorBoundsError {
            tensor_name: name.to_owned(),
            offset: 0,
            byte_len: lm_rows,
            buffer_len: vocab_size,
            reason: format!(
                "LM head tensor '{name}' vocab rows {lm_rows} do not match vocab_size {vocab_size}"
            ),
        });
    }
    Ok(vocab_size)
}

/// Validates GGUF vocabulary size, tokenizer family (`SentencePiece` `32k`/`32.7k` vs `Tekken`
/// `131,072`), BOS/EOS/EOT/padding token bounds, LM-head rows, and `[INST]` chat-template state.
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] or [`K3Error::TensorBoundsError`] when architecture,
/// tokenizer family, tensor LM-head vocabulary rows, or special token bounds are inconsistent.
pub fn validate_gguf_tokenizer_and_lm_head(
    gguf: &GgufFile<'_>,
    logits_len: Option<usize>,
) -> Result<MistralVocabContract, K3Error> {
    let arch = gguf.architecture();
    if !matches!(arch, "mistral" | "mistral3" | "mixtral") {
        return Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: format!(
                "Mistral vocab contract supports 'mistral', 'mistral3', and 'mixtral', got '{arch}'"
            ),
        });
    }

    let vocab_size = resolve_and_verify_gguf_vocab(gguf, arch)?;
    let tokenizer_model = gguf
        .metadata_value("tokenizer.ggml.model")
        .and_then(GgufValue::as_str)
        .map(str::to_owned);
    let family = classify_mistral_vocab_family(vocab_size)?;
    if let Some(ref model_name) = tokenizer_model {
        let allowed = family.expected_ggml_models();
        if !allowed.is_empty() && !allowed.iter().any(|&m| model_name.eq_ignore_ascii_case(m)) {
            return Err(K3Error::CapabilityMismatch {
                architecture: arch.to_owned(),
                reason: format!(
                    "tokenizer.ggml.model '{model_name}' conflicts with {family:?} (vocab_size={vocab_size}, expected {allowed:?})"
                ),
            });
        }
    }

    let bos_token_id = gguf
        .metadata_value("tokenizer.ggml.bos_token_id")
        .and_then(GgufValue::as_u32);
    let eos_token_id = gguf
        .metadata_value("tokenizer.ggml.eos_token_id")
        .and_then(GgufValue::as_u32);
    let end_of_turn_id = gguf
        .metadata_value("tokenizer.ggml.eot_token_id")
        .and_then(GgufValue::as_u32);
    let padding_token_id = gguf
        .metadata_value("tokenizer.ggml.padding_token_id")
        .and_then(GgufValue::as_u32);

    let mut contract = validate_vocab_stop_contract(
        arch,
        vocab_size,
        bos_token_id,
        eos_token_id,
        end_of_turn_id,
        padding_token_id,
        &[],
        logits_len.unwrap_or(vocab_size),
    )?;
    contract.tokenizer_model = tokenizer_model;
    contract.has_inst_template = gguf
        .metadata_value("tokenizer.chat_template")
        .and_then(GgufValue::as_str)
        .is_some_and(|tmpl| tmpl.contains("[INST]"));
    Ok(contract)
}

pub(crate) fn validate_token_id_in_vocab(
    architecture: &str,
    label: &str,
    token_id: u32,
    vocab_size: usize,
) -> Result<(), K3Error> {
    let idx = usize::try_from(token_id).map_err(|_| K3Error::CapabilityMismatch {
        architecture: architecture.to_owned(),
        reason: format!("{label} ({token_id}) overflows usize"),
    })?;
    if idx >= vocab_size {
        return Err(K3Error::CapabilityMismatch {
            architecture: architecture.to_owned(),
            reason: format!(
                "{label} ({token_id}) exceeds vocabulary bounds (vocab_size={vocab_size})"
            ),
        });
    }
    Ok(())
}

fn infer_lm_head_vocab_rows(desc: &TensorDescriptor, hidden_dim: usize) -> Option<usize> {
    let shape = desc.shape();
    if shape.len() != 2 || hidden_dim == 0 {
        return None;
    }
    if shape[0] == hidden_dim {
        Some(shape[1])
    } else if shape[1] == hidden_dim {
        Some(shape[0])
    } else {
        None
    }
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
    use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;
    use jero_core::{GgufFile, GgufValue, K3Error, ModelManifest, QuantType};

    use super::*;
    use crate::recipe::MistralRecipe;

    fn make_manifest(
        arch: &str,
        quant: &str,
        ctx: u64,
        weight_bytes: u64,
        origin: Option<&str>,
    ) -> ModelManifest {
        let mut m = ModelManifest::new(
            "gguf",
            arch,
            "1.0.0",
            quant,
            ctx,
            "sha256:test",
            weight_bytes,
        );
        m.origin = origin.map(str::to_owned);
        m
    }

    #[test]
    fn test_memlock_admission_mistral_small_24b_nemo_12b_and_mixtral_bounds_miri() {
        // Legacy evaluate_moe_memlock_admission remains backwards compatible for jero-engine.
        let non_moe = make_manifest("mistral", "F32", 32_768, 50_000_000_000, None);
        evaluate_moe_memlock_admission(&non_moe, DEFAULT_MEMLOCK_LIMIT_BYTES)
            .expect("non-mixtral passthrough");

        // Mistral-Small-3.2-24B Q4_K_M at 8192 context fits inside 28 GiB with > 13 GiB headroom.
        let small_24b_q4k = make_manifest(
            "mistral3",
            "Q4_K_M",
            8192,
            0,
            Some("mistralai/Mistral-Small-3.2-24B-Instruct"),
        );
        let report_small = verify_mixtral_admission(&small_24b_q4k, DEFAULT_MEMLOCK_LIMIT_BYTES)
            .expect("24B Q4_K_M admitted inside 28 GiB");
        assert_eq!(report_small.scratch_bytes, DEFAULT_MISTRAL_SCRATCH_BYTES);
        assert!(report_small.total_required_bytes <= DEFAULT_MEMLOCK_LIMIT_BYTES);
        assert!(report_small.headroom_bytes > 13_000_000_000);

        // Mistral-Nemo-12B Q8_0 at 16384 context is classified and admitted inside 28 GiB.
        let nemo_12b_q8 = make_manifest(
            "mistral3",
            "Q8_0",
            16_384,
            0,
            Some("mistralai/Mistral-Nemo-12B-2407"),
        );
        assert_eq!(
            classify_mistral_manifest_profile(&nemo_12b_q8),
            MistralModelProfile::MistralNemo12B
        );
        let report_nemo =
            verify_mixtral_admission(&nemo_12b_q8, 0).expect("Nemo 12B Q8_0 fits 28 GiB");
        assert_eq!(report_nemo.limit_bytes, DEFAULT_MEMLOCK_LIMIT_BYTES);

        // Oversized Mistral-Small-3.2-24B F16 fails deterministically before mlock.
        let small_24b_f16 = make_manifest(
            "mistral3",
            "F16",
            4096,
            0,
            Some("mistralai/Mistral-Small-3.2-24B"),
        );
        let err_f16 = verify_mixtral_admission(&small_24b_f16, DEFAULT_MEMLOCK_LIMIT_BYTES)
            .expect_err("24B F16 exceeds 28 GiB");
        assert!(matches!(
            err_f16,
            K3Error::InsufficientLockableMemory { .. }
        ));

        // Extreme context window pushes 24B Q4_K_M over 28 GiB via KV budget.
        let huge_ctx = make_manifest("mistral3", "Q4_K_M", 262_144, 14_330_000_000, None);
        let err_ctx = verify_mixtral_admission(&huge_ctx, DEFAULT_MEMLOCK_LIMIT_BYTES)
            .expect_err("262k context exceeds 28 GiB");
        assert!(matches!(
            err_ctx,
            K3Error::InsufficientLockableMemory { .. }
        ));

        // Saturated / overflow weight size fails closed rather than wrapping around.
        let overflow_m = make_manifest("mixtral", "Q4_0", 4096, u64::MAX - 10, None);
        assert!(matches!(
            verify_mixtral_admission(&overflow_m, DEFAULT_MEMLOCK_LIMIT_BYTES),
            Err(K3Error::InsufficientLockableMemory { .. })
        ));

        // Exact recipe admission for Mistral-Small-3.2-24B (40 layers, 8 kv heads, 128 head_dim, Tekken 131k).
        let mut recipe_24b = MistralRecipe::new_mistral3();
        recipe_24b.num_layers = 40;
        recipe_24b.hidden_dim = 5120;
        recipe_24b.ffn_dim = 32_768;
        recipe_24b.n_heads_q = 32;
        recipe_24b.n_heads_kv = 8;
        recipe_24b.head_dim = 128;
        recipe_24b.vocab_size = TEKKEN_VOCAB_SIZE;
        let ok_4k = verify_mistral_recipe_admission(
            &make_manifest("mistral3", "Q4_K_M", 4096, 14_330_000_000, None),
            &recipe_24b,
            DEFAULT_MEMLOCK_LIMIT_BYTES,
            64,
        )
        .expect("24B Q4_K_M at 4K fits");
        assert!(ok_4k.scratch_bytes >= TEKKEN_VOCAB_SIZE as u64 * 4);

        let rej_128k = verify_mistral_recipe_admission(
            &make_manifest("mistral3", "Q4_K_M", 131_072, 14_330_000_000, None),
            &recipe_24b,
            DEFAULT_MEMLOCK_LIMIT_BYTES,
            64,
        )
        .expect_err("24B Q4_K_M at 128K exceeds 28 GiB");
        assert!(matches!(
            rej_128k,
            K3Error::InsufficientLockableMemory { .. }
        ));
    }

    #[test]
    fn test_vocab_families_stop_tokens_inst_template_and_lm_head_bounds_miri() {
        assert_eq!(
            classify_mistral_vocab_family(SENTENCEPIECE_VOCAB_32K).unwrap(),
            MistralTokenizerFamily::SentencePiece32k
        );
        assert_eq!(
            classify_mistral_vocab_family(SENTENCEPIECE_VOCAB_32K_EXT).unwrap(),
            MistralTokenizerFamily::SentencePiece32kExt
        );
        assert_eq!(
            classify_mistral_vocab_family(TEKKEN_VOCAB_SIZE).unwrap(),
            MistralTokenizerFamily::Tekken131k
        );
        assert_eq!(
            classify_mistral_vocab_family(256).unwrap(),
            MistralTokenizerFamily::Custom { vocab_size: 256 }
        );
        assert!(matches!(
            classify_mistral_vocab_family(0),
            Err(K3Error::CapabilityMismatch { .. })
        ));

        let tekken = validate_vocab_stop_contract(
            "mistral3",
            TEKKEN_VOCAB_SIZE,
            Some(1),
            Some(2),
            Some(10),
            Some(11),
            &[2, 131_071],
            TEKKEN_VOCAB_SIZE,
        )
        .expect("Tekken 131k valid");
        assert_eq!(tekken.family, MistralTokenizerFamily::Tekken131k);
        assert_eq!(tekken.stop_tokens, vec![2, 10, 131_071]);
        assert_eq!(tekken.logit_buffer_bytes, TEKKEN_VOCAB_SIZE * 4);

        // Reject out-of-range special and stop token IDs.
        assert!(matches!(
            validate_vocab_stop_contract(
                "mistral3",
                TEKKEN_VOCAB_SIZE,
                Some(1),
                Some(131_072),
                None,
                None,
                &[],
                TEKKEN_VOCAB_SIZE,
            ),
            Err(K3Error::CapabilityMismatch { .. })
        ));
        assert!(matches!(
            validate_vocab_stop_contract(
                "mistral",
                SENTENCEPIECE_VOCAB_32K,
                Some(1),
                Some(2),
                None,
                None,
                &[32_000],
                SENTENCEPIECE_VOCAB_32K,
            ),
            Err(K3Error::CapabilityMismatch { .. })
        ));

        // Reject undersized caller logit buffer.
        assert!(matches!(
            validate_vocab_stop_contract(
                "mistral3",
                TEKKEN_VOCAB_SIZE,
                Some(1),
                Some(2),
                None,
                None,
                &[],
                SENTENCEPIECE_VOCAB_32K_EXT,
            ),
            Err(K3Error::TensorBoundsError { .. })
        ));

        assert_eq!(
            format_mistral_inst_prompt("Explain K3 KV paging."),
            "[INST] Explain K3 KV paging. [/INST]"
        );
    }

    enum KvVal<'a> {
        U32(u32),
        Str(&'a str),
    }

    fn write_gguf_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn encode_gguf_v3_kv_only(kvs: &[(&str, KvVal<'_>)]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3_u32.to_le_bytes());
        buf.extend_from_slice(&0_u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for (k, v) in kvs {
            write_gguf_str(&mut buf, k);
            match v {
                KvVal::U32(x) => {
                    buf.extend_from_slice(&4_u32.to_le_bytes());
                    buf.extend_from_slice(&x.to_le_bytes());
                }
                KvVal::Str(s) => {
                    buf.extend_from_slice(&8_u32.to_le_bytes());
                    write_gguf_str(&mut buf, s);
                }
            }
        }
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        buf
    }

    #[test]
    fn test_gguf_tokenizer_contract_and_from_gguf_validation_miri() {
        let tekken_bytes = encode_gguf_v3_kv_only(&[
            ("general.architecture", KvVal::Str("mistral3")),
            ("mistral3.block_count", KvVal::U32(40)),
            ("mistral3.embedding_length", KvVal::U32(5120)),
            ("mistral3.feed_forward_length", KvVal::U32(32_768)),
            ("mistral3.attention.head_count", KvVal::U32(32)),
            ("mistral3.attention.head_count_kv", KvVal::U32(8)),
            ("mistral3.attention.key_length", KvVal::U32(128)),
            ("mistral3.vocab_size", KvVal::U32(131_072)),
            ("tokenizer.ggml.model", KvVal::Str("gpt2")),
            ("tokenizer.ggml.bos_token_id", KvVal::U32(1)),
            ("tokenizer.ggml.eos_token_id", KvVal::U32(2)),
            ("tokenizer.ggml.eot_token_id", KvVal::U32(3)),
            ("tokenizer.ggml.padding_token_id", KvVal::U32(0)),
            (
                "tokenizer.chat_template",
                KvVal::Str("<s>[INST] {{ content }} [/INST]"),
            ),
        ]);
        let tekken_gguf = GgufFile::parse(&tekken_bytes).expect("parse tekken gguf");
        let contract = validate_gguf_tokenizer_and_lm_head(&tekken_gguf, None).expect("valid");
        assert_eq!(contract.family, MistralTokenizerFamily::Tekken131k);
        assert!(contract.has_inst_template);
        assert_eq!(contract.stop_tokens, vec![2, 3]);
        let recipe = MistralRecipe::from_gguf(&tekken_gguf).expect("hydrate recipe");
        assert_eq!(recipe.vocab_size, TEKKEN_VOCAB_SIZE);

        // Tokenizer tokens array length mismatch against declared vocab_size fails closed.
        let tok_len_mismatch = create_synthetic_gguf_v3_with_kvs(
            "mistral3",
            32,
            &[("mistral3.vocab_size", GgufValue::Uint32(131_072))],
            &[],
        );
        let tok_len_gguf = GgufFile::parse(&tok_len_mismatch).expect("parse tok mismatch");
        assert!(matches!(
            validate_gguf_tokenizer_and_lm_head(&tok_len_gguf, None),
            Err(K3Error::TensorBoundsError { .. })
        ));

        // Tokenizer family mismatch (llama SPM paired with Tekken 131k vocab) fails closed.
        let bad_tok_model = encode_gguf_v3_kv_only(&[
            ("general.architecture", KvVal::Str("mistral3")),
            ("mistral3.vocab_size", KvVal::U32(131_072)),
            ("tokenizer.ggml.model", KvVal::Str("llama")),
        ]);
        let bad_gguf = GgufFile::parse(&bad_tok_model).expect("parse bad model");
        assert!(matches!(
            validate_gguf_tokenizer_and_lm_head(&bad_gguf, None),
            Err(K3Error::CapabilityMismatch { .. })
        ));

        // Out-of-bounds EOS ID fails both validate_gguf_tokenizer_and_lm_head and from_gguf.
        let bad_eos = encode_gguf_v3_kv_only(&[
            ("general.architecture", KvVal::Str("mistral3")),
            ("mistral3.block_count", KvVal::U32(2)),
            ("mistral3.embedding_length", KvVal::U32(64)),
            ("mistral3.feed_forward_length", KvVal::U32(128)),
            ("mistral3.attention.head_count", KvVal::U32(4)),
            ("mistral3.attention.head_count_kv", KvVal::U32(2)),
            ("mistral3.vocab_size", KvVal::U32(32_000)),
            ("tokenizer.ggml.eos_token_id", KvVal::U32(32_000)),
        ]);
        let bad_eos_gguf = GgufFile::parse(&bad_eos).expect("parse bad eos");
        assert!(matches!(
            validate_gguf_tokenizer_and_lm_head(&bad_eos_gguf, None),
            Err(K3Error::CapabilityMismatch { .. })
        ));
        assert!(matches!(
            MistralRecipe::from_gguf(&bad_eos_gguf),
            Err(K3Error::CapabilityMismatch { .. })
        ));

        // LM head row count mismatch against vocab_size fails closed.
        let bad_lm = create_synthetic_gguf_v3_with_kvs(
            "mistral",
            32,
            &[
                ("mistral.embedding_length", GgufValue::Uint32(32)),
                ("mistral.vocab_size", GgufValue::Uint32(16)),
            ],
            &[("output.weight", QuantType::F32, &[32, 8])],
        );
        let bad_lm_gguf = GgufFile::parse(&bad_lm).expect("parse bad lm");
        assert!(matches!(
            validate_gguf_tokenizer_and_lm_head(&bad_lm_gguf, None),
            Err(K3Error::TensorBoundsError { .. })
        ));
    }
}
