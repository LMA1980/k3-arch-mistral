//! Model-scoped `128-byte`-aligned `RepackedWeight` dense projection cache, zero-copy 3D
//! `_exps` expert slicer, and once-per-load 1D `RMSNorm` cache for `Mistral` / `Mistral3` /
//! `Mixtral`.
//!
//! Ensures dense projections (`attn_q`, `attn_k`, `attn_v`, `attn_output`, dense
//! `ffn_gate`/`ffn_up`/`ffn_down`, and router `ffn_gate_inp`) reuse `128-byte`-aligned layouts
//! and decoded `f32` `RMSNorm` scales across tokens without exceeding `28 GiB` `RLIMIT_MEMLOCK`
//! via zero-copy expert slicing over packed 3D `[n_embd, ffn_dim, n_expert]` `_exps` buffers.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(not(feature = "with-kernels"))]
use jero_core::TensorShape;
use jero_core::{K3Error, ModelWeights, QuantType, TensorView};
#[cfg(feature = "with-kernels")]
use jero_kernels_rvv::{QuantizedActQ8, RepackedWeight};

#[cfg(not(feature = "with-kernels"))]
use crate::kernels_compat::{
    bf16_to_f32_portable, decode_quant_row_portable, dot_f32_compat, f16_to_f32_portable,
};
use crate::kernels_compat::{
    decode_norm_weights, matvec_tensor, resolve_2d_dims, tensor_bounds_err,
};

/// Required payload alignment (`128` bytes) matching `X100` `RVV` / `A100` `IME` tiles.
pub const WEIGHT_CACHE_ALIGNMENT_BYTES: usize = 128;

/// Default maximum resident `RepackedWeight` byte budget (`4 GiB`) when unbounded headroom applies.
pub const DEFAULT_MAX_REPACKED_CACHE_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// Default element threshold above which `F16`/`BF16` weights stay in compact `128B`-aligned `u16` tiles.
pub const DEFAULT_DENSE_F32_MAX_ELEMENTS: usize = 16 * 1024 * 1024;

/// Maximum resident dense projection tensors stored in one [`MistralLayerCache`].
const MAX_CACHED_DENSE_TENSORS: usize = 2048;

/// Maximum concurrent model instances tracked in the process-wide shared layer cache pool.
const MAX_ACTIVE_GLOBAL_MODELS: usize = 4;

/// Candidate anchor tensors probed in priority order to identify a model load.
const MODEL_ANCHOR_CANDIDATES: &[&str] = &[
    "token_embd.weight",
    "output_norm.weight",
    "blk.0.attn_norm.weight",
    "blk.0.attn_q.weight",
];

/// Pointer, geometry, and content fingerprint identifying a loaded [`TensorView`] instance.
///
/// Prevents cross-model stale cache hits and heap allocator address-reuse (`ABA`) collisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorIdentity {
    /// Backing slice base address.
    pub data_ptr: usize,
    /// Backing byte length.
    pub byte_len: usize,
    /// Tensor quantization format.
    pub quant: QuantType,
    /// Number of logical dimensions captured in `shape`.
    pub ndims: usize,
    /// Up to four tensor shape dimensions (`0`-padded).
    pub shape: [usize; 4],
    /// Sampled `FNV-1a` 64-bit fingerprint over name, dimensions, and payload bytes.
    pub fingerprint64: u64,
}

impl TensorIdentity {
    /// Captures identity and content fingerprint for `tensor`.
    #[must_use]
    pub fn from_view(tensor: &TensorView<'_>) -> Self {
        let raw = tensor.as_bytes();
        let src_shape = tensor.shape();
        let ndims = src_shape.len();
        let mut shape = [0_usize; 4];
        for (dst, &dim) in shape.iter_mut().zip(src_shape.iter()) {
            *dst = dim;
        }
        Self {
            data_ptr: raw.as_ptr() as usize,
            byte_len: raw.len(),
            quant: tensor.quant(),
            ndims,
            shape,
            fingerprint64: compute_tensor_fingerprint64(
                tensor.name(),
                tensor.quant(),
                src_shape,
                raw,
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WeightCacheKey {
    identity: TensorIdentity,
    rows: usize,
    cols: usize,
}

/// Telemetry counters recorded by [`MistralLayerCache`] for observability and verification.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MistralCacheStats {
    /// Number of 1D `RMSNorm` (`attn_norm`, `ffn_norm`, `output_norm`) decodes performed.
    pub norm_decodes: usize,
    /// Number of 1D `RMSNorm` lookups satisfied directly from cache.
    pub norm_hits: usize,
    /// Number of dense projection repacks performed.
    pub dense_repacks: usize,
    /// Number of dense projection lookups satisfied from cache.
    pub dense_hits: usize,
    /// Number of dense projections kept zero-copy uncached to stay within memlock budget.
    pub repack_budget_skips: usize,
    /// Number of zero-copy expert slices resolved across `3D` `_exps` or split expert tensors.
    pub expert_zero_copy_slices: usize,
    /// Number of full cache invalidations triggered when the backing model instance changed.
    pub model_invalidations: usize,
}

/// Verified byte-offset metadata describing one zero-copy expert slice inside a packed `3D` tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertSliceMeta {
    /// Zero-based active expert index `e`.
    pub expert_idx: usize,
    /// Total expert count declared on the `3D` `_exps` axis.
    pub num_experts: usize,
    /// Resolved `2D` matrix rows.
    pub rows: usize,
    /// Resolved `2D` matrix columns.
    pub cols: usize,
    /// Byte offset from the start of the `3D` tensor payload to expert `e`.
    pub byte_offset_within_tensor: usize,
    /// Absolute offset recorded on the resulting [`TensorView`].
    pub absolute_offset: usize,
    /// Exact payload byte length for expert `e`.
    pub slice_byte_len: usize,
    /// Quantization format of the expert slice.
    pub quant: QuantType,
}

/// Reusable single-token activation buffer (`Q8` tiles under `with-kernels`, `f32` fallback otherwise).
#[derive(Debug, Default)]
pub struct MistralActScratch {
    #[cfg(feature = "with-kernels")]
    q8: QuantizedActQ8,
    #[cfg(not(feature = "with-kernels"))]
    f32_buf: Vec<f32>,
}

impl MistralActScratch {
    /// Creates an empty reusable activation buffer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Encodes a single `f32` activation row (`[1, cols]`) into this scratch buffer.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if `act` is empty.
    pub fn prepare_row(&mut self, act: &[f32]) -> Result<(), K3Error> {
        if act.is_empty() {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "MistralActScratch".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: "activation vector must not be empty".to_owned(),
            });
        }
        #[cfg(feature = "with-kernels")]
        {
            self.q8.quantize_into(act, 1, act.len())
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            self.f32_buf.clear();
            self.f32_buf.extend_from_slice(act);
            Ok(())
        }
    }
}

#[cfg(not(feature = "with-kernels"))]
#[repr(C, align(128))]
#[derive(Debug, Clone, Copy)]
struct Aligned128Block([u8; WEIGHT_CACHE_ALIGNMENT_BYTES]);

#[cfg(not(feature = "with-kernels"))]
#[derive(Debug)]
struct PortableAlignedMatrix {
    rows: usize,
    cols: usize,
    quant: QuantType,
    is_compact_half: bool,
    row_bytes: usize,
    blocks_per_row: usize,
    blocks: Vec<Aligned128Block>,
}

#[cfg(not(feature = "with-kernels"))]
impl PortableAlignedMatrix {
    fn from_tensor_with_half_limit(
        tensor: &TensorView<'_>,
        rows: usize,
        cols: usize,
        dense_f32_max_elems: usize,
    ) -> Result<Self, K3Error> {
        let quant = tensor.quant();
        let block_elems = quant.block_elements();
        if cols == 0 || cols % block_elems != 0 {
            return Err(tensor_bounds_err(
                tensor,
                0,
                &format!(
                    "columns ({cols}) must be divisible by {} block size ({block_elems})",
                    quant.as_str()
                ),
            ));
        }
        let block_bytes = quant.block_bytes();
        let row_bytes = (cols / block_elems)
            .checked_mul(block_bytes)
            .ok_or_else(|| tensor_bounds_err(tensor, usize::MAX, "row byte stride overflow"))?;
        let required_bytes = rows
            .checked_mul(row_bytes)
            .ok_or_else(|| tensor_bounds_err(tensor, usize::MAX, "matrix byte length overflow"))?;
        let raw = tensor.as_bytes();
        if raw.len() < required_bytes {
            return Err(tensor_bounds_err(
                tensor,
                required_bytes,
                "truncated matrix payload for cached dense projection",
            ));
        }

        let is_compact_half = matches!(quant, QuantType::F16 | QuantType::BF16)
            && rows.saturating_mul(cols) > dense_f32_max_elems;
        let blocks_per_row = row_bytes.div_ceil(WEIGHT_CACHE_ALIGNMENT_BYTES).max(1);
        let total_blocks = rows
            .checked_mul(blocks_per_row)
            .ok_or_else(|| tensor_bounds_err(tensor, usize::MAX, "aligned block count overflow"))?;
        let mut blocks = vec![Aligned128Block([0_u8; WEIGHT_CACHE_ALIGNMENT_BYTES]); total_blocks];

        for r in 0..rows {
            let src_row = &raw[r * row_bytes..(r + 1) * row_bytes];
            let dst_blocks = &mut blocks[r * blocks_per_row..(r + 1) * blocks_per_row];
            for (blk_idx, chunk) in src_row.chunks(WEIGHT_CACHE_ALIGNMENT_BYTES).enumerate() {
                dst_blocks[blk_idx].0[..chunk.len()].copy_from_slice(chunk);
            }
        }

        Ok(Self {
            rows,
            cols,
            quant,
            is_compact_half,
            row_bytes,
            blocks_per_row,
            blocks,
        })
    }

    fn aligned_bytes(&self) -> usize {
        self.blocks.len() * size_of::<Aligned128Block>()
    }

    fn payload_ptr(&self) -> *const u8 {
        self.blocks.as_ptr().cast::<u8>()
    }

    fn gemv_f32(&self, out: &mut [f32], input: &[f32]) -> Result<(), K3Error> {
        if out.len() != self.rows || input.len() != self.cols {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "CachedDenseProjection::gemv".to_owned(),
                offset: 0,
                byte_len: size_of_val(out),
                buffer_len: self.rows * size_of::<f32>(),
                reason: format!(
                    "GEMV shape mismatch: out={}, input={}, expected rows={}, cols={}",
                    out.len(),
                    input.len(),
                    self.rows,
                    self.cols
                ),
            });
        }
        let mut row_raw = vec![0_u8; self.row_bytes];
        let mut row_f32 = vec![0.0_f32; self.cols];
        for (r, dst) in out.iter_mut().enumerate() {
            let row_blocks = &self.blocks[r * self.blocks_per_row..(r + 1) * self.blocks_per_row];
            let mut copied = 0_usize;
            for blk in row_blocks {
                let take = (self.row_bytes - copied).min(WEIGHT_CACHE_ALIGNMENT_BYTES);
                if take == 0 {
                    break;
                }
                row_raw[copied..copied + take].copy_from_slice(&blk.0[..take]);
                copied += take;
            }
            match self.quant {
                QuantType::F32 => {
                    for (i, slot) in row_f32.iter_mut().enumerate() {
                        let off = i * 4;
                        *slot = f32::from_le_bytes([
                            row_raw[off],
                            row_raw[off + 1],
                            row_raw[off + 2],
                            row_raw[off + 3],
                        ]);
                    }
                }
                QuantType::F16 => {
                    for (i, slot) in row_f32.iter_mut().enumerate() {
                        let off = i * 2;
                        *slot = f16_to_f32_portable(u16::from_le_bytes([
                            row_raw[off],
                            row_raw[off + 1],
                        ]));
                    }
                }
                QuantType::BF16 => {
                    for (i, slot) in row_f32.iter_mut().enumerate() {
                        let off = i * 2;
                        *slot = bf16_to_f32_portable(u16::from_le_bytes([
                            row_raw[off],
                            row_raw[off + 1],
                        ]));
                    }
                }
                QuantType::Q4_0
                | QuantType::Q8_0
                | QuantType::Q4_K_M
                | QuantType::Q5_K
                | QuantType::Q6_K => {
                    decode_quant_row_portable(self.quant, &row_raw, &mut row_f32);
                }
                other => {
                    let _ = TensorShape::Inline([0; 8], 0);
                    return Err(K3Error::CapabilityMismatch {
                        architecture: "mistral".to_owned(),
                        reason: format!(
                            "unsupported cached dense projection quant {}",
                            other.as_str()
                        ),
                    });
                }
            }
            *dst = dot_f32_compat(&row_f32, input);
        }
        Ok(())
    }
}

/// Cached `128-byte`-aligned dense projection matrix descriptor.
///
/// Wraps `jero_kernels_rvv::RepackedWeight` under `with-kernels` and a `128-byte`-aligned
/// portable descriptor when built with `--no-default-features`.
#[derive(Debug)]
pub struct CachedDenseProjection {
    #[cfg(feature = "with-kernels")]
    inner: RepackedWeight,
    #[cfg(not(feature = "with-kernels"))]
    inner: PortableAlignedMatrix,
}

impl CachedDenseProjection {
    /// Repacks or wraps `tensor` (`[rows, cols]`) with a configurable `F16`/`BF16` dense-expansion limit.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error`] if `tensor` dimensions, alignment, byte bounds, or quantization are invalid.
    pub fn from_tensor_with_half_limit(
        tensor: &TensorView<'_>,
        rows: usize,
        cols: usize,
        dense_f32_max_elems: usize,
    ) -> Result<Self, K3Error> {
        let quant = tensor.quant();
        if !matches!(
            quant,
            QuantType::F32
                | QuantType::F16
                | QuantType::BF16
                | QuantType::Q4_0
                | QuantType::Q4_K_M
                | QuantType::Q5_K
                | QuantType::Q6_K
                | QuantType::Q8_0
        ) {
            return Err(K3Error::CapabilityMismatch {
                architecture: "mistral".to_owned(),
                reason: format!(
                    "unsupported quant type {} for matrix '{}'",
                    quant.as_str(),
                    tensor.name()
                ),
            });
        }
        #[cfg(feature = "with-kernels")]
        {
            let inner = RepackedWeight::from_tensor_with_half_limit(
                tensor,
                rows,
                cols,
                dense_f32_max_elems,
            )?;
            Ok(Self { inner })
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            let inner = PortableAlignedMatrix::from_tensor_with_half_limit(
                tensor,
                rows,
                cols,
                dense_f32_max_elems,
            )?;
            Ok(Self { inner })
        }
    }

    /// Number of output rows `N`.
    #[must_use]
    pub const fn rows(&self) -> usize {
        #[cfg(feature = "with-kernels")]
        {
            self.inner.rows()
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            self.inner.rows
        }
    }

    /// Number of logical input columns `K`.
    #[must_use]
    pub const fn cols(&self) -> usize {
        #[cfg(feature = "with-kernels")]
        {
            self.inner.cols()
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            self.inner.cols
        }
    }

    /// Source quantization format.
    #[must_use]
    pub const fn quant(&self) -> QuantType {
        #[cfg(feature = "with-kernels")]
        {
            self.inner.quant()
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            self.inner.quant
        }
    }

    /// Total bytes occupied by the `128-byte`-aligned resident descriptor.
    #[must_use]
    pub fn aligned_bytes(&self) -> usize {
        self.inner.aligned_bytes()
    }

    /// Alignment guarantee in bytes (`128`).
    #[must_use]
    pub const fn alignment_bytes(&self) -> usize {
        WEIGHT_CACHE_ALIGNMENT_BYTES
    }

    /// Base payload pointer (`ptr as usize % 128 == 0`).
    #[must_use]
    pub fn payload_ptr(&self) -> *const u8 {
        self.inner.payload_ptr()
    }

    /// Returns `true` when the resident weight payload starts on a `128-byte` boundary.
    #[must_use]
    pub fn is_128_byte_aligned(&self) -> bool {
        (self.payload_ptr() as usize) % WEIGHT_CACHE_ALIGNMENT_BYTES == 0
    }

    /// Returns `true` if `F16`/`BF16` weights are kept in compact `16-bit` mode rather than `f32`.
    #[must_use]
    pub const fn is_compact_half(&self) -> bool {
        #[cfg(feature = "with-kernels")]
        {
            self.inner.is_compact_half()
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            self.inner.is_compact_half
        }
    }

    /// Borrows the underlying `jero_kernels_rvv::RepackedWeight`.
    #[cfg(feature = "with-kernels")]
    #[must_use]
    pub const fn repacked_weight(&self) -> &RepackedWeight {
        &self.inner
    }

    /// Executes `out = weights * input` from an `f32` activation row.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error`] on dimension or buffer mismatch.
    pub fn gemv_f32(&self, out: &mut [f32], input: &[f32]) -> Result<(), K3Error> {
        #[cfg(feature = "with-kernels")]
        {
            let mut scratch = MistralActScratch::new();
            scratch.prepare_row(input)?;
            self.gemv_with_scratch(out, &scratch)
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            self.inner.gemv_f32(out, input)
        }
    }

    /// Executes `out = weights * act` using a pre-quantized [`MistralActScratch`].
    ///
    /// # Errors
    ///
    /// Returns [`K3Error`] on dimension or buffer mismatch.
    pub fn gemv_with_scratch(
        &self,
        out: &mut [f32],
        scratch: &MistralActScratch,
    ) -> Result<(), K3Error> {
        #[cfg(feature = "with-kernels")]
        {
            self.inner.gemv(out, &scratch.q8)
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            self.inner.gemv_f32(out, &scratch.f32_buf)
        }
    }
}

/// Outcome of requesting a dense projection from [`MistralLayerCache`].
#[derive(Debug)]
pub enum CachedProjectionLookup {
    /// Cached `128-byte`-aligned descriptor ready for repeated GEMV execution.
    Repacked(Arc<CachedDenseProjection>),
    /// Matrix kept uncached because adding it would exceed the memlock repack budget;
    /// execute directly on borrowed [`TensorView`] bytes without duplicating weight RAM.
    DirectFallback,
}

/// Model-scoped cache for `Mistral` / `Mistral3` / `Mixtral` layer weights and norms.
///
/// Caches decoded `f32` `attn_norm`, `ffn_norm`, and `output_norm` vectors once per model load,
/// caches dense projections (`attn_q/k/v/output`, dense `ffn_gate/up/down`, and `ffn_gate_inp`)
/// as `128-byte`-aligned [`CachedDenseProjection`] entries within a bounded byte budget, and
/// resolves packed `3D` `_exps` expert matrices zero-copy via [`slice_packed_expert`].
#[derive(Debug)]
pub struct MistralLayerCache {
    model_anchor: Option<TensorIdentity>,
    norms: HashMap<String, (TensorIdentity, Arc<[f32]>)>,
    dense_projections: HashMap<String, (WeightCacheKey, Arc<CachedDenseProjection>)>,
    dense_order: VecDeque<String>,
    resident_repacked_bytes: usize,
    max_repacked_bytes: usize,
    dense_f32_max_elems: usize,
    stats: MistralCacheStats,
}

impl Default for MistralLayerCache {
    fn default() -> Self {
        Self::new()
    }
}

impl MistralLayerCache {
    /// Creates an empty [`MistralLayerCache`] with default memlock repack and half-expansion caps.
    #[must_use]
    pub fn new() -> Self {
        Self {
            model_anchor: None,
            norms: HashMap::new(),
            dense_projections: HashMap::new(),
            dense_order: VecDeque::new(),
            resident_repacked_bytes: 0,
            max_repacked_bytes: DEFAULT_MAX_REPACKED_CACHE_BYTES,
            dense_f32_max_elems: DEFAULT_DENSE_F32_MAX_ELEMENTS,
            stats: MistralCacheStats::default(),
        }
    }

    /// Configures the maximum resident `RepackedWeight` cache footprint in bytes.
    ///
    /// Setting `0` enforces strict zero resident expansion (all projections use zero-copy views
    /// while 1D `RMSNorm` scales remain cached).
    #[must_use]
    pub const fn with_max_repacked_bytes(mut self, max_bytes: usize) -> Self {
        self.max_repacked_bytes = max_bytes;
        self
    }

    /// Configures the `F16`/`BF16` element threshold below which half-precision matrices expand
    /// to aligned `f32` rows (above this threshold they stay in compact `u16` tiles).
    #[must_use]
    pub const fn with_dense_f32_max_elements(mut self, max_elems: usize) -> Self {
        self.dense_f32_max_elems = max_elems;
        self
    }

    /// Derives a cache budget from remaining `RLIMIT_MEMLOCK` headroom (`limit - base_resident`).
    #[must_use]
    pub fn with_memlock_headroom(limit_bytes: u64, resident_model_and_kv_bytes: u64) -> Self {
        let headroom = limit_bytes.saturating_sub(resident_model_and_kv_bytes);
        let cap = usize::try_from(headroom)
            .unwrap_or(usize::MAX)
            .min(DEFAULT_MAX_REPACKED_CACHE_BYTES);
        Self::new().with_max_repacked_bytes(cap)
    }

    /// Returns the number of cached decoded 1D norm vectors.
    #[must_use]
    pub fn cached_norm_count(&self) -> usize {
        self.norms.len()
    }

    /// Returns the number of cached dense projection matrices.
    #[must_use]
    pub fn cached_dense_count(&self) -> usize {
        self.dense_projections.len()
    }

    /// Returns total resident bytes occupied by cached `128-byte`-aligned projections.
    #[must_use]
    pub const fn resident_repacked_bytes(&self) -> usize {
        self.resident_repacked_bytes
    }

    /// Returns cumulative hit/miss/slice telemetry for this cache instance.
    #[must_use]
    pub const fn stats(&self) -> MistralCacheStats {
        self.stats
    }

    /// Clears all cached norms and dense projections and resets the bound model anchor.
    pub fn clear(&mut self) {
        self.model_anchor = None;
        self.norms.clear();
        self.dense_projections.clear();
        self.dense_order.clear();
        self.resident_repacked_bytes = 0;
    }

    /// Verifies whether `self` is bound to `weights`, invalidating stale entries on model swap.
    ///
    /// # Errors
    ///
    /// Propagates errors from inspecting anchor tensors if an anchor tensor exists yet fails bounds.
    pub fn bind_weights(&mut self, weights: &dyn ModelWeights) -> Result<(), K3Error> {
        let next_anchor = detect_model_anchor(weights)?;
        if let Some(next_id) = next_anchor {
            if let Some(prev_id) = self.model_anchor
                && prev_id != next_id
            {
                self.norms.clear();
                self.dense_projections.clear();
                self.dense_order.clear();
                self.resident_repacked_bytes = 0;
                self.stats.model_invalidations += 1;
            }
            self.model_anchor = Some(next_id);
        }
        Ok(())
    }

    /// Returns decoded `f32` weights for `tensor_name` (`attn_norm`, `ffn_norm`, or `output_norm`),
    /// decoding once per model load and serving subsequent lookups from cache.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error`] if the norm tensor is missing, malformed, or mismatches `expected_len`.
    pub fn get_or_decode_norm(
        &mut self,
        weights: &dyn ModelWeights,
        tensor_name: &str,
        expected_len: usize,
    ) -> Result<Arc<[f32]>, K3Error> {
        self.bind_weights(weights)?;
        let tensor = weights.tensor(tensor_name)?;
        let identity = TensorIdentity::from_view(&tensor);
        if let Some((cached_id, vec_arc)) = self.norms.get(tensor_name)
            && *cached_id == identity
            && vec_arc.len() == expected_len
        {
            self.stats.norm_hits += 1;
            return Ok(Arc::clone(vec_arc));
        }

        let mut decoded = vec![0.0_f32; expected_len];
        decode_norm_weights(&tensor, &mut decoded)?;
        let arc_vec: Arc<[f32]> = Arc::from(decoded);
        self.norms
            .insert(tensor_name.to_owned(), (identity, Arc::clone(&arc_vec)));
        self.stats.norm_decodes += 1;
        Ok(arc_vec)
    }

    /// Retrieves or repacks dense projection `tensor_name` into `128-byte`-aligned storage while
    /// enforcing the configured memlock repack budget.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error`] if the tensor is missing, malformed, or mismatches `(rows, cols)`.
    pub fn get_or_repack_dense(
        &mut self,
        weights: &dyn ModelWeights,
        tensor_name: &str,
        expected_rows: usize,
        expected_cols: usize,
    ) -> Result<CachedProjectionLookup, K3Error> {
        self.bind_weights(weights)?;
        let tensor = weights.tensor(tensor_name)?;
        let (rows, cols) = resolve_2d_dims(&tensor, expected_rows, expected_cols)?;
        let key = WeightCacheKey {
            identity: TensorIdentity::from_view(&tensor),
            rows,
            cols,
        };

        if let Some((cached_key, cached_proj)) = self.dense_projections.get(tensor_name)
            && *cached_key == key
        {
            self.stats.dense_hits += 1;
            return Ok(CachedProjectionLookup::Repacked(Arc::clone(cached_proj)));
        }

        if self.max_repacked_bytes == 0 {
            self.stats.repack_budget_skips += 1;
            return Ok(CachedProjectionLookup::DirectFallback);
        }

        let proj = Arc::new(CachedDenseProjection::from_tensor_with_half_limit(
            &tensor,
            rows,
            cols,
            self.dense_f32_max_elems,
        )?);
        let entry_bytes = proj.aligned_bytes();
        if entry_bytes > self.max_repacked_bytes {
            self.stats.repack_budget_skips += 1;
            return Ok(CachedProjectionLookup::DirectFallback);
        }

        self.remove_dense_entry(tensor_name);
        while self.resident_repacked_bytes.saturating_add(entry_bytes) > self.max_repacked_bytes
            || self.dense_projections.len() >= MAX_CACHED_DENSE_TENSORS
        {
            if !self.evict_oldest_dense() {
                break;
            }
        }
        if self.resident_repacked_bytes.saturating_add(entry_bytes) > self.max_repacked_bytes {
            self.stats.repack_budget_skips += 1;
            return Ok(CachedProjectionLookup::DirectFallback);
        }

        self.resident_repacked_bytes = self.resident_repacked_bytes.saturating_add(entry_bytes);
        self.dense_order.push_back(tensor_name.to_owned());
        self.dense_projections
            .insert(tensor_name.to_owned(), (key, Arc::clone(&proj)));
        self.stats.dense_repacks += 1;
        Ok(CachedProjectionLookup::Repacked(proj))
    }

    /// Executes `output = weights[tensor_name] * input`, caching `tensor_name` in `128-byte`
    /// repacked storage when within budget and optionally reusing `scratch`.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error`] on missing tensor, dimension mismatch, or kernel failure.
    pub fn matvec_dense_cached(
        &mut self,
        weights: &dyn ModelWeights,
        tensor_name: &str,
        output: &mut [f32],
        input: &[f32],
        scratch: Option<&MistralActScratch>,
    ) -> Result<(), K3Error> {
        match self.get_or_repack_dense(weights, tensor_name, output.len(), input.len())? {
            CachedProjectionLookup::Repacked(proj) => match scratch {
                Some(act) => proj.gemv_with_scratch(output, act),
                None => proj.gemv_f32(output, input),
            },
            CachedProjectionLookup::DirectFallback => {
                let tv = weights.tensor(tensor_name)?;
                matvec_tensor(output, &tv, input)
            }
        }
    }

    /// Slices active expert `expert_idx` for layer projection `proj` (`"gate"`, `"up"`, `"down"`)
    /// zero-copy from stacked `3D` `_exps` GGUF tensors or direct split `2D` expert tensors.
    ///
    /// Never unpacks all `num_experts` into resident `i8` cache tiles, keeping `Mixtral-8x7B`
    /// safe beneath the `28 GiB` `RLIMIT_MEMLOCK` ceiling.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if no matching tensor exists, `expert_idx >= num_experts`,
    /// or the sliced expert dimensions do not match `(expected_rows, expected_cols)`.
    #[expect(
        clippy::too_many_arguments,
        reason = "layer expert lookup combines layer, projection name, expert bounds, and shape checks"
    )]
    pub fn slice_expert_projection<'a>(
        &mut self,
        weights: &'a dyn ModelWeights,
        layer: usize,
        proj: &str,
        expert_idx: usize,
        num_experts: usize,
        expected_rows: usize,
        expected_cols: usize,
    ) -> Result<TensorView<'a>, K3Error> {
        self.stats.expert_zero_copy_slices += 1;

        let stacked = format!("blk.{layer}.ffn_{proj}_exps.weight");
        if weights.contains_tensor(&stacked) {
            let tensor = weights.tensor(&stacked)?;
            return slice_packed_expert(
                &tensor,
                expert_idx,
                num_experts,
                expected_rows,
                expected_cols,
            );
        }

        let direct_indexed = format!("blk.{layer}.ffn_{proj}.{expert_idx}.weight");
        if weights.contains_tensor(&direct_indexed) {
            let tensor = weights.tensor(&direct_indexed)?;
            let _ = resolve_2d_dims(&tensor, expected_rows, expected_cols)?;
            return Ok(tensor);
        }

        let exps_indexed = format!("blk.{layer}.ffn_{proj}_exps.{expert_idx}.weight");
        if weights.contains_tensor(&exps_indexed) {
            let tensor = weights.tensor(&exps_indexed)?;
            let _ = resolve_2d_dims(&tensor, expected_rows, expected_cols)?;
            return Ok(tensor);
        }

        weights.tensor(&direct_indexed)
    }

    fn remove_dense_entry(&mut self, name: &str) {
        if let Some((_, old)) = self.dense_projections.remove(name) {
            self.resident_repacked_bytes = self
                .resident_repacked_bytes
                .saturating_sub(old.aligned_bytes());
        }
        self.dense_order.retain(|existing| existing != name);
    }

    fn evict_oldest_dense(&mut self) -> bool {
        while let Some(oldest) = self.dense_order.pop_front() {
            if let Some((_, old)) = self.dense_projections.remove(&oldest) {
                self.resident_repacked_bytes = self
                    .resident_repacked_bytes
                    .saturating_sub(old.aligned_bytes());
                return true;
            }
        }
        false
    }
}

/// Validates and slices expert `expert_idx` zero-copy from a packed `3D` `_exps` [`TensorView`].
///
/// Accepts `[n_embd, ffn_dim, n_expert]` (GGUF packed `3D` layout) as well as transposed
/// `[n_expert, rows, cols]` and verifies exact byte-offset bounds plus `(expected_rows, expected_cols)`.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if the tensor is not `3D`, `expert_idx >= num_experts`,
/// byte offsets overflow or overrun the buffer, or dimensions mismatch `(expected_rows, expected_cols)`.
pub fn slice_packed_expert<'a>(
    tensor: &TensorView<'a>,
    expert_idx: usize,
    num_experts: usize,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<TensorView<'a>, K3Error> {
    let sliced = tensor.slice_expert(expert_idx, num_experts)?;
    let _ = resolve_2d_dims(&sliced, expected_rows, expected_cols)?;
    Ok(sliced)
}

/// Computes and verifies exact zero-copy byte-offset metadata for expert `expert_idx` inside a
/// packed `3D` `_exps` tensor without copying weights.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if bounds, shape, or expert count verification fails.
pub fn inspect_packed_expert_slice(
    tensor: &TensorView<'_>,
    expert_idx: usize,
    num_experts: usize,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<ExpertSliceMeta, K3Error> {
    let sliced = slice_packed_expert(
        tensor,
        expert_idx,
        num_experts,
        expected_rows,
        expected_cols,
    )?;
    let slice_len = sliced.as_bytes().len();
    let byte_offset_within_tensor =
        sliced
            .offset()
            .checked_sub(tensor.offset())
            .ok_or_else(|| {
                tensor_bounds_err(
                    tensor,
                    sliced.offset(),
                    "sliced expert offset precedes parent 3D tensor offset",
                )
            })?;
    if byte_offset_within_tensor != expert_idx * slice_len {
        return Err(tensor_bounds_err(
            tensor,
            byte_offset_within_tensor,
            &format!(
                "expert {expert_idx} byte offset {byte_offset_within_tensor} != expected {}",
                expert_idx * slice_len
            ),
        ));
    }

    Ok(ExpertSliceMeta {
        expert_idx,
        num_experts,
        rows: expected_rows,
        cols: expected_cols,
        byte_offset_within_tensor,
        absolute_offset: sliced.offset(),
        slice_byte_len: slice_len,
        quant: sliced.quant(),
    })
}

fn detect_model_anchor(weights: &dyn ModelWeights) -> Result<Option<TensorIdentity>, K3Error> {
    for &candidate in MODEL_ANCHOR_CANDIDATES {
        if weights.contains_tensor(candidate) {
            let view = weights.tensor(candidate)?;
            return Ok(Some(TensorIdentity::from_view(&view)));
        }
    }
    Ok(None)
}

fn compute_tensor_fingerprint64(name: &str, quant: QuantType, shape: &[usize], raw: &[u8]) -> u64 {
    let mut hash = 0xCBF2_9CE4_8422_2325_u64;
    for &b in name.as_bytes() {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0100_0000_01B3);
    }
    hash ^= quant.as_str().len() as u64;
    hash = hash.wrapping_mul(0x0100_0000_01B3);
    for &dim in shape {
        hash ^= dim as u64;
        hash = hash.wrapping_mul(0x0100_0000_01B3);
    }
    hash ^= raw.len() as u64;
    hash = hash.wrapping_mul(0x0100_0000_01B3);

    if raw.len() <= 256 {
        for &b in raw {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(0x0100_0000_01B3);
        }
        return hash;
    }

    let samples = 64_usize;
    let step = (raw.len() - 8) / samples;
    for s in 0..=samples {
        let off = (s * step).min(raw.len() - 8);
        let mut word = [0_u8; 8];
        word.copy_from_slice(&raw[off..off + 8]);
        hash ^= u64::from_le_bytes(word);
        hash = hash.wrapping_mul(0x0100_0000_01B3);
    }
    hash
}

#[derive(Default)]
struct SharedModelPool {
    slots: Vec<(Option<TensorIdentity>, Arc<Mutex<MistralLayerCache>>)>,
}

impl SharedModelPool {
    fn acquire(&mut self, anchor: Option<TensorIdentity>) -> Arc<Mutex<MistralLayerCache>> {
        if let Some(pos) = self.slots.iter().position(|(id, _)| *id == anchor) {
            let entry = self.slots.remove(pos);
            self.slots.push(entry);
            return Arc::clone(&self.slots[self.slots.len() - 1].1);
        }

        if self.slots.len() >= MAX_ACTIVE_GLOBAL_MODELS {
            self.slots.remove(0);
        }
        let cache = Arc::new(Mutex::new(MistralLayerCache::new()));
        self.slots.push((anchor, Arc::clone(&cache)));
        cache
    }

    fn clear(&mut self) {
        self.slots.clear();
    }
}

static GLOBAL_MISTRAL_POOL: OnceLock<Mutex<SharedModelPool>> = OnceLock::new();

fn global_pool() -> &'static Mutex<SharedModelPool> {
    GLOBAL_MISTRAL_POOL.get_or_init(|| Mutex::new(SharedModelPool::default()))
}

/// Clears all entries in the shared process-wide layer cache pool.
pub fn clear_global_layer_cache() {
    if let Ok(mut guard) = global_pool().lock() {
        guard.clear();
    }
}

/// Executes `op` with the shared [`MistralLayerCache`] scoped to `weights`.
pub(crate) fn with_shared_model_cache<R>(
    weights: &dyn ModelWeights,
    op: impl FnOnce(&mut MistralLayerCache) -> Result<R, K3Error>,
) -> Result<R, K3Error> {
    let anchor = detect_model_anchor(weights)?;
    let slot = {
        let mut pool = global_pool()
            .lock()
            .map_err(|_| K3Error::CapabilityMismatch {
                architecture: "mistral".to_owned(),
                reason: "shared MistralLayerCache pool lock poisoned".to_owned(),
            })?;
        pool.acquire(anchor)
    };
    let mut cache = slot.lock().map_err(|_| K3Error::CapabilityMismatch {
        architecture: "mistral".to_owned(),
        reason: "model MistralLayerCache slot lock poisoned".to_owned(),
    })?;
    op(&mut cache)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use jero_core::{ExecutionContext, ModelWeights, QuantType, TensorView};

    use super::*;
    use crate::kernels_compat::{dequantize_embedding_row, dot_f32_compat};
    use crate::recipe::MistralRecipe;

    struct RawTensor {
        name: String,
        quant: QuantType,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    }

    #[derive(Default)]
    struct CacheTestWeights {
        tensors: HashMap<String, RawTensor>,
    }

    impl CacheTestWeights {
        fn insert(&mut self, name: &str, quant: QuantType, shape: Vec<usize>, bytes: Vec<u8>) {
            self.tensors.insert(
                name.to_owned(),
                RawTensor {
                    name: name.to_owned(),
                    quant,
                    shape,
                    bytes,
                },
            );
        }

        fn mutate_byte(&mut self, name: &str, idx: usize) {
            if let Some(t) = self.tensors.get_mut(name) {
                t.bytes[idx] ^= 0x5A;
            }
        }
    }

    impl ModelWeights for CacheTestWeights {
        fn contains_tensor(&self, name: &str) -> bool {
            self.tensors.contains_key(name)
        }

        fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
            let t = self
                .tensors
                .get(name)
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: name.to_owned(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: 0,
                    reason: format!("missing tensor '{name}' in CacheTestWeights"),
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

    fn f32_to_f16_bits(val: f32) -> u16 {
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

    fn f32_to_bf16_bits(val: f32) -> u16 {
        u16::try_from(val.to_bits() >> 16).unwrap_or(0)
    }

    #[expect(clippy::cast_precision_loss, reason = "small synthetic dimensions")]
    fn encode_f32_bytes(len: usize, seed: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(len * 4);
        for i in 0..len {
            let v = (((i * 7 + seed * 13) % 19) as f32 - 9.0) * 0.05;
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[expect(clippy::cast_precision_loss, reason = "small synthetic dimensions")]
    fn encode_matrix_bytes(quant: QuantType, rows: usize, cols: usize, seed: usize) -> Vec<u8> {
        match quant {
            QuantType::F32 => encode_f32_bytes(rows * cols, seed),
            QuantType::F16 => {
                let mut out = Vec::with_capacity(rows * cols * 2);
                for i in 0..rows * cols {
                    let v = (((i * 5 + seed * 11) % 17) as f32 - 8.0) * 0.04;
                    out.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
                }
                out
            }
            QuantType::BF16 => {
                let mut out = Vec::with_capacity(rows * cols * 2);
                for i in 0..rows * cols {
                    let v = (((i * 5 + seed * 11) % 17) as f32 - 8.0) * 0.04;
                    out.extend_from_slice(&f32_to_bf16_bits(v).to_le_bytes());
                }
                out
            }
            QuantType::Q4_0 => {
                let bpr = cols / 32;
                let mut out = Vec::with_capacity(rows * bpr * 18);
                for b in 0..rows * bpr {
                    let d = f32_to_f16_bits(0.03 + ((b % 4) as f32) * 0.005);
                    out.extend_from_slice(&d.to_le_bytes());
                    for k in 0..16 {
                        let lo = u8::try_from((b * 3 + k + seed) % 16).unwrap_or(0);
                        let hi = u8::try_from((b * 5 + k + seed + 2) % 16).unwrap_or(0);
                        out.push(lo | (hi << 4));
                    }
                }
                out
            }
            QuantType::Q8_0 => {
                let bpr = cols / 32;
                let mut out = Vec::with_capacity(rows * bpr * 34);
                for b in 0..rows * bpr {
                    let d = f32_to_f16_bits(0.02 + ((b % 3) as f32) * 0.005);
                    out.extend_from_slice(&d.to_le_bytes());
                    for k in 0..32 {
                        let q = i8::try_from(
                            (i32::try_from((b * 7 + k + seed) % 21).unwrap_or(0)) - 10,
                        )
                        .unwrap_or(0);
                        out.push(q.to_le_bytes()[0]);
                    }
                }
                out
            }
            other => panic!("unexpected quant {other:?} in unit test"),
        }
    }

    fn build_compact_dense_model(seed: usize) -> (MistralRecipe, CacheTestWeights) {
        let mut recipe = MistralRecipe::new().with_sliding_window(None);
        recipe.num_layers = 1;
        recipe.hidden_dim = 32;
        recipe.ffn_dim = 32;
        recipe.n_heads_q = 2;
        recipe.n_heads_kv = 1;
        recipe.head_dim = 16;
        recipe.vocab_size = 4;

        let mut weights = CacheTestWeights::default();
        weights.insert(
            "token_embd.weight",
            QuantType::F32,
            vec![32, 4],
            encode_f32_bytes(4 * 32, seed + 1),
        );
        weights.insert(
            "output.weight",
            QuantType::Q4_0,
            vec![32, 4],
            encode_matrix_bytes(QuantType::Q4_0, 4, 32, seed + 2),
        );
        weights.insert(
            "output_norm.weight",
            QuantType::F32,
            vec![32],
            vec![0.0_f32; 32]
                .into_iter()
                .flat_map(|_| 1.0_f32.to_le_bytes())
                .collect(),
        );
        weights.insert(
            "blk.0.attn_norm.weight",
            QuantType::F32,
            vec![32],
            encode_f32_bytes(32, seed + 3),
        );
        weights.insert(
            "blk.0.ffn_norm.weight",
            QuantType::F32,
            vec![32],
            encode_f32_bytes(32, seed + 4),
        );
        for (idx, suffix) in [
            ("attn_q", (32, 32)),
            ("attn_k", (16, 32)),
            ("attn_v", (16, 32)),
            ("attn_output", (32, 32)),
            ("ffn_gate", (32, 32)),
            ("ffn_up", (32, 32)),
            ("ffn_down", (32, 32)),
        ] {
            weights.insert(
                &format!("blk.0.{idx}.weight"),
                QuantType::Q4_0,
                vec![suffix.1, suffix.0],
                encode_matrix_bytes(QuantType::Q4_0, suffix.0, suffix.1, seed + 10 + suffix.0),
            );
        }
        (recipe, weights)
    }

    #[test]
    #[expect(clippy::cast_precision_loss, reason = "small synthetic dimensions")]
    fn test_dense_projection_128_alignment_compact_half_and_gemv_miri() {
        let input: Vec<f32> = (0..32).map(|i| ((i % 7) as f32 - 3.0) * 0.15).collect();

        for quant in [
            QuantType::F32,
            QuantType::F16,
            QuantType::BF16,
            QuantType::Q4_0,
            QuantType::Q8_0,
        ] {
            let raw = encode_matrix_bytes(quant, 4, 32, 9);
            let tv = TensorView::from_bytes("blk.0.attn_q.weight", quant, &[32, 4], 0, &raw)
                .expect("valid view");
            let proj = CachedDenseProjection::from_tensor_with_half_limit(
                &tv,
                4,
                32,
                DEFAULT_DENSE_F32_MAX_ELEMENTS,
            )
            .expect("repack");
            assert_eq!(proj.alignment_bytes(), WEIGHT_CACHE_ALIGNMENT_BYTES);
            assert!(
                proj.is_128_byte_aligned(),
                "128B align failed for {quant:?}"
            );
            assert_eq!(proj.rows(), 4);
            assert_eq!(proj.cols(), 32);
            assert_eq!(proj.quant(), quant);

            let mut out = [0.0_f32; 4];
            proj.gemv_f32(&mut out, &input).expect("gemv");
            let mut row_f32 = [0.0_f32; 32];
            for (r, &actual) in out.iter().enumerate() {
                dequantize_embedding_row(&tv, r, &mut row_f32).expect("dequantize row");
                let expected = dot_f32_compat(&row_f32, &input);
                let tol = if matches!(quant, QuantType::F32 | QuantType::F16 | QuantType::BF16) {
                    1e-4
                } else {
                    2e-2
                };
                assert!(
                    (actual - expected).abs() < tol,
                    "{quant:?} row {r}: actual={actual}, expected={expected}"
                );
            }

            #[cfg(feature = "with-kernels")]
            if matches!(quant, QuantType::F32 | QuantType::Q4_0 | QuantType::Q8_0) {
                let mut dq = [0.0_f32; 32];
                proj.repacked_weight()
                    .dequantize_row(1, &mut dq)
                    .expect("dequantize");
                let mut ref_dq = [0.0_f32; 32];
                dequantize_embedding_row(&tv, 1, &mut ref_dq).expect("ref row");
                for c in 0..32 {
                    assert!((dq[c] - ref_dq[c]).abs() < 1e-5);
                }
            }
        }

        // Verify compact F16/BF16 threshold preserves 16-bit footprint above limit.
        for half_q in [QuantType::F16, QuantType::BF16] {
            let raw = encode_matrix_bytes(half_q, 4, 32, 3);
            let tv = TensorView::from_bytes("blk.0.ffn_down.weight", half_q, &[32, 4], 0, &raw)
                .expect("half view");
            let expanded = CachedDenseProjection::from_tensor_with_half_limit(&tv, 4, 32, 1024)
                .expect("expanded");
            let compact =
                CachedDenseProjection::from_tensor_with_half_limit(&tv, 4, 32, 0).expect("compact");
            assert!(!expanded.is_compact_half());
            assert!(compact.is_compact_half());
            assert!(compact.aligned_bytes() <= expanded.aligned_bytes());
        }

        // Fail closed on non-admitted quant.
        let bad_q2 = vec![0_u8; 84];
        let bad_view = TensorView::from_bytes("bad.q2k", QuantType::Q2_K, &[256, 1], 0, &bad_q2)
            .expect("q2 view");
        let err = CachedDenseProjection::from_tensor_with_half_limit(&bad_view, 1, 256, 1024)
            .expect_err("Q2_K rejected");
        assert!(matches!(err, K3Error::CapabilityMismatch { .. }));
    }

    #[test]
    fn test_layer_cache_norms_once_cross_model_and_aba_invalidation_miri() {
        let (recipe, mut weights_a) = build_compact_dense_model(1);
        let (_, weights_b) = build_compact_dense_model(99);
        let mut cache = MistralLayerCache::new();

        let mut kv_a = vec![0.0_f32; 2 * 6 * 16];
        let mut logits_a = vec![0.0_f32; 4];
        let mut ctx_a = ExecutionContext::new(&[0_u32], 6, &mut logits_a, &mut kv_a).unwrap();
        recipe
            .prefill_step_with_cache(&mut ctx_a, &weights_a, &mut cache)
            .expect("prefill a");
        assert_eq!(cache.cached_norm_count(), 3);
        let norms_after_prefill = cache.stats().norm_decodes;
        let dense_after_prefill = cache.stats().dense_repacks;
        assert_eq!(norms_after_prefill, 3);
        assert_eq!(dense_after_prefill, 7);

        // Repeated decode steps hit cached norms and repacked matrices with zero extra work.
        ctx_a.prepare_decode(1).unwrap();
        recipe
            .decode_step_with_cache(&mut ctx_a, &weights_a, &mut cache)
            .expect("decode 1");
        ctx_a.prepare_decode(2).unwrap();
        recipe
            .decode_step_with_cache(&mut ctx_a, &weights_a, &mut cache)
            .expect("decode 2");
        assert_eq!(cache.stats().norm_decodes, norms_after_prefill);
        assert_eq!(cache.stats().dense_repacks, dense_after_prefill);
        assert!(cache.stats().norm_hits >= 6);
        assert!(cache.stats().dense_hits >= 14);

        // ABA content mutation inside blk.0.attn_norm.weight triggers deterministic re-decode.
        weights_a.mutate_byte("blk.0.attn_norm.weight", 4);
        let _ = cache
            .get_or_decode_norm(&weights_a, "blk.0.attn_norm.weight", 32)
            .expect("re-decode on mutation");
        assert_eq!(cache.stats().norm_decodes, norms_after_prefill + 1);

        // Cross-model switch invalidates previous entries rather than serving stale weights.
        let mut kv_b = vec![0.0_f32; 2 * 6 * 16];
        let mut logits_b = vec![0.0_f32; 4];
        let mut ctx_b = ExecutionContext::new(&[0_u32], 6, &mut logits_b, &mut kv_b).unwrap();
        recipe
            .prefill_step_with_cache(&mut ctx_b, &weights_b, &mut cache)
            .expect("prefill b");
        assert_eq!(cache.stats().model_invalidations, 1);
        assert_eq!(cache.cached_norm_count(), 3);
        assert_ne!(logits_a, logits_b);
    }

    fn run_mixtral_prefill_with_cache(
        recipe: &MistralRecipe,
        weights: &CacheTestWeights,
        cache: &mut MistralLayerCache,
    ) -> Vec<f32> {
        let mut kv = vec![0.0_f32; 2 * 4 * 16];
        let mut logits = vec![0.0_f32; 4];
        let mut ctx = ExecutionContext::new(&[1_u32], 4, &mut logits, &mut kv).unwrap();
        recipe
            .prefill_step_with_cache(&mut ctx, weights, cache)
            .expect("mixtral prefill");
        logits
    }

    #[test]
    fn test_zero_copy_3d_exps_slicer_bounds_and_memlock_headroom_miri() {
        let mut recipe = MistralRecipe::new_mixtral()
            .with_sliding_window(Some(4))
            .with_experts(4, 2);
        recipe.num_layers = 1;
        recipe.hidden_dim = 32;
        recipe.ffn_dim = 16;
        recipe.n_heads_q = 2;
        recipe.n_heads_kv = 1;
        recipe.head_dim = 16;
        recipe.vocab_size = 4;

        let (_, shared) = build_compact_dense_model(7);
        let mut weights = CacheTestWeights::default();
        for (name, rt) in shared.tensors {
            if !name.contains("ffn_gate") && !name.contains("ffn_up") && !name.contains("ffn_down")
            {
                weights.insert(&name, rt.quant, rt.shape, rt.bytes);
            }
        }
        weights.insert(
            "blk.0.ffn_gate_inp.weight",
            QuantType::Q4_0,
            vec![32, 4],
            encode_matrix_bytes(QuantType::Q4_0, 4, 32, 33),
        );

        let mut expected_slices = HashMap::new();
        for proj in ["gate", "up", "down"] {
            let (rows, cols) = if proj == "down" { (32, 16) } else { (16, 32) };
            // Pack expert bytes cleanly; `down` uses Q4_0 only when cols divisible by 32,
            // so build F16 experts here for compact `16`-wide `down` under Miri.
            let mut stacked = Vec::new();
            for e in 0..4 {
                let slice_bytes =
                    encode_matrix_bytes(QuantType::F16, rows, cols, 50 + e * 7 + rows);
                expected_slices.insert((proj, e), slice_bytes.clone());
                stacked.extend_from_slice(&slice_bytes);
            }
            weights.insert(
                &format!("blk.0.ffn_{proj}_exps.weight"),
                QuantType::F16,
                vec![cols, rows, 4],
                stacked,
            );
        }

        let mut cache = MistralLayerCache::new();
        for e in 0..4 {
            let meta = inspect_packed_expert_slice(
                &weights.tensor("blk.0.ffn_gate_exps.weight").unwrap(),
                e,
                4,
                16,
                32,
            )
            .expect("inspect gate");
            assert_eq!(meta.expert_idx, e);
            assert_eq!(meta.byte_offset_within_tensor, e * (16 * 32 * 2));
            assert_eq!(meta.slice_byte_len, 16 * 32 * 2);

            let sliced = cache
                .slice_expert_projection(&weights, 0, "gate", e, 4, 16, 32)
                .expect("zero-copy slice");
            assert_eq!(sliced.as_bytes(), expected_slices[&("gate", e)].as_slice());
        }
        assert_eq!(cache.resident_repacked_bytes(), 0);
        assert_eq!(cache.stats().expert_zero_copy_slices, 4);

        // Out-of-bounds and wrong-dimension expert requests fail closed.
        let gate_view = weights.tensor("blk.0.ffn_gate_exps.weight").unwrap();
        assert!(matches!(
            slice_packed_expert(&gate_view, 4, 4, 16, 32),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert!(matches!(
            slice_packed_expert(&gate_view, 0, 4, 16, 64),
            Err(K3Error::TensorBoundsError { .. })
        ));
        let dense_view = weights.tensor("blk.0.attn_q.weight").unwrap();
        assert!(matches!(
            slice_packed_expert(&dense_view, 0, 1, 32, 32),
            Err(K3Error::TensorBoundsError { .. })
        ));

        // Memlock headroom mode skips expert repacking and handles zero-budget fallback cleanly.
        let mut fallback_cache = MistralLayerCache::with_memlock_headroom(
            28 * 1024 * 1024 * 1024,
            28 * 1024 * 1024 * 1024,
        );
        let logits_fallback =
            run_mixtral_prefill_with_cache(&recipe, &weights, &mut fallback_cache);
        assert_eq!(fallback_cache.resident_repacked_bytes(), 0);
        assert!(fallback_cache.stats().repack_budget_skips > 0);
        assert!(logits_fallback.iter().all(|v| v.is_finite()));

        // Bounded headroom enforces budget subtraction and zero expert allocation.
        let mut bounded_cache = MistralLayerCache::with_memlock_headroom(
            28 * 1024 * 1024 * 1024,
            26 * 1024 * 1024 * 1024,
        );
        let logits_bounded = run_mixtral_prefill_with_cache(&recipe, &weights, &mut bounded_cache);
        assert!(logits_bounded.iter().all(|v| v.is_finite()));
        assert!(bounded_cache.stats().expert_zero_copy_slices > 0);
    }
}
