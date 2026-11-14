//! Unified compute and dequantization dispatchers across `with-kernels` and portable fallback builds.

use jero_core::{K3Error, QuantType, TensorView};

#[cfg(feature = "with-kernels")]
pub use jero_kernels_rvv::RopeScalingConfig;
#[cfg(feature = "with-kernels")]
use jero_kernels_rvv::{
    AttentionMaskConfig, BaselineKernel, VectorKernel, bf16_to_f32_finite, dequantize_token_row,
    f16_to_f32_finite, gqa_causal_attention_f32, gqa_causal_attention_prefill_f32,
    precompute_rope_inv_freq, swiglu_f32, vec_dot_bf16, vec_dot_f16, vec_dot_q4_0, vec_dot_q5_k,
    vec_dot_q6_k, vec_dot_q8_0,
};

#[cfg(not(feature = "with-kernels"))]
use core::f32::consts::PI;

/// Rotary position embedding scaling configurations.
#[cfg(not(feature = "with-kernels"))]
#[derive(Debug, Clone, PartialEq)]
pub enum RopeScalingConfig {
    /// Linear frequency scaling (`inv_freq = inv_freq / factor`).
    Linear {
        /// Frequency scaling factor (`> 0.0`).
        factor: f32,
    },
    /// Smooth wavelength-ramped scaling (Llama-3 / `YaRN` NTK-by-parts).
    Llama3Smooth {
        /// Low-frequency scaling multiplier (`> 0.0`).
        factor: f32,
        /// Low-frequency ramp boundary multiplier (`> 0.0`).
        low_freq_factor: f32,
        /// High-frequency ramp boundary multiplier (`> low_freq_factor`).
        high_freq_factor: f32,
        /// Original pre-training context window length (`> 0`).
        original_max_position_embeddings: usize,
    },
    /// Dual-factor per-dimension scaling across short and extended contexts.
    LongRope {
        /// Scaling factors when `seq_len <= original_max_position_embeddings`.
        short_factor: Vec<f32>,
        /// Scaling factors when `seq_len > original_max_position_embeddings`.
        long_factor: Vec<f32>,
        /// Threshold separating short and long factor tables.
        original_max_position_embeddings: usize,
    },
}

/// Validates and resolves 2D weight matrix `(rows, cols)` accepting both GGUF `[cols, rows]`
/// and row-major `[rows, cols]` layouts, eliminating square-matrix and alignment ambiguities.
pub(crate) fn resolve_2d_dims(
    tensor: &TensorView<'_>,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<(usize, usize), K3Error> {
    if expected_rows == 0 || expected_cols == 0 {
        return Err(tensor_bounds_err(
            tensor,
            0,
            "matrix dimensions must be non-zero",
        ));
    }

    let shape = tensor.shape();
    if shape.len() == 2 {
        let d0 = shape[0];
        let d1 = shape[1];
        if (d0 == expected_cols && d1 == expected_rows)
            || (d0 == expected_rows && d1 == expected_cols)
        {
            return Ok((expected_rows, expected_cols));
        }
        return Err(tensor_bounds_err(
            tensor,
            tensor.byte_len(),
            &format!(
                "matrix shape [{d0}, {d1}] does not match expected [{expected_rows}, {expected_cols}]"
            ),
        ));
    }

    let total = tensor.num_elements();
    let expected_total = expected_rows
        .checked_mul(expected_cols)
        .ok_or_else(|| tensor_bounds_err(tensor, usize::MAX, "matrix element count overflowed"))?;
    if total != expected_total {
        return Err(tensor_bounds_err(
            tensor,
            tensor.byte_len(),
            &format!(
                "tensor elements {total} do not match expected {expected_rows}x{expected_cols}"
            ),
        ));
    }
    Ok((expected_rows, expected_cols))
}

/// Decodes a 1D floating-point tensor (`F32`, `F16`, or `BF16`) into `out`, failing closed on
/// dimension mismatch, unsupported quant types, or non-finite `F32` elements.
pub(crate) fn decode_norm_weights(tensor: &TensorView<'_>, out: &mut [f32]) -> Result<(), K3Error> {
    let expected = out.len();
    if tensor.num_elements() != expected {
        return Err(tensor_bounds_err(
            tensor,
            tensor.byte_len(),
            &format!(
                "1D tensor element count {} != expected {expected}",
                tensor.num_elements()
            ),
        ));
    }
    let raw = tensor.as_bytes();
    match tensor.quant() {
        QuantType::F32 => {
            if raw.len() < expected * 4 {
                return Err(tensor_bounds_err(
                    tensor,
                    expected * 4,
                    "truncated F32 tensor payload",
                ));
            }
            for (i, slot) in out.iter_mut().enumerate() {
                let off = i * 4;
                let val = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
                if !val.is_finite() {
                    return Err(tensor_bounds_err(
                        tensor,
                        off,
                        "non-finite F32 element encountered in weight tensor",
                    ));
                }
                *slot = val;
            }
        }
        QuantType::F16 => {
            if raw.len() < expected * 2 {
                return Err(tensor_bounds_err(
                    tensor,
                    expected * 2,
                    "truncated F16 tensor payload",
                ));
            }
            for (i, slot) in out.iter_mut().enumerate() {
                let off = i * 2;
                let bits = u16::from_le_bytes([raw[off], raw[off + 1]]);
                *slot = f16_to_f32_compat(bits);
            }
        }
        QuantType::BF16 => {
            if raw.len() < expected * 2 {
                return Err(tensor_bounds_err(
                    tensor,
                    expected * 2,
                    "truncated BF16 tensor payload",
                ));
            }
            for (i, slot) in out.iter_mut().enumerate() {
                let off = i * 2;
                let bits = u16::from_le_bytes([raw[off], raw[off + 1]]);
                *slot = bf16_to_f32_compat(bits);
            }
        }
        other => {
            return Err(K3Error::CapabilityMismatch {
                architecture: "mistral".to_owned(),
                reason: format!(
                    "unsupported quant type {} for 1D norm tensor '{}'",
                    other.as_str(),
                    tensor.name()
                ),
            });
        }
    }
    Ok(())
}

/// Dequantizes a single row from a token embedding table across all admitted Mistral quantizations.
pub(crate) fn dequantize_embedding_row(
    table: &TensorView<'_>,
    row_idx: usize,
    out: &mut [f32],
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        dequantize_token_row(table, row_idx, out)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        dequantize_row_portable(table, row_idx, out)
    }
}

/// Executes a 2D matrix-vector product `output = weights * input` across all admitted quantizations.
pub(crate) fn matvec_tensor(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    matmul_tensor(output, weights, 1, input, output.len(), input.len())
}

/// Executes a batched 2D matrix multiplication `output[M, rows] = input[M, cols] * weights[rows, cols]^T`
/// across all admitted quantizations.
pub(crate) fn matmul_tensor(
    output: &mut [f32],
    weights: &TensorView<'_>,
    m: usize,
    input: &[f32],
    expected_rows: usize,
    expected_cols: usize,
) -> Result<(), K3Error> {
    if m == 0 || expected_rows == 0 || expected_cols == 0 {
        return Err(tensor_bounds_err(
            weights,
            0,
            "batched matmul dimensions must be non-zero",
        ));
    }
    let expected_in = m
        .checked_mul(expected_cols)
        .ok_or_else(|| tensor_bounds_err(weights, usize::MAX, "matmul input length overflow"))?;
    let expected_out = m
        .checked_mul(expected_rows)
        .ok_or_else(|| tensor_bounds_err(weights, usize::MAX, "matmul output length overflow"))?;
    if input.len() != expected_in || output.len() != expected_out {
        return Err(tensor_bounds_err(
            weights,
            0,
            &format!(
                "matmul buffer mismatch: m={m}, in={} (expected {expected_in}), out={} (expected {expected_out})",
                input.len(),
                output.len()
            ),
        ));
    }

    let (rows, cols) = resolve_2d_dims(weights, expected_rows, expected_cols)?;

    if weights.quant() == QuantType::F32 {
        return matmul_f32_rows(output, weights, m, input, rows, cols);
    }

    #[cfg(feature = "with-kernels")]
    {
        for t in 0..m {
            let x_row = &input[t * cols..(t + 1) * cols];
            let out_row = &mut output[t * rows..(t + 1) * rows];
            match weights.quant() {
                QuantType::F16 => vec_dot_f16(out_row, weights, x_row)?,
                QuantType::BF16 => vec_dot_bf16(out_row, weights, x_row)?,
                QuantType::Q4_0 => vec_dot_q4_0(out_row, weights, x_row)?,
                QuantType::Q4_K_M => {
                    BaselineKernel::new().vec_dot_q4_k_m(out_row, weights, x_row)?;
                }
                QuantType::Q5_K => vec_dot_q5_k(out_row, weights, x_row)?,
                QuantType::Q6_K => vec_dot_q6_k(out_row, weights, x_row)?,
                QuantType::Q8_0 => vec_dot_q8_0(out_row, weights, x_row)?,
                other => {
                    return Err(K3Error::CapabilityMismatch {
                        architecture: "mistral".to_owned(),
                        reason: format!(
                            "unsupported quant type {} for matrix '{}'",
                            other.as_str(),
                            weights.name()
                        ),
                    });
                }
            }
        }
        Ok(())
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        matmul_portable(output, weights, m, input, rows, cols)
    }
}

/// Applies RMS normalization with pre-decoded weight scale `weight`.
pub(crate) fn rmsnorm_compat(
    out: &mut [f32],
    input: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        BaselineKernel::new().rmsnorm(out, input, weight, eps)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        let dim = input.len();
        if dim == 0 || out.len() != dim || weight.len() != dim || !eps.is_finite() || eps <= 0.0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: "rmsnorm".to_owned(),
                reason: "invalid RMSNorm dimensions or epsilon".to_owned(),
            });
        }
        let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
        #[expect(clippy::cast_precision_loss, reason = "dim fits cleanly in f32")]
        let inv_rms = 1.0_f32 / ((sum_sq / (dim as f32)) + eps).sqrt();
        for i in 0..dim {
            out[i] = input[i] * inv_rms * weight[i];
        }
        Ok(())
    }
}

/// Applies elementwise `SwiGLU` activation `out[i] = silu(gate[i]) * up[i]`.
pub(crate) fn swiglu_compat(out: &mut [f32], gate: &[f32], up: &[f32]) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        swiglu_f32(out, gate, up)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        if out.len() != gate.len() || out.len() != up.len() {
            return Err(K3Error::CapabilityMismatch {
                architecture: "swiglu".to_owned(),
                reason: "SwiGLU slice length mismatch".to_owned(),
            });
        }
        for i in 0..out.len() {
            let g = gate[i];
            let sig = if g <= -20.0 {
                0.0
            } else if g >= 20.0 {
                1.0
            } else {
                1.0 / (1.0 + (-g).exp())
            };
            out[i] = (g * sig) * up[i];
        }
        Ok(())
    }
}

/// Precomputes `RoPE` inverse frequency table `inv_freq[0..head_dim/2]`.
pub(crate) fn rope_inv_freq_compat(
    head_dim: usize,
    theta: f32,
    scaling: Option<&RopeScalingConfig>,
    seq_len: usize,
) -> Result<Vec<f32>, K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        precompute_rope_inv_freq(head_dim, theta, scaling, seq_len)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        precompute_rope_inv_freq_portable(head_dim, theta, scaling, seq_len)
    }
}

/// Executes grouped-query causal attention over the active KV cache slice.
#[expect(
    clippy::too_many_arguments,
    reason = "matches gqa_causal_attention_f32 signature"
)]
pub(crate) fn gqa_attention_compat(
    out: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        gqa_causal_attention_f32(
            out, q, k_cache, v_cache, n_heads_q, n_heads_kv, head_dim, seq_len, scale,
        )
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        gqa_attention_portable(
            out, q, k_cache, v_cache, n_heads_q, n_heads_kv, head_dim, seq_len, scale,
        )
    }
}

/// Executes multi-token batched causal GQA prefill over `[m, n_heads_q, head_dim]` with optional
/// sliding window and starting sequence offset `start_pos`.
#[expect(
    clippy::too_many_arguments,
    reason = "matches gqa_causal_attention_prefill_f32 signature"
)]
pub(crate) fn gqa_causal_attention_prefill_compat(
    out: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    m: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    total_seq_len: usize,
    start_pos: usize,
    sliding_window: Option<usize>,
    scale: f32,
    scores_scratch: &mut [f32],
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        let config =
            AttentionMaskConfig::causal(start_pos, scale).with_sliding_window(sliding_window);
        gqa_causal_attention_prefill_f32(
            out,
            q,
            k_cache,
            v_cache,
            m,
            n_heads_q,
            n_heads_kv,
            head_dim,
            total_seq_len,
            config,
            scores_scratch,
        )
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        gqa_attention_prefill_portable(
            out,
            q,
            k_cache,
            v_cache,
            m,
            n_heads_q,
            n_heads_kv,
            head_dim,
            total_seq_len,
            start_pos,
            sliding_window,
            scale,
            scores_scratch,
        )
    }
}

fn matmul_f32_rows(
    output: &mut [f32],
    weights: &TensorView<'_>,
    m: usize,
    input: &[f32],
    rows: usize,
    cols: usize,
) -> Result<(), K3Error> {
    let raw = weights.as_bytes();
    let row_bytes = cols * 4;
    if raw.len() < rows * row_bytes {
        return Err(tensor_bounds_err(
            weights,
            rows * row_bytes,
            "truncated F32 matrix payload",
        ));
    }

    let mut row_buf = vec![0.0_f32; cols];
    for r in 0..rows {
        let base = r * row_bytes;
        for (c, dst) in row_buf.iter_mut().enumerate() {
            let off = base + c * 4;
            let val = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
            if !val.is_finite() {
                return Err(tensor_bounds_err(
                    weights,
                    off,
                    "non-finite F32 element encountered in matrix weight",
                ));
            }
            *dst = val;
        }
        for t in 0..m {
            let x_row = &input[t * cols..(t + 1) * cols];
            output[t * rows + r] = dot_f32_compat(&row_buf, x_row);
        }
    }
    Ok(())
}

#[inline]
pub(crate) fn dot_f32_compat(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "with-kernels")]
    {
        BaselineKernel::new().dot_product_f32(a, b)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }
}

#[inline]
fn f16_to_f32_compat(bits: u16) -> f32 {
    #[cfg(feature = "with-kernels")]
    {
        f16_to_f32_finite(bits)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        f16_to_f32_portable(bits)
    }
}

#[inline]
fn bf16_to_f32_compat(bits: u16) -> f32 {
    #[cfg(feature = "with-kernels")]
    {
        bf16_to_f32_finite(bits)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        bf16_to_f32_portable(bits)
    }
}

pub(crate) fn tensor_bounds_err(tensor: &TensorView<'_>, offset: usize, reason: &str) -> K3Error {
    K3Error::TensorBoundsError {
        tensor_name: tensor.name().to_owned(),
        offset,
        byte_len: tensor.byte_len(),
        buffer_len: tensor.as_bytes().len(),
        reason: reason.to_owned(),
    }
}

#[cfg(not(feature = "with-kernels"))]
pub(crate) fn f16_to_f32_portable(bits: u16) -> f32 {
    let sign = u32::from((bits >> 15) & 0x0001);
    let exp = u32::from((bits >> 10) & 0x001F);
    let mant = u32::from(bits & 0x03FF);
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut m = mant;
        let mut e = 127_u32 - 14;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03FF;
        return f32::from_bits((sign << 31) | (e << 23) | (m << 13));
    }
    if exp == 0x1F {
        return if sign == 1 { -0.0625 } else { 0.0625 };
    }
    f32::from_bits((sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13))
}

#[cfg(not(feature = "with-kernels"))]
pub(crate) fn bf16_to_f32_portable(bits: u16) -> f32 {
    let exp = (bits >> 7) & 0x00FF;
    let sign = (bits >> 15) & 0x0001;
    if exp == 0x00FF {
        return if sign == 1 { -0.0625 } else { 0.0625 };
    }
    f32::from_bits(u32::from(bits) << 16)
}

#[cfg(not(feature = "with-kernels"))]
fn dequantize_row_portable(
    table: &TensorView<'_>,
    row_idx: usize,
    out: &mut [f32],
) -> Result<(), K3Error> {
    let cols = out.len();
    if cols == 0 {
        return Err(tensor_bounds_err(table, 0, "output slice cannot be empty"));
    }
    let total = table.num_elements();
    if total == 0 || total % cols != 0 {
        return Err(tensor_bounds_err(
            table,
            0,
            "tensor elements not a multiple of row width",
        ));
    }
    let num_rows = total / cols;
    let _ = resolve_2d_dims(table, num_rows, cols)?;
    if row_idx >= num_rows {
        return Err(tensor_bounds_err(
            table,
            row_idx,
            &format!("row index {row_idx} out of bounds for {num_rows} rows"),
        ));
    }

    let quant = table.quant();
    let block_elems = quant.block_elements();
    if cols % block_elems != 0 {
        return Err(tensor_bounds_err(
            table,
            0,
            &format!(
                "row width {cols} not divisible by {} block size {block_elems}",
                quant.as_str()
            ),
        ));
    }
    let block_bytes = quant.block_bytes();
    let row_bytes = (cols / block_elems) * block_bytes;
    let row_start = row_idx * row_bytes;
    let row_end = row_start + row_bytes;
    let raw = table.as_bytes();
    if row_end > raw.len() {
        return Err(tensor_bounds_err(table, row_start, "truncated row payload"));
    }

    let row_slice = &raw[row_start..row_end];
    match quant {
        QuantType::F32 => {
            for (i, dst) in out.iter_mut().enumerate() {
                let off = i * 4;
                *dst = f32::from_le_bytes([
                    row_slice[off],
                    row_slice[off + 1],
                    row_slice[off + 2],
                    row_slice[off + 3],
                ]);
            }
        }
        QuantType::F16 => {
            for (i, dst) in out.iter_mut().enumerate() {
                let off = i * 2;
                *dst =
                    f16_to_f32_portable(u16::from_le_bytes([row_slice[off], row_slice[off + 1]]));
            }
        }
        QuantType::BF16 => {
            for (i, dst) in out.iter_mut().enumerate() {
                let off = i * 2;
                *dst =
                    bf16_to_f32_portable(u16::from_le_bytes([row_slice[off], row_slice[off + 1]]));
            }
        }
        QuantType::Q4_0
        | QuantType::Q8_0
        | QuantType::Q4_K_M
        | QuantType::Q5_K
        | QuantType::Q6_K => {
            decode_quant_row_portable(quant, row_slice, out);
        }
        other => {
            return Err(K3Error::CapabilityMismatch {
                architecture: "mistral".to_owned(),
                reason: format!("unsupported embedding quant {}", other.as_str()),
            });
        }
    }
    Ok(())
}

#[cfg(not(feature = "with-kernels"))]
fn matmul_portable(
    output: &mut [f32],
    weights: &TensorView<'_>,
    m: usize,
    input: &[f32],
    rows: usize,
    cols: usize,
) -> Result<(), K3Error> {
    let quant = weights.quant();
    if !matches!(
        quant,
        QuantType::F16
            | QuantType::BF16
            | QuantType::Q4_0
            | QuantType::Q8_0
            | QuantType::Q4_K_M
            | QuantType::Q5_K
            | QuantType::Q6_K
    ) {
        return Err(K3Error::CapabilityMismatch {
            architecture: "mistral".to_owned(),
            reason: format!(
                "unsupported quant {} for matrix '{}'",
                quant.as_str(),
                weights.name()
            ),
        });
    }

    let block_elems = quant.block_elements();
    if cols % block_elems != 0 {
        return Err(tensor_bounds_err(
            weights,
            0,
            &format!(
                "input length {cols} is not divisible by {} block size {block_elems}",
                quant.as_str()
            ),
        ));
    }

    let mut row_buf = vec![0.0_f32; cols];
    for r in 0..rows {
        dequantize_row_portable(weights, r, &mut row_buf)?;
        for t in 0..m {
            let x_row = &input[t * cols..(t + 1) * cols];
            output[t * rows + r] = dot_f32_compat(&row_buf, x_row);
        }
    }
    Ok(())
}

#[cfg(not(feature = "with-kernels"))]
pub(crate) fn decode_quant_row_portable(quant: QuantType, row_slice: &[u8], out: &mut [f32]) {
    let block_elems = quant.block_elements();
    let block_bytes = quant.block_bytes();
    for (blk_idx, chunk) in out.chunks_exact_mut(block_elems).enumerate() {
        let block = &row_slice[blk_idx * block_bytes..(blk_idx + 1) * block_bytes];
        match quant {
            QuantType::Q4_0 => decode_q4_0_portable(block, chunk),
            QuantType::Q8_0 => decode_q8_0_portable(block, chunk),
            QuantType::Q4_K_M => decode_q4_k_m_portable(block, chunk),
            QuantType::Q5_K => decode_q5_k_portable(block, chunk),
            QuantType::Q6_K => decode_q6_k_portable(block, chunk),
            _ => chunk.fill(0.0),
        }
    }
}

#[cfg(not(feature = "with-kernels"))]
fn decode_q4_0_portable(block: &[u8], out: &mut [f32]) {
    let d = f16_to_f32_portable(u16::from_le_bytes([block[0], block[1]]));
    if d == 0.0 {
        out.fill(0.0);
        return;
    }
    for i in 0..16 {
        let b = block[2 + i];
        #[expect(clippy::cast_possible_wrap, reason = "4-bit nibble offset by -8")]
        let q0 = ((b & 0x0F) as i8) - 8;
        #[expect(clippy::cast_possible_wrap, reason = "4-bit nibble offset by -8")]
        let q1 = (((b >> 4) & 0x0F) as i8) - 8;
        out[i] = d * f32::from(q0);
        out[i + 16] = d * f32::from(q1);
    }
}

#[cfg(not(feature = "with-kernels"))]
fn decode_q8_0_portable(block: &[u8], out: &mut [f32]) {
    let d = f16_to_f32_portable(u16::from_le_bytes([block[0], block[1]]));
    if d == 0.0 {
        out.fill(0.0);
        return;
    }
    #[expect(
        clippy::cast_possible_wrap,
        reason = "reinterpreting byte as signed i8"
    )]
    for i in 0..32 {
        out[i] = d * f32::from(block[2 + i] as i8);
    }
}

#[cfg(not(feature = "with-kernels"))]
fn unpack_kquant_scales(scales: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut sc = [0_u8; 8];
    let mut m = [0_u8; 8];
    for j in 0..4 {
        sc[j] = scales[j] & 0x3F;
        m[j] = scales[j + 4] & 0x3F;
    }
    for j in 4..8 {
        sc[j] = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        m[j] = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
    (sc, m)
}

#[cfg(not(feature = "with-kernels"))]
fn decode_q4_k_m_portable(block: &[u8], out: &mut [f32]) {
    let d = f16_to_f32_portable(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32_portable(u16::from_le_bytes([block[2], block[3]]));
    let (sc, m) = unpack_kquant_scales(&block[4..16]);
    let qs = &block[16..144];
    for chunk in 0..4 {
        let d1 = d * f32::from(sc[chunk * 2]);
        let m1 = dmin * f32::from(m[chunk * 2]);
        let d2 = d * f32::from(sc[chunk * 2 + 1]);
        let m2 = dmin * f32::from(m[chunk * 2 + 1]);
        for l in 0..32 {
            out[chunk * 64 + l] = d1 * f32::from(qs[chunk * 32 + l] & 0x0F) - m1;
            out[chunk * 64 + 32 + l] = d2 * f32::from(qs[chunk * 32 + l] >> 4) - m2;
        }
    }
}

#[cfg(not(feature = "with-kernels"))]
fn decode_q5_k_portable(block: &[u8], out: &mut [f32]) {
    let d = f16_to_f32_portable(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32_portable(u16::from_le_bytes([block[2], block[3]]));
    if d == 0.0 && dmin == 0.0 {
        out.fill(0.0);
        return;
    }
    let (sc, m) = unpack_kquant_scales(&block[4..16]);
    let qh = &block[16..48];
    let qs = &block[48..176];
    let mut u1 = 1_u8;
    let mut u2 = 2_u8;
    for chunk in 0..4 {
        let d1 = d * f32::from(sc[chunk * 2]);
        let m1 = dmin * f32::from(m[chunk * 2]);
        let d2 = d * f32::from(sc[chunk * 2 + 1]);
        let m2 = dmin * f32::from(m[chunk * 2 + 1]);
        for l in 0..32 {
            let q1 = (qs[chunk * 32 + l] & 0x0F) + if (qh[l] & u1) != 0 { 16 } else { 0 };
            let q2 = (qs[chunk * 32 + l] >> 4) + if (qh[l] & u2) != 0 { 16 } else { 0 };
            out[chunk * 64 + l] = d1 * f32::from(q1) - m1;
            out[chunk * 64 + 32 + l] = d2 * f32::from(q2) - m2;
        }
        u1 <<= 2;
        u2 <<= 2;
    }
}

#[cfg(not(feature = "with-kernels"))]
fn decode_q6_k_portable(block: &[u8], out: &mut [f32]) {
    let d = f16_to_f32_portable(u16::from_le_bytes([block[208], block[209]]));
    if d == 0.0 {
        out.fill(0.0);
        return;
    }
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    for (half_idx, n) in [0_usize, 128_usize].into_iter().enumerate() {
        let low_base = half_idx * 64;
        let high_base = half_idx * 32;
        let scale_base = half_idx * 8;
        for l in 0..32 {
            let is = l / 16;
            #[expect(clippy::cast_possible_wrap, reason = "quant offset by -32")]
            let q1 = (((ql[low_base + l] & 0x0F) | ((qh[high_base + l] & 0x03) << 4)) as i8) - 32;
            #[expect(clippy::cast_possible_wrap, reason = "quant offset by -32")]
            let q2 = (((ql[low_base + l + 32] & 0x0F) | (((qh[high_base + l] >> 2) & 0x03) << 4))
                as i8)
                - 32;
            #[expect(clippy::cast_possible_wrap, reason = "quant offset by -32")]
            let q3 = ((((ql[low_base + l] >> 4) & 0x0F) | (((qh[high_base + l] >> 4) & 0x03) << 4))
                as i8)
                - 32;
            #[expect(clippy::cast_possible_wrap, reason = "quant offset by -32")]
            let q4 = ((((ql[low_base + l + 32] >> 4) & 0x0F)
                | (((qh[high_base + l] >> 6) & 0x03) << 4)) as i8)
                - 32;
            #[expect(clippy::cast_possible_wrap, reason = "signed scale byte")]
            let sc1 = f32::from(scales[scale_base + is] as i8);
            #[expect(clippy::cast_possible_wrap, reason = "signed scale byte")]
            let sc2 = f32::from(scales[scale_base + is + 2] as i8);
            #[expect(clippy::cast_possible_wrap, reason = "signed scale byte")]
            let sc3 = f32::from(scales[scale_base + is + 4] as i8);
            #[expect(clippy::cast_possible_wrap, reason = "signed scale byte")]
            let sc4 = f32::from(scales[scale_base + is + 6] as i8);
            out[n + l] = d * sc1 * f32::from(q1);
            out[n + l + 32] = d * sc2 * f32::from(q2);
            out[n + l + 64] = d * sc3 * f32::from(q3);
            out[n + l + 96] = d * sc4 * f32::from(q4);
        }
    }
}

#[cfg(not(feature = "with-kernels"))]
fn precompute_rope_inv_freq_portable(
    head_dim: usize,
    theta: f32,
    scaling: Option<&RopeScalingConfig>,
    seq_len: usize,
) -> Result<Vec<f32>, K3Error> {
    if head_dim == 0 || head_dim % 2 != 0 || !theta.is_finite() || theta <= 0.0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "rope".to_owned(),
            reason: "invalid RoPE parameters".to_owned(),
        });
    }
    let dim_pairs = head_dim / 2;
    let mut inv_freq = Vec::with_capacity(dim_pairs);
    #[expect(
        clippy::cast_precision_loss,
        reason = "head_dim fits comfortably in f32"
    )]
    let hd_f32 = head_dim as f32;
    for i in 0..dim_pairs {
        #[expect(clippy::cast_precision_loss, reason = "index fits cleanly in f32")]
        let base_theta = 1.0_f32 / theta.powf(((2 * i) as f32) / hd_f32);
        let scaled = match scaling {
            None => base_theta,
            Some(RopeScalingConfig::Linear { factor }) => {
                if !factor.is_finite() || *factor <= 0.0 {
                    return Err(K3Error::CapabilityMismatch {
                        architecture: "rope".to_owned(),
                        reason: format!("invalid linear RoPE factor {factor}"),
                    });
                }
                base_theta / factor
            }
            Some(RopeScalingConfig::Llama3Smooth {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position_embeddings,
            }) => {
                if !factor.is_finite()
                    || *factor <= 0.0
                    || !low_freq_factor.is_finite()
                    || *low_freq_factor <= 0.0
                    || !high_freq_factor.is_finite()
                    || *high_freq_factor <= 0.0
                    || *original_max_position_embeddings == 0
                    || low_freq_factor >= high_freq_factor
                {
                    return Err(K3Error::CapabilityMismatch {
                        architecture: "rope".to_owned(),
                        reason: "invalid smooth RoPE scaling parameters".to_owned(),
                    });
                }
                let wavelen = (2.0_f32 * PI) / base_theta;
                #[expect(clippy::cast_precision_loss, reason = "context length fits in f32")]
                let orig_f32 = *original_max_position_embeddings as f32;
                let low_wavelen = orig_f32 / low_freq_factor;
                let high_wavelen = orig_f32 / high_freq_factor;
                if wavelen < high_wavelen {
                    base_theta
                } else if wavelen > low_wavelen {
                    base_theta / factor
                } else {
                    let smooth = (((orig_f32 / wavelen) - low_freq_factor)
                        / (high_freq_factor - low_freq_factor))
                        .clamp(0.0, 1.0);
                    (1.0_f32 - smooth) * (base_theta / factor) + smooth * base_theta
                }
            }
            Some(RopeScalingConfig::LongRope {
                short_factor,
                long_factor,
                original_max_position_embeddings,
            }) => {
                if short_factor.len() != dim_pairs || long_factor.len() != dim_pairs {
                    return Err(K3Error::CapabilityMismatch {
                        architecture: "rope".to_owned(),
                        reason: "LongRope factor lengths must match head_dim / 2".to_owned(),
                    });
                }
                let f = if seq_len > *original_max_position_embeddings {
                    long_factor[i]
                } else {
                    short_factor[i]
                };
                if !f.is_finite() || f <= 0.0 {
                    return Err(K3Error::CapabilityMismatch {
                        architecture: "rope".to_owned(),
                        reason: format!("invalid LongRope factor {f}"),
                    });
                }
                base_theta / f
            }
        };
        inv_freq.push(scaled);
    }
    Ok(inv_freq)
}

#[cfg(not(feature = "with-kernels"))]
fn softmax_scores_portable(scores: &mut [f32]) {
    let mut max_score = f32::NEG_INFINITY;
    for &s in scores.iter() {
        if s > max_score {
            max_score = s;
        }
    }
    let mut sum_exp = 0.0_f32;
    for s in scores.iter_mut() {
        let w = (*s - max_score).exp();
        *s = w;
        sum_exp += w;
    }
    let inv_sum = if sum_exp > 0.0 && sum_exp.is_finite() {
        1.0_f32 / sum_exp
    } else {
        0.0
    };
    for s in scores.iter_mut() {
        *s *= inv_sum;
    }
}

#[cfg(not(feature = "with-kernels"))]
#[expect(
    clippy::too_many_arguments,
    reason = "matches gqa_causal_attention_f32 signature"
)]
fn gqa_attention_portable(
    out: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) -> Result<(), K3Error> {
    if n_heads_q == 0 || n_heads_kv == 0 || head_dim == 0 || seq_len == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "gqa".to_owned(),
            reason: "GQA dimensions must be non-zero".to_owned(),
        });
    }
    if n_heads_q % n_heads_kv != 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "gqa".to_owned(),
            reason: format!("n_heads_q ({n_heads_q}) not divisible by n_heads_kv ({n_heads_kv})"),
        });
    }
    let q_expected = n_heads_q * head_dim;
    let kv_expected = seq_len * n_heads_kv * head_dim;
    if q.len() != q_expected
        || out.len() != q_expected
        || k_cache.len() != kv_expected
        || v_cache.len() != kv_expected
    {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "gqa".to_owned(),
            offset: 0,
            byte_len: size_of_val(k_cache),
            buffer_len: kv_expected * size_of::<f32>(),
            reason: "GQA buffer length mismatch".to_owned(),
        });
    }

    let kv_group_size = n_heads_q / n_heads_kv;
    let kv_stride = n_heads_kv * head_dim;
    let mut scores = vec![0.0_f32; seq_len];

    for h in 0..n_heads_q {
        let kv_h = h / kv_group_size;
        let q_head = &q[h * head_dim..(h + 1) * head_dim];
        for (t, score_slot) in scores.iter_mut().enumerate() {
            let k_off = t * kv_stride + kv_h * head_dim;
            *score_slot = dot_f32_compat(q_head, &k_cache[k_off..k_off + head_dim]) * scale;
        }
        softmax_scores_portable(&mut scores);
        let out_head = &mut out[h * head_dim..(h + 1) * head_dim];
        out_head.fill(0.0);
        for (t, &prob) in scores.iter().enumerate() {
            if prob == 0.0_f32 {
                continue;
            }
            let v_off = t * kv_stride + kv_h * head_dim;
            for (dst, &v_val) in out_head.iter_mut().zip(&v_cache[v_off..v_off + head_dim]) {
                *dst += prob * v_val;
            }
        }
    }
    Ok(())
}

#[cfg(not(feature = "with-kernels"))]
#[expect(
    clippy::too_many_arguments,
    reason = "matches gqa_causal_attention_prefill_f32 signature"
)]
fn gqa_attention_prefill_portable(
    out: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    m: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    total_seq_len: usize,
    start_pos: usize,
    sliding_window: Option<usize>,
    scale: f32,
    scores_scratch: &mut [f32],
) -> Result<(), K3Error> {
    if m == 0 || n_heads_q == 0 || n_heads_kv == 0 || head_dim == 0 || total_seq_len == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "gqa".to_owned(),
            reason: "GQA prefill dimensions must be non-zero".to_owned(),
        });
    }
    if n_heads_q % n_heads_kv != 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "gqa".to_owned(),
            reason: format!("n_heads_q ({n_heads_q}) not divisible by n_heads_kv ({n_heads_kv})"),
        });
    }
    if sliding_window == Some(0) || !scale.is_finite() {
        return Err(K3Error::CapabilityMismatch {
            architecture: "gqa".to_owned(),
            reason: "invalid GQA sliding window or non-finite scale".to_owned(),
        });
    }
    let query_end = start_pos
        .checked_add(m)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "gqa_start_pos".to_owned(),
            offset: start_pos,
            byte_len: m,
            buffer_len: total_seq_len,
            reason: "start_pos + m overflow".to_owned(),
        })?;
    if query_end > total_seq_len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "gqa_start_pos".to_owned(),
            offset: start_pos,
            byte_len: m,
            buffer_len: total_seq_len,
            reason: format!("start_pos + m ({query_end}) > total_seq_len ({total_seq_len})"),
        });
    }
    let q_expected = m * n_heads_q * head_dim;
    let kv_stride = n_heads_kv * head_dim;
    let kv_expected = total_seq_len * kv_stride;
    let min_scratch = sliding_window.map_or(query_end, |w| w.min(query_end));
    if q.len() != q_expected
        || out.len() != q_expected
        || k_cache.len() < kv_expected
        || v_cache.len() < kv_expected
        || scores_scratch.len() < min_scratch
    {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "gqa_prefill".to_owned(),
            offset: 0,
            byte_len: size_of_val(k_cache),
            buffer_len: kv_expected * size_of::<f32>(),
            reason: "GQA prefill buffer or scratch length mismatch".to_owned(),
        });
    }

    let group_size = n_heads_q / n_heads_kv;
    for kv_h in 0..n_heads_kv {
        let kv_head_offset = kv_h * head_dim;
        for m_idx in 0..m {
            let query_pos = start_pos + m_idx;
            let valid_end = (query_pos + 1).min(total_seq_len);
            let valid_start =
                sliding_window.map_or(0, |w| query_pos.saturating_add(1).saturating_sub(w));
            let active_len = valid_end.saturating_sub(valid_start);
            let score_buf = &mut scores_scratch[..active_len];

            for g in 0..group_size {
                let q_h = kv_h * group_size + g;
                let q_slice = &q[(m_idx * n_heads_q + q_h) * head_dim..][..head_dim];
                for (slot, s) in score_buf.iter_mut().zip(valid_start..valid_end) {
                    let kv_off = s * kv_stride + kv_head_offset;
                    *slot = dot_f32_compat(q_slice, &k_cache[kv_off..kv_off + head_dim]) * scale;
                }
                if active_len > 0 {
                    softmax_scores_portable(score_buf);
                }
                let out_slice = &mut out[(m_idx * n_heads_q + q_h) * head_dim..][..head_dim];
                out_slice.fill(0.0_f32);
                for (s, &weight) in (valid_start..valid_end).zip(score_buf.iter()) {
                    if weight == 0.0_f32 {
                        continue;
                    }
                    let kv_off = s * kv_stride + kv_head_offset;
                    for (dst, &v_val) in out_slice
                        .iter_mut()
                        .zip(&v_cache[kv_off..kv_off + head_dim])
                    {
                        *dst += weight * v_val;
                    }
                }
            }
        }
    }
    Ok(())
}
