//! Exhaustive 256x256 multi-quant (`F32`, `F16`, `BF16`, `Q4_0`, `Q4_K_M`, `Q5_K`, `Q6_K`, `Q8_0`)
//! and Mixtral packed-3D vs split-2D Sparse `MoE` integration parity tests against an
//! independent scalar reference forward pass.

use std::collections::HashMap;

use jero_arch_mistral::{MistralLayerCache, MistralRecipe, RopeScalingConfig, route_top2_experts};
use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;
use jero_core::{
    ArchRecipe, ExecutionContext, GgufFile, GgufValue, K3Error, ModelWeights, QuantType, TensorView,
};

fn assert_logits_parity(
    label: &str,
    quant: QuantType,
    got: &[f32],
    expected: &[f32],
    strict_float_parity: bool,
) {
    assert_eq!(got.len(), expected.len(), "{label} length mismatch");
    let use_q8_bounds = cfg!(feature = "with-kernels")
        && !strict_float_parity
        && !matches!(quant, QuantType::F32 | QuantType::F16 | QuantType::BF16);

    let mut dot = 0.0_f64;
    let mut norm_g = 0.0_f64;
    let mut norm_e = 0.0_f64;
    let mut diff_sq = 0.0_f64;
    for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
        assert!(
            g.is_finite() && e.is_finite(),
            "{label} non-finite at [{i}]"
        );
        if !use_q8_bounds {
            assert!(
                (g - e).abs() < 1e-3,
                "{label} ({quant:?}) logit[{i}] mismatch: got {g}, expected {e}"
            );
        }
        let gf = f64::from(g);
        let ef = f64::from(e);
        dot += gf * ef;
        norm_g += gf * gf;
        norm_e += ef * ef;
        diff_sq += (gf - ef) * (gf - ef);
    }

    if use_q8_bounds {
        let cosine = if norm_g > 1e-9 && norm_e > 1e-9 {
            dot / (norm_g.sqrt() * norm_e.sqrt())
        } else {
            1.0
        };
        let l2_rel = diff_sq.sqrt() / norm_e.sqrt().max(1.0);
        assert!(
            cosine > 0.999 && l2_rel < 2e-2,
            "{label} ({quant:?}) Q8-repacked parity failure: cosine={cosine}, l2_rel={l2_rel}, got={got:?}, expected={expected:?}"
        );
    }
}

struct StoredTensor {
    name: String,
    quant: QuantType,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

#[derive(Default)]
struct IntegrationWeights {
    tensors: HashMap<String, StoredTensor>,
}

impl IntegrationWeights {
    fn insert_raw(&mut self, name: &str, quant: QuantType, shape: Vec<usize>, bytes: Vec<u8>) {
        self.tensors.insert(
            name.to_owned(),
            StoredTensor {
                name: name.to_owned(),
                quant,
                shape,
                bytes,
            },
        );
    }

    fn insert_f16(&mut self, name: &str, shape: Vec<usize>, vals: &[f32]) {
        let mut bytes = Vec::with_capacity(vals.len() * 2);
        for &v in vals {
            bytes.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
        }
        self.insert_raw(name, QuantType::F16, shape, bytes);
    }

    fn insert_bf16(&mut self, name: &str, shape: Vec<usize>, vals: &[f32]) {
        let mut bytes = Vec::with_capacity(vals.len() * 2);
        for &v in vals {
            bytes.extend_from_slice(&f32_to_bf16_bits(v).to_le_bytes());
        }
        self.insert_raw(name, QuantType::BF16, shape, bytes);
    }

    fn insert_f32(&mut self, name: &str, shape: Vec<usize>, vals: &[f32]) {
        let mut bytes = Vec::with_capacity(vals.len() * 4);
        for &v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.insert_raw(name, QuantType::F32, shape, bytes);
    }
}

impl ModelWeights for IntegrationWeights {
    fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
        let entry = self
            .tensors
            .get(name)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: format!("missing tensor '{name}' in IntegrationWeights"),
            })?;
        TensorView::from_bytes(&entry.name, entry.quant, &entry.shape, 0, &entry.bytes)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
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
    let mant = (bits >> 13) & 0x03FF;
    if exp <= 0 {
        sign
    } else if exp >= 31 {
        sign | 0x7C00
    } else {
        #[expect(clippy::cast_sign_loss, reason = "exp bounded to 1..31")]
        let exp_u16 = u16::try_from(exp as u32).unwrap_or(0);
        let mant_u16 = u16::try_from(mant).unwrap_or(0);
        sign | (exp_u16 << 10) | mant_u16
    }
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from((bits >> 10) & 0x1F);
    let mant = u32::from(bits & 0x03FF);
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let mut m = mant;
        let mut e = 127_u32 - 14;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03FF;
        f32::from_bits(sign | (e << 23) | (m << 13))
    } else if exp == 31 {
        f32::from_bits(sign | (0xFF << 23) | (mant << 13))
    } else {
        f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
    }
}

fn f32_to_bf16_bits(val: f32) -> u16 {
    u16::try_from(val.to_bits() >> 16).unwrap_or(0)
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

#[expect(clippy::cast_precision_loss)]
fn deterministic_scalar(idx: usize, seed: u32) -> f32 {
    let raw = i32::try_from((idx * 17 + seed as usize * 31) % 29).unwrap_or(0) - 14;
    (raw as f32) * 0.025
}

fn encode_quant_matrix(quant: QuantType, rows: usize, cols: usize, seed: u32) -> Vec<u8> {
    match quant {
        QuantType::F32 | QuantType::F16 | QuantType::BF16 => {
            let mut out = Vec::new();
            for r in 0..rows {
                for c in 0..cols {
                    let val = deterministic_scalar(r * cols + c, seed);
                    match quant {
                        QuantType::F32 => out.extend_from_slice(&val.to_le_bytes()),
                        QuantType::F16 => {
                            out.extend_from_slice(&f32_to_f16_bits(val).to_le_bytes());
                        }
                        QuantType::BF16 => {
                            out.extend_from_slice(&f32_to_bf16_bits(val).to_le_bytes());
                        }
                        _ => unreachable!(),
                    }
                }
            }
            out
        }
        QuantType::Q4_0
        | QuantType::Q8_0
        | QuantType::Q4_K_M
        | QuantType::Q5_K
        | QuantType::Q6_K => {
            let block_elems = quant.block_elements();
            let block_bytes = quant.block_bytes();
            let total_blocks = rows * (cols / block_elems);
            let mut out = Vec::with_capacity(total_blocks * block_bytes);
            for block_idx in 0..total_blocks {
                out.extend_from_slice(&encode_quant_block(quant, block_bytes, block_idx, seed));
            }
            out
        }
        other => panic!("unsupported test encoder quant {other:?}"),
    }
}

#[expect(clippy::cast_precision_loss)]
fn encode_quant_block(
    quant: QuantType,
    block_bytes: usize,
    block_idx: usize,
    seed: u32,
) -> Vec<u8> {
    let d_bits = f32_to_f16_bits(0.02 + ((block_idx % 5) as f32) * 0.005);
    let dmin_bits = f32_to_f16_bits(0.004);
    let mut blk = vec![0_u8; block_bytes];
    match quant {
        QuantType::Q4_0 => {
            blk[0..2].copy_from_slice(&d_bits.to_le_bytes());
            for (idx, slot) in blk[2..18].iter_mut().enumerate() {
                let low_nib = u8::try_from((block_idx * 3 + idx + seed as usize) % 16).unwrap_or(0);
                let high_nib = u8::try_from((block_idx * 5 + idx + 3) % 16).unwrap_or(0);
                *slot = low_nib | (high_nib << 4);
            }
        }
        QuantType::Q8_0 => {
            blk[0..2].copy_from_slice(&d_bits.to_le_bytes());
            for (idx, slot) in blk[2..34].iter_mut().enumerate() {
                let mag = i32::try_from((idx + block_idx + seed as usize) % 7).unwrap_or(0) - 3;
                *slot = i8::try_from(mag).unwrap_or(0).to_ne_bytes()[0];
            }
        }
        QuantType::Q4_K_M => {
            blk[0..2].copy_from_slice(&d_bits.to_le_bytes());
            blk[2..4].copy_from_slice(&dmin_bits.to_le_bytes());
            for (j, slot) in blk[4..16].iter_mut().enumerate() {
                *slot = u8::try_from((j + block_idx + seed as usize) % 31 + 1).unwrap_or(1);
            }
            for (j, slot) in blk[16..144].iter_mut().enumerate() {
                *slot =
                    u8::try_from(((j * 7 + block_idx + seed as usize) % 251) & 0x77).unwrap_or(0);
            }
        }
        QuantType::Q5_K => {
            blk[0..2].copy_from_slice(&d_bits.to_le_bytes());
            blk[2..4].copy_from_slice(&dmin_bits.to_le_bytes());
            for (j, slot) in blk[4..16].iter_mut().enumerate() {
                *slot = u8::try_from((j + block_idx + seed as usize) % 31 + 1).unwrap_or(1);
            }
            for (j, slot) in blk[16..48].iter_mut().enumerate() {
                *slot = u8::try_from((j + block_idx) % 255).unwrap_or(0);
            }
            for (j, slot) in blk[48..176].iter_mut().enumerate() {
                *slot = u8::try_from((j * 5 + block_idx + seed as usize) % 251).unwrap_or(0);
            }
        }
        QuantType::Q6_K => {
            blk[208..210].copy_from_slice(&d_bits.to_le_bytes());
            for (j, slot) in blk[0..128].iter_mut().enumerate() {
                *slot = u8::try_from((j * 3 + block_idx + seed as usize) % 251).unwrap_or(0);
            }
            for (j, slot) in blk[128..192].iter_mut().enumerate() {
                *slot = u8::try_from((j + block_idx) % 255).unwrap_or(0);
            }
            blk[192..208].fill(2_u8);
        }
        _ => unreachable!(),
    }
    blk
}

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

fn decode_scalar_block(quant: QuantType, block: &[u8], out: &mut [f32]) {
    match quant {
        QuantType::Q4_0 => {
            let d = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
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
        QuantType::Q8_0 => {
            let d = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
            #[expect(clippy::cast_possible_wrap, reason = "signed q8 weight byte")]
            for i in 0..32 {
                out[i] = d * f32::from(block[2 + i] as i8);
            }
        }
        QuantType::Q4_K_M => {
            let d = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let dmin = f16_bits_to_f32(u16::from_le_bytes([block[2], block[3]]));
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
        QuantType::Q5_K => {
            let d = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let dmin = f16_bits_to_f32(u16::from_le_bytes([block[2], block[3]]));
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
        QuantType::Q6_K => decode_q6_k_block(block, out),
        other => panic!("unexpected quant {other:?} in decode_scalar_block"),
    }
}

fn decode_q6_k_block(block: &[u8], out: &mut [f32]) {
    let d = f16_bits_to_f32(u16::from_le_bytes([block[208], block[209]]));
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    for (half_idx, n) in [0_usize, 128_usize].into_iter().enumerate() {
        let low_base = half_idx * 64;
        let high_base = half_idx * 32;
        let scale_base = half_idx * 8;
        for l in 0..32 {
            let is = l / 16;
            #[expect(clippy::cast_possible_wrap, reason = "q6 value offset by -32")]
            let q1 = (((ql[low_base + l] & 0x0F) | ((qh[high_base + l] & 0x03) << 4)) as i8) - 32;
            #[expect(clippy::cast_possible_wrap, reason = "q6 value offset by -32")]
            let q2 = (((ql[low_base + l + 32] & 0x0F) | (((qh[high_base + l] >> 2) & 0x03) << 4))
                as i8)
                - 32;
            #[expect(clippy::cast_possible_wrap, reason = "q6 value offset by -32")]
            let q3 = ((((ql[low_base + l] >> 4) & 0x0F) | (((qh[high_base + l] >> 4) & 0x03) << 4))
                as i8)
                - 32;
            #[expect(clippy::cast_possible_wrap, reason = "q6 value offset by -32")]
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

fn decode_matrix_rows_f32(tensor: &TensorView<'_>, rows: usize, cols: usize) -> Vec<Vec<f32>> {
    let quant = tensor.quant();
    let row_bytes = quant
        .storage_bytes(tensor.name(), cols)
        .expect("valid row size");
    let raw = tensor.as_bytes();
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let row_slice = &raw[r * row_bytes..(r + 1) * row_bytes];
        let mut row = vec![0.0_f32; cols];
        match quant {
            QuantType::F32 => {
                for (c, slot) in row.iter_mut().enumerate() {
                    let b = &row_slice[c * 4..(c + 1) * 4];
                    *slot = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                }
            }
            QuantType::F16 => {
                for (c, slot) in row.iter_mut().enumerate() {
                    let b = &row_slice[c * 2..(c + 1) * 2];
                    *slot = f16_bits_to_f32(u16::from_le_bytes([b[0], b[1]]));
                }
            }
            QuantType::BF16 => {
                for (c, slot) in row.iter_mut().enumerate() {
                    let b = &row_slice[c * 2..(c + 1) * 2];
                    *slot = bf16_bits_to_f32(u16::from_le_bytes([b[0], b[1]]));
                }
            }
            _ => {
                let block_elems = quant.block_elements();
                let block_bytes = quant.block_bytes();
                for (blk_idx, blk) in row_slice.chunks_exact(block_bytes).enumerate() {
                    decode_scalar_block(
                        quant,
                        blk,
                        &mut row[blk_idx * block_elems..(blk_idx + 1) * block_elems],
                    );
                }
            }
        }
        out.push(row);
    }
    out
}

#[expect(clippy::cast_precision_loss)]
fn build_dense_weights(recipe: &MistralRecipe, q: QuantType, tied: bool) -> IntegrationWeights {
    let mut w = IntegrationWeights::default();
    let h = recipe.hidden_dim;
    let total_q = recipe.n_heads_q * recipe.head_dim;
    let total_kv = recipe.n_heads_kv * recipe.head_dim;

    w.insert_raw(
        "token_embd.weight",
        q,
        vec![h, recipe.vocab_size],
        encode_quant_matrix(q, recipe.vocab_size, h, 1),
    );
    for l in 0..recipe.num_layers {
        let seed_base = u32::try_from(l + 1).unwrap() * 10;
        w.insert_f16(
            &format!("blk.{l}.attn_norm.weight"),
            vec![h],
            &(0..h).map(|i| 0.9 + (i as f32) * 0.01).collect::<Vec<_>>(),
        );
        w.insert_raw(
            &format!("blk.{l}.attn_q.weight"),
            q,
            vec![h, total_q],
            encode_quant_matrix(q, total_q, h, seed_base + 1),
        );
        w.insert_raw(
            &format!("blk.{l}.attn_k.weight"),
            q,
            vec![h, total_kv],
            encode_quant_matrix(q, total_kv, h, seed_base + 2),
        );
        w.insert_raw(
            &format!("blk.{l}.attn_v.weight"),
            q,
            vec![h, total_kv],
            encode_quant_matrix(q, total_kv, h, seed_base + 3),
        );
        w.insert_raw(
            &format!("blk.{l}.attn_output.weight"),
            q,
            vec![total_q, h],
            encode_quant_matrix(q, h, total_q, seed_base + 4),
        );
        w.insert_bf16(
            &format!("blk.{l}.ffn_norm.weight"),
            vec![h],
            &(0..h)
                .map(|i| 1.05 - (i as f32) * 0.005)
                .collect::<Vec<_>>(),
        );
        w.insert_raw(
            &format!("blk.{l}.ffn_gate.weight"),
            q,
            vec![h, recipe.ffn_dim],
            encode_quant_matrix(q, recipe.ffn_dim, h, seed_base + 5),
        );
        w.insert_raw(
            &format!("blk.{l}.ffn_up.weight"),
            q,
            vec![h, recipe.ffn_dim],
            encode_quant_matrix(q, recipe.ffn_dim, h, seed_base + 6),
        );
        w.insert_raw(
            &format!("blk.{l}.ffn_down.weight"),
            q,
            vec![recipe.ffn_dim, h],
            encode_quant_matrix(q, h, recipe.ffn_dim, seed_base + 7),
        );
    }
    w.insert_f32(
        "output_norm.weight",
        vec![h],
        &(0..h).map(|i| 0.95 + (i as f32) * 0.01).collect::<Vec<_>>(),
    );
    if !tied {
        w.insert_raw(
            "output.weight",
            q,
            vec![h, recipe.vocab_size],
            encode_quant_matrix(q, recipe.vocab_size, h, 99),
        );
    }
    w
}

#[expect(clippy::cast_precision_loss)]
fn reference_rmsnorm(hidden: &[f32], norm_tv: &TensorView<'_>, rms_eps: f32) -> Vec<f32> {
    let nw = decode_matrix_rows_f32(norm_tv, 1, hidden.len()).remove(0);
    let sum_sq: f32 = hidden.iter().map(|&x| x * x).sum();
    let rms = 1.0_f32 / ((sum_sq / hidden.len() as f32) + rms_eps).sqrt();
    hidden
        .iter()
        .zip(nw.iter())
        .map(|(&h, &w)| h * rms * w)
        .collect()
}

fn reference_swiglu_ffn(
    g_tv: &TensorView<'_>,
    u_tv: &TensorView<'_>,
    d_tv: &TensorView<'_>,
    ffn_dim: usize,
    hidden_dim: usize,
    input: &[f32],
) -> Vec<f32> {
    let g_rows = decode_matrix_rows_f32(g_tv, ffn_dim, hidden_dim);
    let u_rows = decode_matrix_rows_f32(u_tv, ffn_dim, hidden_dim);
    let d_rows = decode_matrix_rows_f32(d_tv, hidden_dim, ffn_dim);
    let mut act = Vec::with_capacity(ffn_dim);
    for (gr, ur) in g_rows.iter().zip(&u_rows) {
        let g: f32 = gr.iter().zip(input).map(|(&a, &b)| a * b).sum();
        let u: f32 = ur.iter().zip(input).map(|(&a, &b)| a * b).sum();
        let sig = 1.0_f32 / (1.0_f32 + (-g).exp());
        act.push((g * sig) * u);
    }
    d_rows
        .iter()
        .map(|dr| dr.iter().zip(&act).map(|(&a, &b)| a * b).sum())
        .collect()
}

fn expert_tv_for_test<'a>(
    weights: &'a IntegrationWeights,
    layer: usize,
    proj: &str,
    expert_idx: usize,
    n_exp: usize,
) -> TensorView<'a> {
    let stacked = format!("blk.{layer}.ffn_{proj}_exps.weight");
    if weights.contains_tensor(&stacked) {
        return weights
            .tensor(&stacked)
            .unwrap()
            .slice_expert(expert_idx, n_exp)
            .unwrap();
    }
    weights
        .tensor(&format!("blk.{layer}.ffn_{proj}.{expert_idx}.weight"))
        .unwrap()
}

struct RefKvState<'a> {
    inv_freq: &'a [f32],
    k_hist: Vec<Vec<Vec<f32>>>,
    v_hist: Vec<Vec<Vec<f32>>>,
}

#[expect(clippy::cast_precision_loss)]
fn reference_attention_step(
    recipe: &MistralRecipe,
    weights: &IntegrationWeights,
    layer: usize,
    pos: usize,
    kv_state: &mut RefKvState<'_>,
    hidden: &mut [f32],
) {
    let hidden_dim = recipe.hidden_dim;
    let total_q = recipe.n_heads_q * recipe.head_dim;
    let total_kv = recipe.n_heads_kv * recipe.head_dim;
    let kv_group = recipe.n_heads_q / recipe.n_heads_kv;
    let scale = 1.0_f32 / (recipe.head_dim as f32).sqrt();

    let attn_norm_tv = weights
        .tensor(&format!("blk.{layer}.attn_norm.weight"))
        .unwrap();
    let norm_h = reference_rmsnorm(hidden, &attn_norm_tv, recipe.rms_eps);

    let q_rows = decode_matrix_rows_f32(
        &weights
            .tensor(&format!("blk.{layer}.attn_q.weight"))
            .unwrap(),
        total_q,
        hidden_dim,
    );
    let k_rows = decode_matrix_rows_f32(
        &weights
            .tensor(&format!("blk.{layer}.attn_k.weight"))
            .unwrap(),
        total_kv,
        hidden_dim,
    );
    let v_rows = decode_matrix_rows_f32(
        &weights
            .tensor(&format!("blk.{layer}.attn_v.weight"))
            .unwrap(),
        total_kv,
        hidden_dim,
    );
    let o_rows = decode_matrix_rows_f32(
        &weights
            .tensor(&format!("blk.{layer}.attn_output.weight"))
            .unwrap(),
        hidden_dim,
        total_q,
    );

    let mut q_proj: Vec<f32> = q_rows
        .iter()
        .map(|r| r.iter().zip(&norm_h).map(|(&a, &b)| a * b).sum())
        .collect();
    let mut k_proj: Vec<f32> = k_rows
        .iter()
        .map(|r| r.iter().zip(&norm_h).map(|(&a, &b)| a * b).sum())
        .collect();
    let v_proj: Vec<f32> = v_rows
        .iter()
        .map(|r| r.iter().zip(&norm_h).map(|(&a, &b)| a * b).sum())
        .collect();

    for vec_qk in [&mut q_proj, &mut k_proj] {
        for head in vec_qk.chunks_exact_mut(recipe.head_dim) {
            for (pair_idx, &freq) in kv_state.inv_freq.iter().enumerate() {
                let (sin_a, cos_a) = (pos as f32 * freq).sin_cos();
                let even = head[2 * pair_idx];
                let odd = head[2 * pair_idx + 1];
                head[2 * pair_idx] = even * cos_a - odd * sin_a;
                head[2 * pair_idx + 1] = even * sin_a + odd * cos_a;
            }
        }
    }

    kv_state.k_hist[layer].push(k_proj);
    kv_state.v_hist[layer].push(v_proj);

    let end_pos = pos + 1;
    let start_pos = match recipe.sliding_window {
        Some(win) => end_pos.saturating_sub(win),
        None => 0,
    };
    let mut attn_out = vec![0.0_f32; total_q];
    for h_idx in 0..recipe.n_heads_q {
        let kvh = h_idx / kv_group;
        let qh = &q_proj[h_idx * recipe.head_dim..(h_idx + 1) * recipe.head_dim];
        let mut scores = Vec::with_capacity(end_pos - start_pos);
        let mut max_s = f32::NEG_INFINITY;
        for k_step in kv_state.k_hist[layer].iter().take(end_pos).skip(start_pos) {
            let kh = &k_step[kvh * recipe.head_dim..(kvh + 1) * recipe.head_dim];
            let score = qh.iter().zip(kh).map(|(&a, &b)| a * b).sum::<f32>() * scale;
            max_s = max_s.max(score);
            scores.push(score);
        }
        let exps: Vec<f32> = scores.iter().map(|&sc| (sc - max_s).exp()).collect();
        let sum_e: f32 = exps.iter().sum();
        let out_h = &mut attn_out[h_idx * recipe.head_dim..(h_idx + 1) * recipe.head_dim];
        for (offset, e_val) in exps.into_iter().enumerate() {
            let prob = e_val / sum_e;
            let vh = &kv_state.v_hist[layer][start_pos + offset]
                [kvh * recipe.head_dim..(kvh + 1) * recipe.head_dim];
            for (dst, &val) in out_h.iter_mut().zip(vh) {
                *dst += prob * val;
            }
        }
    }

    for (r, h_slot) in hidden.iter_mut().enumerate() {
        *h_slot += o_rows[r]
            .iter()
            .zip(&attn_out)
            .map(|(&a, &b)| a * b)
            .sum::<f32>();
    }
}

#[expect(clippy::cast_precision_loss)]
fn reference_forward_sequence(
    recipe: &MistralRecipe,
    tokens: &[u32],
    weights: &IntegrationWeights,
) -> Vec<f32> {
    let hidden_dim = recipe.hidden_dim;
    let inv_freq: Vec<f32> = (0..(recipe.head_dim / 2))
        .map(|i| {
            1.0_f32
                / recipe
                    .rope_theta
                    .powf((2 * i) as f32 / recipe.head_dim as f32)
        })
        .collect();

    let mut kv_state = RefKvState {
        inv_freq: &inv_freq,
        k_hist: vec![Vec::new(); recipe.num_layers],
        v_hist: vec![Vec::new(); recipe.num_layers],
    };
    let mut last_hidden = vec![0.0_f32; hidden_dim];

    for (pos, &tok) in tokens.iter().enumerate() {
        let emb_tv = weights.tensor("token_embd.weight").expect("embd");
        let mut hidden = decode_matrix_rows_f32(&emb_tv, recipe.vocab_size, hidden_dim)
            [usize::try_from(tok).unwrap()]
        .clone();

        for layer in 0..recipe.num_layers {
            reference_attention_step(recipe, weights, layer, pos, &mut kv_state, &mut hidden);
            let ffn_norm_tv = weights
                .tensor(&format!("blk.{layer}.ffn_norm.weight"))
                .unwrap();
            let ffn_in = reference_rmsnorm(&hidden, &ffn_norm_tv, recipe.rms_eps);

            if let Some(n_exp) = recipe.num_experts {
                let router_tv = weights
                    .tensor(&format!("blk.{layer}.ffn_gate_inp.weight"))
                    .unwrap();
                let r_rows = decode_matrix_rows_f32(&router_tv, n_exp, hidden_dim);
                let r_logits: Vec<f32> = r_rows
                    .iter()
                    .map(|r| r.iter().zip(&ffn_in).map(|(&a, &b)| a * b).sum())
                    .collect();
                for (exp_idx, gate_weight) in route_top2_experts(&r_logits).unwrap() {
                    let g_tv = expert_tv_for_test(weights, layer, "gate", exp_idx, n_exp);
                    let u_tv = expert_tv_for_test(weights, layer, "up", exp_idx, n_exp);
                    let d_tv = expert_tv_for_test(weights, layer, "down", exp_idx, n_exp);
                    let d_vec = reference_swiglu_ffn(
                        &g_tv,
                        &u_tv,
                        &d_tv,
                        recipe.ffn_dim,
                        hidden_dim,
                        &ffn_in,
                    );
                    for (dst, &val) in hidden.iter_mut().zip(&d_vec) {
                        *dst += gate_weight * val;
                    }
                }
            } else {
                let g_tv = weights
                    .tensor(&format!("blk.{layer}.ffn_gate.weight"))
                    .unwrap();
                let u_tv = weights
                    .tensor(&format!("blk.{layer}.ffn_up.weight"))
                    .unwrap();
                let d_tv = weights
                    .tensor(&format!("blk.{layer}.ffn_down.weight"))
                    .unwrap();
                let d_vec =
                    reference_swiglu_ffn(&g_tv, &u_tv, &d_tv, recipe.ffn_dim, hidden_dim, &ffn_in);
                for (dst, &val) in hidden.iter_mut().zip(&d_vec) {
                    *dst += val;
                }
            }
        }

        let out_norm_tv = weights.tensor("output_norm.weight").unwrap();
        last_hidden = reference_rmsnorm(&hidden, &out_norm_tv, recipe.rms_eps);
    }

    let head_tv = if weights.contains_tensor("output.weight") {
        weights.tensor("output.weight").unwrap()
    } else {
        weights.tensor("token_embd.weight").unwrap()
    };
    let head_rows = decode_matrix_rows_f32(&head_tv, recipe.vocab_size, hidden_dim);
    head_rows
        .iter()
        .map(|r| r.iter().zip(&last_hidden).map(|(&a, &b)| a * b).sum())
        .collect()
}

#[test]
fn test_all_admitted_quants_and_square_matrix_disambiguation_parity() {
    let mut recipe = MistralRecipe::new().with_sliding_window(None);
    recipe.num_layers = 1;
    recipe.hidden_dim = 256;
    recipe.ffn_dim = 256;
    recipe.n_heads_q = 4;
    recipe.n_heads_kv = 2;
    recipe.head_dim = 64;
    recipe.vocab_size = 4;

    let quants = [
        QuantType::F32,
        QuantType::F16,
        QuantType::BF16,
        QuantType::Q4_0,
        QuantType::Q4_K_M,
        QuantType::Q5_K,
        QuantType::Q6_K,
        QuantType::Q8_0,
    ];

    for quant in quants {
        assert!(recipe.supports_quant(quant));
        let tied = quant == QuantType::Q5_K || quant == QuantType::Q4_0;
        let weights = build_dense_weights(&recipe, quant, tied);

        let sq = weights.tensor("blk.0.attn_q.weight").expect("attn_q");
        assert_eq!(sq.shape(), &[256, 256]);

        let mut kv_cache = vec![0.0_f32; 2 * 6 * (2 * 64)];
        let mut logits = vec![0.0_f32; recipe.vocab_size];
        let mut ctx = ExecutionContext::new(&[1_u32, 2], 6, &mut logits, &mut kv_cache).unwrap();
        recipe.prefill_step(&mut ctx, &weights).expect("prefill");
        ctx.prepare_decode(3).unwrap();
        recipe.decode_step(&mut ctx, &weights).expect("decode");

        let ref_logits = reference_forward_sequence(&recipe, &[1_u32, 2, 3], &weights);
        assert_logits_parity(
            "256x256 default path",
            quant,
            ctx.logits(),
            &ref_logits,
            false,
        );

        let mut fb_cache = MistralLayerCache::new().with_max_repacked_bytes(0);
        let mut kv_fb = vec![0.0_f32; 2 * 6 * (2 * 64)];
        let mut logits_fb = vec![0.0_f32; recipe.vocab_size];
        let mut ctx_fb = ExecutionContext::new(&[1_u32, 2], 6, &mut logits_fb, &mut kv_fb).unwrap();
        recipe
            .prefill_step_with_cache(&mut ctx_fb, &weights, &mut fb_cache)
            .expect("fallback prefill");
        ctx_fb.prepare_decode(3).unwrap();
        recipe
            .decode_step_with_cache(&mut ctx_fb, &weights, &mut fb_cache)
            .expect("fallback decode");
        assert_logits_parity(
            "256x256 DirectFallback path",
            quant,
            ctx_fb.logits(),
            &ref_logits,
            true,
        );
    }
}

#[test]
fn test_mixtral_router_matvec_packed_3d_vs_split_2d_experts_parity() {
    for quant in [
        QuantType::F16,
        QuantType::Q4_0,
        QuantType::Q8_0,
        QuantType::Q5_K,
    ] {
        let mut recipe = MistralRecipe::new_mixtral()
            .with_sliding_window(Some(4))
            .with_experts(4, 2);
        recipe.num_layers = 1;
        recipe.hidden_dim = 256;
        recipe.ffn_dim = 256;
        recipe.n_heads_q = 4;
        recipe.n_heads_kv = 2;
        recipe.head_dim = 64;
        recipe.vocab_size = 4;

        let mut w_packed = IntegrationWeights::default();
        let mut w_split = IntegrationWeights::default();

        let shared = build_dense_weights(&recipe, quant, true);
        for (name, st) in shared.tensors {
            if !name.contains("ffn_gate.")
                && !name.contains("ffn_up.")
                && !name.contains("ffn_down.")
            {
                w_packed.insert_raw(&name, st.quant, st.shape.clone(), st.bytes.clone());
                w_split.insert_raw(&name, st.quant, st.shape, st.bytes);
            }
        }

        let router_bytes = encode_quant_matrix(quant, 4, 256, 77);
        w_packed.insert_raw(
            "blk.0.ffn_gate_inp.weight",
            quant,
            vec![256, 4],
            router_bytes.clone(),
        );
        w_split.insert_raw(
            "blk.0.ffn_gate_inp.weight",
            quant,
            vec![256, 4],
            router_bytes,
        );

        for proj in ["gate", "up", "down"] {
            let mut stacked_bytes = Vec::new();
            for e in 0..4_u32 {
                let seed = 200 + e * 13 + u32::from(proj.as_bytes()[0]);
                let exp_bytes = encode_quant_matrix(quant, 256, 256, seed);
                stacked_bytes.extend_from_slice(&exp_bytes);
                w_split.insert_raw(
                    &format!("blk.0.ffn_{proj}.{e}.weight"),
                    quant,
                    vec![256, 256],
                    exp_bytes,
                );
            }
            w_packed.insert_raw(
                &format!("blk.0.ffn_{proj}_exps.weight"),
                quant,
                vec![256, 256, 4],
                stacked_bytes,
            );
        }

        let mut kv_packed = vec![0.0_f32; 2 * 4 * 128];
        let mut kv_split = vec![0.0_f32; 2 * 4 * 128];
        let mut logits_packed = vec![0.0_f32; 4];
        let mut logits_split = vec![0.0_f32; 4];

        let mut ctx_packed =
            ExecutionContext::new(&[1_u32, 2], 4, &mut logits_packed, &mut kv_packed).unwrap();
        let mut ctx_split =
            ExecutionContext::new(&[1_u32, 2], 4, &mut logits_split, &mut kv_split).unwrap();
        recipe
            .prefill_step(&mut ctx_packed, &w_packed)
            .expect("packed moe");
        recipe
            .prefill_step(&mut ctx_split, &w_split)
            .expect("split moe");

        let ref_logits = reference_forward_sequence(&recipe, &[1_u32, 2], &w_packed);
        verify_mixtral_parity_outputs(
            quant,
            &recipe,
            &w_packed,
            ctx_packed.logits(),
            ctx_split.logits(),
            &ref_logits,
        );
    }
}

fn verify_mixtral_parity_outputs(
    quant: QuantType,
    recipe: &MistralRecipe,
    w_packed: &IntegrationWeights,
    packed_logits: &[f32],
    split_logits: &[f32],
    ref_logits: &[f32],
) {
    for (i, &split_val) in split_logits.iter().enumerate() {
        assert!(
            (packed_logits[i] - split_val).abs() < 1e-6,
            "packed vs split MoE divergence for {quant:?}"
        );
    }
    assert_logits_parity(
        "Mixtral 256x256 MoE default path",
        quant,
        packed_logits,
        ref_logits,
        false,
    );

    let mut fb_cache = MistralLayerCache::new().with_max_repacked_bytes(0);
    let mut kv_fb = vec![0.0_f32; 2 * 4 * 128];
    let mut logits_fb = vec![0.0_f32; 4];
    let mut ctx_fb = ExecutionContext::new(&[1_u32, 2], 4, &mut logits_fb, &mut kv_fb).unwrap();
    recipe
        .prefill_step_with_cache(&mut ctx_fb, w_packed, &mut fb_cache)
        .expect("fallback moe prefill");
    assert_logits_parity(
        "Mixtral 256x256 MoE DirectFallback path",
        quant,
        ctx_fb.logits(),
        ref_logits,
        true,
    );
}

#[test]
fn test_from_gguf_mistral3_5120_4096_decoupled_q_tensor() {
    let mistral3_bytes = create_synthetic_gguf_v3_with_kvs(
        "mistral3",
        32,
        &[
            ("mistral3.block_count", GgufValue::Uint32(40)),
            ("mistral3.embedding_length", GgufValue::Uint32(5120)),
            ("mistral3.attention.head_count", GgufValue::Uint32(32)),
            ("mistral3.attention.head_count_kv", GgufValue::Uint32(8)),
            ("mistral3.feed_forward_length", GgufValue::Uint32(32_768)),
            ("mistral3.vocab_size", GgufValue::Uint32(131_072)),
            (
                "mistral3.rope.freq_base",
                GgufValue::Float32(1_000_000_000.0),
            ),
            ("mistral3.attention.sliding_window", GgufValue::Uint32(0)),
            ("mistral3.rope.scaling.type", GgufValue::String("yarn")),
            ("mistral3.rope.scaling.factor", GgufValue::Float32(4.0)),
            (
                "mistral3.rope.scaling.original_context_length",
                GgufValue::Uint32(8192),
            ),
        ],
        &[("blk.0.attn_q.weight", QuantType::Q4_K_M, &[5120, 4096])],
    );
    let mistral3_gguf = GgufFile::parse(&mistral3_bytes).expect("parse mistral3 gguf");
    let mistral3 = MistralRecipe::from_gguf(&mistral3_gguf).expect("hydrate mistral3");
    assert_eq!(mistral3.architecture_id(), "mistral3");
    assert_eq!(mistral3.hidden_dim, 5120);
    assert_eq!(mistral3.n_heads_q, 32);
    assert_eq!(mistral3.head_dim, 128);
    assert_eq!(mistral3.n_heads_q * mistral3.head_dim, 4096);
    assert_eq!(mistral3.ffn_dim, 32_768);
    assert_eq!(mistral3.vocab_size, 131_072);
    assert_eq!(mistral3.sliding_window, None);
    assert!((mistral3.rope_theta - 1_000_000_000.0).abs() < 1.0);
    assert_eq!(
        mistral3.rope_scaling,
        Some(RopeScalingConfig::Llama3Smooth {
            factor: 4.0,
            low_freq_factor: 1.0,
            high_freq_factor: 32.0,
            original_max_position_embeddings: 8192,
        })
    );
}
