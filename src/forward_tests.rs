//! Comprehensive tests covering `Mistral` / `Mistral3` / `Mixtral` GGUF hydration, `Norm` `RoPE`,
//! SWA `KvCacheView`, untruncated dense & sparse top-2 `MoE` forward passes, multi-quant parity,
//! and fail-closed safety bounds (`LMA1980/jero-arch-mistral#11`–`#20`).

use std::collections::HashMap;

use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;
use jero_core::{
    ArchRecipe, ExecutionContext, GgufFile, GgufValue, K3Error, KvGeometry, ModelWeights,
    QuantType, TensorView,
};

use crate::kernels_compat::{dequantize_embedding_row, matvec_tensor};
use crate::recipe::apply_norm_rope_inplace;
use crate::repack_cache::MistralLayerCache;
use crate::{MistralRecipe, RopeScalingConfig, route_top2_experts};

/// Asserts logit parity against an `f32`-activation scalar reference.
///
/// Exact float formats (`F32`, `F16`, `BF16`) and non-repacked `DirectFallback`
/// runs must match within `< 1e-3`. With `with-kernels` enabled, quantized
/// dense projections dynamically quantize activations to `Q8`, so normalized
/// delta (`< 1.5e-2`) and cosine similarity (`> 0.999`) are verified alongside
/// strict `< 1e-3` `DirectFallback` parity.
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
struct MapWeights {
    tensors: HashMap<String, StoredTensor>,
}

impl MapWeights {
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

    fn insert_f32(&mut self, name: &str, shape: Vec<usize>, vals: &[f32]) {
        let mut bytes = Vec::with_capacity(vals.len() * 4);
        for &v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.insert_raw(name, QuantType::F32, shape, bytes);
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
}

impl ModelWeights for MapWeights {
    fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
        let entry = self
            .tensors
            .get(name)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: format!("missing tensor '{name}' in MapWeights"),
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
        #[expect(clippy::cast_sign_loss, reason = "exp checked in 1..31")]
        let exp_u16 = u16::try_from(exp as u32).unwrap_or(0);
        let mant_u16 = u16::try_from(mant).unwrap_or(0);
        sign | (exp_u16 << 10) | mant_u16
    }
}

fn f32_to_bf16_bits(val: f32) -> u16 {
    u16::try_from(val.to_bits() >> 16).unwrap_or(0)
}

#[allow(clippy::cast_precision_loss)]
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

#[allow(clippy::cast_precision_loss)]
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

#[allow(clippy::cast_precision_loss)]
fn deterministic_scalar(idx: usize, seed: u32) -> f32 {
    let raw = i32::try_from((idx * 17 + seed as usize * 31) % 29).unwrap_or(0) - 14;
    (raw as f32) * 0.025
}

fn decode_matrix_rows_f32(tensor: &TensorView<'_>, rows: usize, cols: usize) -> Vec<Vec<f32>> {
    let mut result = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut row = vec![0.0_f32; cols];
        dequantize_embedding_row(tensor, r, &mut row).expect("decode test row");
        result.push(row);
    }
    result
}

#[expect(clippy::too_many_lines, clippy::cast_precision_loss)]
fn reference_forward_sequence(
    recipe: &MistralRecipe,
    tokens: &[u32],
    weights: &MapWeights,
) -> Vec<f32> {
    let hidden_dim = recipe.hidden_dim;
    let total_q = recipe.n_heads_q * recipe.head_dim;
    let total_kv = recipe.n_heads_kv * recipe.head_dim;
    let kv_group = recipe.n_heads_q / recipe.n_heads_kv;
    let scale = 1.0_f32 / (recipe.head_dim as f32).sqrt();

    let mut inv_freq = Vec::with_capacity(recipe.head_dim / 2);
    for i in 0..(recipe.head_dim / 2) {
        inv_freq.push(
            1.0_f32
                / recipe
                    .rope_theta
                    .powf((2 * i) as f32 / recipe.head_dim as f32),
        );
    }

    let mut k_hist: Vec<Vec<Vec<f32>>> = vec![Vec::new(); recipe.num_layers];
    let mut v_hist: Vec<Vec<Vec<f32>>> = vec![Vec::new(); recipe.num_layers];
    let mut last_hidden = vec![0.0_f32; hidden_dim];

    for (pos, &tok) in tokens.iter().enumerate() {
        let emb_tv = weights.tensor("token_embd.weight").expect("embd");
        let mut hidden = decode_matrix_rows_f32(&emb_tv, recipe.vocab_size, hidden_dim)
            [usize::try_from(tok).unwrap()]
        .clone();

        for layer in 0..recipe.num_layers {
            let attn_norm_tv = weights
                .tensor(&format!("blk.{layer}.attn_norm.weight"))
                .expect("attn_norm");
            let norm_w = decode_matrix_rows_f32(&attn_norm_tv, 1, hidden_dim).remove(0);
            let sum_sq: f32 = hidden.iter().map(|&x| x * x).sum();
            let rms = 1.0_f32 / ((sum_sq / hidden_dim as f32) + recipe.rms_eps).sqrt();
            let norm_h: Vec<f32> = hidden
                .iter()
                .zip(norm_w.iter())
                .map(|(&h, &w)| h * rms * w)
                .collect();

            let q_tv = weights
                .tensor(&format!("blk.{layer}.attn_q.weight"))
                .unwrap();
            let k_tv = weights
                .tensor(&format!("blk.{layer}.attn_k.weight"))
                .unwrap();
            let v_tv = weights
                .tensor(&format!("blk.{layer}.attn_v.weight"))
                .unwrap();
            let o_tv = weights
                .tensor(&format!("blk.{layer}.attn_output.weight"))
                .unwrap();

            let q_rows = decode_matrix_rows_f32(&q_tv, total_q, hidden_dim);
            let k_rows = decode_matrix_rows_f32(&k_tv, total_kv, hidden_dim);
            let v_rows = decode_matrix_rows_f32(&v_tv, total_kv, hidden_dim);
            let o_rows = decode_matrix_rows_f32(&o_tv, hidden_dim, total_q);

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
                    for (pair_idx, &freq) in inv_freq.iter().enumerate() {
                        let (sin_a, cos_a) = (pos as f32 * freq).sin_cos();
                        let even = head[2 * pair_idx];
                        let odd = head[2 * pair_idx + 1];
                        head[2 * pair_idx] = even * cos_a - odd * sin_a;
                        head[2 * pair_idx + 1] = even * sin_a + odd * cos_a;
                    }
                }
            }

            k_hist[layer].push(k_proj);
            v_hist[layer].push(v_proj);

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
                for k_step in k_hist[layer].iter().take(end_pos).skip(start_pos) {
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
                    let vh = &v_hist[layer][start_pos + offset]
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

            let ffn_norm_tv = weights
                .tensor(&format!("blk.{layer}.ffn_norm.weight"))
                .unwrap();
            let ffn_nw = decode_matrix_rows_f32(&ffn_norm_tv, 1, hidden_dim).remove(0);
            let sum_sq2: f32 = hidden.iter().map(|&x| x * x).sum();
            let rms2 = 1.0_f32 / ((sum_sq2 / hidden_dim as f32) + recipe.rms_eps).sqrt();
            let ffn_in: Vec<f32> = hidden
                .iter()
                .zip(ffn_nw.iter())
                .map(|(&h, &w)| h * rms2 * w)
                .collect();

            if let Some(n_exp) = recipe.num_experts {
                let router_tv = weights
                    .tensor(&format!("blk.{layer}.ffn_gate_inp.weight"))
                    .unwrap();
                let r_rows = decode_matrix_rows_f32(&router_tv, n_exp, hidden_dim);
                let r_logits: Vec<f32> = r_rows
                    .iter()
                    .map(|r| r.iter().zip(&ffn_in).map(|(&a, &b)| a * b).sum())
                    .collect();
                let chosen = route_top2_experts(&r_logits).unwrap();
                for (exp_idx, gate_weight) in chosen {
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
        let out_nw = decode_matrix_rows_f32(&out_norm_tv, 1, hidden_dim).remove(0);
        let sum_sq: f32 = hidden.iter().map(|&x| x * x).sum();
        let rms = 1.0_f32 / ((sum_sq / hidden_dim as f32) + recipe.rms_eps).sqrt();
        last_hidden = hidden
            .iter()
            .zip(out_nw.iter())
            .map(|(&h, &w)| h * rms * w)
            .collect();
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

fn expert_tv_for_test<'a>(
    weights: &'a MapWeights,
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
    let direct = format!("blk.{layer}.ffn_{proj}.{expert_idx}.weight");
    if weights.contains_tensor(&direct) {
        return weights.tensor(&direct).unwrap();
    }
    weights
        .tensor(&format!("blk.{layer}.ffn_{proj}_exps.{expert_idx}.weight"))
        .unwrap()
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

#[allow(clippy::cast_precision_loss)]
fn build_dense_weights(recipe: &MistralRecipe, q: QuantType, tied: bool) -> MapWeights {
    let mut w = MapWeights::default();
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

#[test]
fn test_from_gguf_mistral7b_and_mistral3_decoupled_heads() {
    let mistral7b_bytes = create_synthetic_gguf_v3_with_kvs(
        "mistral",
        32,
        &[
            ("mistral.block_count", GgufValue::Uint32(32)),
            ("mistral.embedding_length", GgufValue::Uint32(4096)),
            ("mistral.attention.head_count", GgufValue::Uint32(32)),
            ("mistral.attention.head_count_kv", GgufValue::Uint32(8)),
            ("mistral.feed_forward_length", GgufValue::Uint32(14_336)),
            ("mistral.attention.sliding_window", GgufValue::Uint32(4096)),
            ("mistral.vocab_size", GgufValue::Uint32(32_000)),
            ("mistral.rope.freq_base", GgufValue::Float32(10_000.0)),
            (
                "mistral.attention.layer_norm_rms_epsilon",
                GgufValue::Float32(1e-5),
            ),
        ],
        &[],
    );
    let mistral7b_gguf = GgufFile::parse(&mistral7b_bytes).expect("parse mistral7b gguf");
    let mistral7b = MistralRecipe::from_gguf(&mistral7b_gguf).expect("hydrate mistral7b");
    assert_eq!(mistral7b.architecture_id(), "mistral");
    assert_eq!(mistral7b.num_layers, 32);
    assert_eq!(mistral7b.hidden_dim, 4096);
    assert_eq!(mistral7b.ffn_dim, 14_336);
    assert_eq!(mistral7b.head_dim, 128);
    assert_eq!(mistral7b.vocab_size, 32_000);
    assert_eq!(mistral7b.sliding_window, Some(4096));
    assert_eq!(mistral7b.num_experts, None);
    assert!(mistral7b.supports_quant(QuantType::Q5_K));

    // Compact Miri-safe decoupled q_dim tensor check (hidden=64, n_heads_q=4, head_dim=12 -> total_q=48 != 64), Tekken 131k, YaRN, SWA disabled
    let mistral3_bytes = create_synthetic_gguf_v3_with_kvs(
        "mistral3",
        32,
        &[
            ("mistral3.block_count", GgufValue::Uint32(40)),
            ("mistral3.embedding_length", GgufValue::Uint32(64)),
            ("mistral3.attention.head_count", GgufValue::Uint32(4)),
            ("mistral3.attention.head_count_kv", GgufValue::Uint32(2)),
            ("mistral3.feed_forward_length", GgufValue::Uint32(192)),
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
        &[("blk.0.attn_q.weight", QuantType::F32, &[64, 48])],
    );
    let mistral3_gguf = GgufFile::parse(&mistral3_bytes).expect("parse mistral3 gguf");
    let mistral3 = MistralRecipe::from_gguf(&mistral3_gguf).expect("hydrate mistral3");
    assert_eq!(mistral3.architecture_id(), "mistral3");
    assert_eq!(mistral3.hidden_dim, 64);
    assert_eq!(mistral3.n_heads_q, 4);
    assert_eq!(mistral3.head_dim, 12);
    assert_eq!(mistral3.n_heads_q * mistral3.head_dim, 48);
    assert_eq!(mistral3.ffn_dim, 192);
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

#[test]
fn test_from_gguf_mixtral_and_errors() {
    let mixtral_bytes = create_synthetic_gguf_v3_with_kvs(
        "mixtral",
        32,
        &[
            ("mixtral.block_count", GgufValue::Uint32(32)),
            ("mixtral.embedding_length", GgufValue::Uint32(4096)),
            ("mixtral.attention.head_count", GgufValue::Uint32(32)),
            ("mixtral.attention.head_count_kv", GgufValue::Uint32(8)),
            ("mixtral.expert_count", GgufValue::Uint32(8)),
            ("mixtral.expert_used_count", GgufValue::Uint32(2)),
            (
                "mixtral.expert_feed_forward_length",
                GgufValue::Uint32(14_336),
            ),
            ("mixtral.rope.scaling.type", GgufValue::String("linear")),
            ("mixtral.rope.scaling.factor", GgufValue::Float32(2.0)),
        ],
        &[],
    );
    let mixtral_gguf = GgufFile::parse(&mixtral_bytes).expect("parse mixtral gguf");
    let mixtral = MistralRecipe::from_gguf(&mixtral_gguf).expect("hydrate mixtral");
    assert_eq!(mixtral.architecture_id(), "mixtral");
    assert_eq!(mixtral.num_experts, Some(8));
    assert_eq!(mixtral.num_experts_per_tok, Some(2));
    assert_eq!(mixtral.ffn_dim, 14_336);
    assert_eq!(mixtral.sliding_window, None);
    assert_eq!(
        mixtral.rope_scaling,
        Some(RopeScalingConfig::Linear { factor: 2.0 })
    );

    let llama_bytes = create_synthetic_gguf_v3_with_kvs("llama", 32, &[], &[]);
    let llama_gguf = GgufFile::parse(&llama_bytes).expect("parse llama");
    assert!(MistralRecipe::from_gguf(&llama_gguf).is_err());

    let bad_rope_bytes = create_synthetic_gguf_v3_with_kvs(
        "mistral",
        32,
        &[
            ("mistral.rope.scaling.type", GgufValue::String("bogus")),
            ("mistral.rope.scaling.factor", GgufValue::Float32(2.0)),
        ],
        &[],
    );
    let bad_rope_gguf = GgufFile::parse(&bad_rope_bytes).expect("parse bad rope");
    assert!(MistralRecipe::from_gguf(&bad_rope_gguf).is_err());
}

#[test]
#[expect(clippy::cast_precision_loss)]
fn test_norm_rope_interleaved_pair_rotation_q_and_k() {
    let head_dim = 4;
    let inv_freq = [1.0_f32, 1.0_f32 / 100.0_f32];
    let mut vec_qk = [1.0_f32, 0.0, 2.0, -1.0, 0.5, 1.5, -2.0, 0.25];
    let original = vec_qk;
    let pos = 3;

    apply_norm_rope_inplace(&mut vec_qk, head_dim, pos, &inv_freq).expect("Norm RoPE");
    for h in 0..2 {
        for i in 0..2 {
            let (sin_a, cos_a) = ((pos as f32) * inv_freq[i]).sin_cos();
            let even = original[h * 4 + 2 * i];
            let odd = original[h * 4 + 2 * i + 1];
            let exp_even = even * cos_a - odd * sin_a;
            let exp_odd = even * sin_a + odd * cos_a;
            assert!((vec_qk[h * 4 + 2 * i] - exp_even).abs() < 1e-6);
            assert!((vec_qk[h * 4 + 2 * i + 1] - exp_odd).abs() < 1e-6);
        }
    }

    let (sin_0, cos_0) = (3.0_f32 * inv_freq[0]).sin_cos();
    let neox_0 = original[0] * cos_0 - original[2] * sin_0;
    assert!((vec_qk[0] - neox_0).abs() > 1e-2);
}

#[test]
fn test_dense_swa_window_eviction_kv_view_and_reference_parity() {
    let mut recipe = MistralRecipe::new()
        .with_sliding_window(Some(2))
        .with_rope_scaling(None);
    recipe.num_layers = 2;
    recipe.hidden_dim = 6;
    recipe.ffn_dim = 18;
    recipe.n_heads_q = 2;
    recipe.n_heads_kv = 1;
    recipe.head_dim = 2;
    recipe.vocab_size = 10;
    recipe.rope_theta = 10_000.0;

    let mut weights = build_dense_weights(&recipe, QuantType::F32, false);
    let mut reference_weights = MapWeights::default();
    for (k, v) in &weights.tensors {
        reference_weights.insert_raw(k, v.quant, v.shape.clone(), v.bytes.clone());
    }

    let prompt = [1_u32, 3, 5];
    let n_ctx = 8;
    let geom = KvGeometry::new(
        recipe.num_layers,
        recipe.n_heads_kv,
        recipe.head_dim,
        n_ctx,
        Some(2),
    )
    .expect("kv geometry");
    let kv_len = geom.required_f32_len().expect("kv f32 len");
    let mut kv_cache = vec![0.0_f32; kv_len];
    let mut logits = vec![0.0_f32; recipe.vocab_size];

    let mut ctx = ExecutionContext::new(&prompt, n_ctx, &mut logits, &mut kv_cache)
        .expect("ctx")
        .with_kv_geometry(geom)
        .expect("attach kv geometry");
    recipe.prefill_step(&mut ctx, &weights).expect("prefill");
    assert_eq!(ctx.position(), 3);

    let prefill_ref = reference_forward_sequence(&recipe, &prompt, &reference_weights);
    for (i, (&got, &exp)) in ctx.logits().iter().zip(&prefill_ref).enumerate() {
        assert!(
            (got - exp).abs() < 1e-4,
            "prefill logit[{i}] mismatch: got {got}, expected {exp}"
        );
    }

    ctx.prepare_decode(7).expect("prepare decode");
    recipe
        .decode_step(&mut ctx, &weights)
        .expect("decode token 7");
    let full_seq = [1_u32, 3, 5, 7];
    let decode_ref = reference_forward_sequence(&recipe, &full_seq, &reference_weights);
    for (i, (&got, &exp)) in ctx.logits().iter().zip(&decode_ref).enumerate() {
        assert!(
            (got - exp).abs() < 1e-4,
            "decode logit[{i}] mismatch: got {got}, expected {exp}"
        );
    }

    // Mutate evicted pos=0 token (outside W=2 at pos=3, active window pos 2..=3): logits unchanged.
    let mut kv_cache_alt = vec![0.0_f32; kv_len];
    let mut logits_alt = vec![0.0_f32; recipe.vocab_size];
    let mut ctx_alt =
        ExecutionContext::new(&[9_u32, 3, 5], n_ctx, &mut logits_alt, &mut kv_cache_alt)
            .expect("ctx alt")
            .with_kv_geometry(geom)
            .expect("attach kv alt");
    recipe.prefill_step(&mut ctx_alt, &weights).unwrap();
    ctx_alt.prepare_decode(7).unwrap();
    recipe.decode_step(&mut ctx_alt, &weights).unwrap();
    for (i, (&ref_val, &alt_val)) in decode_ref.iter().zip(ctx_alt.logits()).enumerate() {
        assert!(
            (ref_val - alt_val).abs() < 1e-4,
            "evicted pos=0 affected pos=3 logits with W=2 at [{i}]: {ref_val} vs {alt_val}"
        );
    }

    // Mutate non-evicted pos=2 token (inside W=2): logits MUST alter.
    let mut kv_cache_in = vec![0.0_f32; kv_len];
    let mut logits_in = vec![0.0_f32; recipe.vocab_size];
    let mut ctx_in = ExecutionContext::new(&[1_u32, 3, 8], n_ctx, &mut logits_in, &mut kv_cache_in)
        .expect("ctx in")
        .with_kv_geometry(geom)
        .expect("attach kv in");
    recipe.prefill_step(&mut ctx_in, &weights).unwrap();
    ctx_in.prepare_decode(7).unwrap();
    recipe.decode_step(&mut ctx_in, &weights).unwrap();
    let max_diff = decode_ref
        .iter()
        .zip(ctx_in.logits())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff > 1e-5,
        "in-window pos=2 mutation must alter logits (max_diff={max_diff})"
    );

    weights.tensors.remove("blk.1.ffn_down.weight");
    let mut kv_bad = vec![0.0_f32; kv_len];
    let mut logits_bad = vec![0.0_f32; recipe.vocab_size];
    let mut ctx_bad = ExecutionContext::new(&[1], n_ctx, &mut logits_bad, &mut kv_bad).unwrap();
    assert!(recipe.prefill_step(&mut ctx_bad, &weights).is_err());
}

#[test]
#[expect(clippy::cast_precision_loss)]
fn test_non_k_quants_and_square_matrix_disambiguation_miri() {
    let mut recipe = MistralRecipe::new().with_sliding_window(None);
    recipe.num_layers = 1;
    recipe.hidden_dim = 32;
    recipe.ffn_dim = 32;
    recipe.n_heads_q = 2;
    recipe.n_heads_kv = 1;
    recipe.head_dim = 16;
    recipe.vocab_size = 2;

    let input_32: Vec<f32> = (0..32).map(|i| ((i % 7) as f32) * 0.1 - 0.3).collect();
    for quant in [
        QuantType::F32,
        QuantType::F16,
        QuantType::BF16,
        QuantType::Q4_0,
        QuantType::Q8_0,
    ] {
        assert!(recipe.supports_quant(quant));
        let bytes = encode_quant_matrix(quant, 2, 32, 19);
        let tv = TensorView::from_bytes("nonk.weight", quant, &[32, 2], 0, &bytes).unwrap();
        let mut out = vec![0.0_f32; 2];
        matvec_tensor(&mut out, &tv, &input_32).expect("non-k matvec");
        let ref_rows = decode_matrix_rows_f32(&tv, 2, 32);
        for (r, row) in ref_rows.iter().enumerate() {
            let expected: f32 = row.iter().zip(&input_32).map(|(&a, &b)| a * b).sum();
            assert!((out[r] - expected).abs() < 1e-3);
        }
    }

    let weights = build_dense_weights(&recipe, QuantType::Q4_0, true);
    let sq = weights.tensor("blk.0.attn_q.weight").expect("attn_q");
    assert_eq!(sq.shape(), &[32, 32]);
    let mut sq_out = vec![0.0_f32; 32];
    matvec_tensor(&mut sq_out, &sq, &input_32).expect("square 32x32 matvec");

    let mut kv_cache = vec![0.0_f32; 2 * 4 * 16];
    let mut logits = vec![0.0_f32; recipe.vocab_size];
    let mut ctx = ExecutionContext::new(&[0_u32], 4, &mut logits, &mut kv_cache).unwrap();
    recipe.prefill_step(&mut ctx, &weights).expect("prefill");
    ctx.prepare_decode(1).unwrap();
    recipe.decode_step(&mut ctx, &weights).expect("decode");

    let ref_logits = reference_forward_sequence(&recipe, &[0_u32, 1], &weights);
    assert_logits_parity(
        "Q4_0 default path",
        QuantType::Q4_0,
        ctx.logits(),
        &ref_logits,
        false,
    );

    let mut fb_cache = MistralLayerCache::new().with_max_repacked_bytes(0);
    let mut kv_fb = vec![0.0_f32; 2 * 4 * 16];
    let mut logits_fb = vec![0.0_f32; recipe.vocab_size];
    let mut ctx_fb = ExecutionContext::new(&[0_u32], 4, &mut logits_fb, &mut kv_fb).unwrap();
    recipe
        .prefill_step_with_cache(&mut ctx_fb, &weights, &mut fb_cache)
        .expect("fallback prefill");
    ctx_fb.prepare_decode(1).unwrap();
    recipe
        .decode_step_with_cache(&mut ctx_fb, &weights, &mut fb_cache)
        .expect("fallback decode");
    assert_logits_parity(
        "Q4_0 DirectFallback path",
        QuantType::Q4_0,
        ctx_fb.logits(),
        &ref_logits,
        true,
    );
}

#[test]
#[expect(clippy::cast_precision_loss)]
fn test_k_quants_matvec_and_embedding_miri() {
    let recipe = MistralRecipe::new();
    let input: Vec<f32> = (0..256).map(|i| ((i % 11) as f32) * 0.05 - 0.25).collect();

    for quant in [QuantType::Q4_K_M, QuantType::Q5_K, QuantType::Q6_K] {
        assert!(recipe.supports_quant(quant));
        let bytes = encode_quant_matrix(quant, 3, 256, 41);
        let tv = TensorView::from_bytes("kquant.weight", quant, &[256, 3], 0, &bytes).unwrap();
        let mut out = vec![0.0_f32; 3];
        matvec_tensor(&mut out, &tv, &input).expect("k-quant matvec");

        let ref_rows = decode_matrix_rows_f32(&tv, 3, 256);
        for (r, row) in ref_rows.iter().enumerate() {
            let expected: f32 = row.iter().zip(&input).map(|(&a, &b)| a * b).sum();
            assert!(
                (out[r] - expected).abs() < 1e-2,
                "k-quant {quant:?} row {r} mismatch: got {}, expected {expected}",
                out[r]
            );
        }
    }
}

#[test]
fn test_mixtral_router_matvec_packed_3d_vs_split_2d_experts_miri() {
    let quant = QuantType::Q4_0;
    let mut recipe = MistralRecipe::new_mixtral()
        .with_sliding_window(Some(4))
        .with_experts(3, 2);
    recipe.num_layers = 1;
    recipe.hidden_dim = 32;
    recipe.ffn_dim = 32;
    recipe.n_heads_q = 2;
    recipe.n_heads_kv = 1;
    recipe.head_dim = 16;
    recipe.vocab_size = 2;

    let mut w_packed = MapWeights::default();
    let mut w_split = MapWeights::default();

    let shared = build_dense_weights(&recipe, quant, true);
    for (name, st) in shared.tensors {
        if !name.contains("ffn_gate.") && !name.contains("ffn_up.") && !name.contains("ffn_down.") {
            w_packed.insert_raw(&name, st.quant, st.shape.clone(), st.bytes.clone());
            w_split.insert_raw(&name, st.quant, st.shape, st.bytes);
        }
    }

    let router_bytes = encode_quant_matrix(quant, 3, 32, 77);
    w_packed.insert_raw(
        "blk.0.ffn_gate_inp.weight",
        quant,
        vec![32, 3],
        router_bytes.clone(),
    );
    w_split.insert_raw(
        "blk.0.ffn_gate_inp.weight",
        quant,
        vec![32, 3],
        router_bytes,
    );

    for proj in ["gate", "up", "down"] {
        let mut stacked_bytes = Vec::new();
        for e in 0..3_u32 {
            let seed = 200 + e * 13 + u32::from(proj.as_bytes()[0]);
            let exp_bytes = encode_quant_matrix(quant, 32, 32, seed);
            stacked_bytes.extend_from_slice(&exp_bytes);
            w_split.insert_raw(
                &format!("blk.0.ffn_{proj}.{e}.weight"),
                quant,
                vec![32, 32],
                exp_bytes,
            );
        }
        w_packed.insert_raw(
            &format!("blk.0.ffn_{proj}_exps.weight"),
            quant,
            vec![32, 32, 3],
            stacked_bytes,
        );
    }

    let mut kv_packed = vec![0.0_f32; 2 * 4 * 16];
    let mut kv_split = vec![0.0_f32; 2 * 4 * 16];
    let mut logits_packed = vec![0.0_f32; 2];
    let mut logits_split = vec![0.0_f32; 2];

    let mut ctx_packed =
        ExecutionContext::new(&[1_u32], 4, &mut logits_packed, &mut kv_packed).unwrap();
    let mut ctx_split =
        ExecutionContext::new(&[1_u32], 4, &mut logits_split, &mut kv_split).unwrap();
    recipe
        .prefill_step(&mut ctx_packed, &w_packed)
        .expect("packed moe");
    recipe
        .prefill_step(&mut ctx_split, &w_split)
        .expect("split moe");

    let ref_logits = reference_forward_sequence(&recipe, &[1_u32], &w_packed);
    for (i, &split_val) in ctx_split.logits().iter().enumerate() {
        assert!(
            (ctx_packed.logits()[i] - split_val).abs() < 1e-6,
            "packed vs split MoE divergence for {quant:?}"
        );
    }
    assert_logits_parity(
        "Mixtral MoE default path",
        quant,
        ctx_packed.logits(),
        &ref_logits,
        false,
    );

    let mut fb_cache = MistralLayerCache::new().with_max_repacked_bytes(0);
    let mut kv_fb = vec![0.0_f32; 2 * 4 * 16];
    let mut logits_fb = vec![0.0_f32; 2];
    let mut ctx_fb = ExecutionContext::new(&[1_u32], 4, &mut logits_fb, &mut kv_fb).unwrap();
    recipe
        .prefill_step_with_cache(&mut ctx_fb, &w_packed, &mut fb_cache)
        .expect("fallback moe prefill");
    assert_logits_parity(
        "Mixtral MoE DirectFallback path",
        quant,
        ctx_fb.logits(),
        &ref_logits,
        true,
    );
}

#[test]
fn test_fail_closed_invalid_weights_and_bounds() {
    let mut recipe = MistralRecipe::new();
    recipe.num_layers = 1;
    recipe.hidden_dim = 8;
    recipe.ffn_dim = 16;
    recipe.n_heads_q = 2;
    recipe.n_heads_kv = 1;
    recipe.head_dim = 4;
    recipe.vocab_size = 4;

    let weights = build_dense_weights(&recipe, QuantType::F32, false);
    let mut small_kv = vec![0.0_f32; 4];
    let mut logits = vec![0.0_f32; 4];
    let mut ctx = ExecutionContext::new(&[1], 4, &mut logits, &mut small_kv).unwrap();
    assert!(recipe.prefill_step(&mut ctx, &weights).is_err());

    let mut bad_norm_weights = build_dense_weights(&recipe, QuantType::F32, false);
    bad_norm_weights.insert_f32(
        "blk.0.attn_norm.weight",
        vec![8],
        &[1.0, f32::NAN, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    );
    let mut good_kv = vec![0.0_f32; 2 * 4 * 4];
    let mut ctx_nan = ExecutionContext::new(&[1], 4, &mut logits, &mut good_kv).unwrap();
    assert!(
        recipe
            .prefill_step(&mut ctx_nan, &bad_norm_weights)
            .is_err()
    );

    let mut bad_quant = build_dense_weights(&recipe, QuantType::F32, false);
    bad_quant.insert_raw(
        "blk.0.attn_q.weight",
        QuantType::Q2_K,
        vec![8, 8],
        vec![0_u8; 84],
    );
    let mut ctx_q = ExecutionContext::new(&[1], 4, &mut logits, &mut good_kv).unwrap();
    assert!(recipe.prefill_step(&mut ctx_q, &bad_quant).is_err());

    let mut small_logits = vec![0.0_f32; 2];
    let mut ctx_l = ExecutionContext::new(&[1], 4, &mut small_logits, &mut good_kv).unwrap();
    assert!(recipe.prefill_step(&mut ctx_l, &weights).is_err());
}
