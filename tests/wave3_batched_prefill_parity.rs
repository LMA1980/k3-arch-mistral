//! Wave 3 integration tests verifying batched `[M, K]` dense prefill GEMMs, causal + `SWA`
//! prefill attention, chunked prefill `start_position` / `compute_logits` contracts, `131,072`-vocab
//! final-row LM-head projection, and `Mixtral` token-bucketed active-expert prefill parity
//! across all 8 admitted quantizations.

use std::collections::HashMap;
use std::sync::Mutex;

use jero_arch_mistral::{
    DEFAULT_MEMLOCK_LIMIT_BYTES, MistralLayerCache, MistralRecipe, TEKKEN_VOCAB_SIZE,
    clear_global_layer_cache,
};
use jero_core::{ExecutionContext, K3Error, ModelWeights, QuantType, TensorView};

struct StoredTensor {
    name: String,
    quant: QuantType,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

#[derive(Default)]
struct CountingWeights {
    tensors: HashMap<String, StoredTensor>,
    calls: Mutex<HashMap<String, usize>>,
}

impl CountingWeights {
    fn insert(&mut self, name: &str, quant: QuantType, shape: Vec<usize>, bytes: Vec<u8>) {
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
        self.insert(name, QuantType::F32, shape, bytes);
    }

    fn remove(&mut self, name: &str) {
        self.tensors.remove(name);
    }

    fn reset_calls(&self) {
        self.calls.lock().unwrap().clear();
    }

    fn call_count(&self, name: &str) -> usize {
        self.calls.lock().unwrap().get(name).copied().unwrap_or(0)
    }
}

impl ModelWeights for CountingWeights {
    fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
        *self
            .calls
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

#[expect(
    clippy::cast_precision_loss,
    reason = "small synthetic indices fit cleanly"
)]
fn deterministic_scalar(idx: usize, seed: u32) -> f32 {
    let raw = i32::try_from((idx * 17 + seed as usize * 31) % 29).unwrap_or(0) - 14;
    (raw as f32) * 0.025
}

#[expect(
    clippy::cast_precision_loss,
    reason = "small synthetic indices fit cleanly"
)]
fn encode_block(quant: QuantType, block_bytes: usize, block_idx: usize, seed: u32) -> Vec<u8> {
    let d_bits = f32_to_f16_bits(0.02 + ((block_idx % 5) as f32) * 0.005);
    let dmin_bits = f32_to_f16_bits(0.004);
    let mut blk = vec![0_u8; block_bytes];
    match quant {
        QuantType::Q4_0 => {
            blk[0..2].copy_from_slice(&d_bits.to_le_bytes());
            for (idx, slot) in blk[2..18].iter_mut().enumerate() {
                let lo = u8::try_from((block_idx * 3 + idx + seed as usize) % 16).unwrap_or(0);
                let hi = u8::try_from((block_idx * 5 + idx + 3) % 16).unwrap_or(0);
                *slot = lo | (hi << 4);
            }
        }
        QuantType::Q8_0 => {
            blk[0..2].copy_from_slice(&d_bits.to_le_bytes());
            for (idx, slot) in blk[2..34].iter_mut().enumerate() {
                let q = i8::try_from(
                    i32::try_from((block_idx * 5 + idx + seed as usize) % 21).unwrap_or(0) - 10,
                )
                .unwrap_or(0);
                *slot = q.to_le_bytes()[0];
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

fn encode_matrix(quant: QuantType, rows: usize, cols: usize, seed: u32) -> Vec<u8> {
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
                out.extend_from_slice(&encode_block(quant, block_bytes, block_idx, seed));
            }
            out
        }
        other => panic!("unsupported test quant {other:?}"),
    }
}

#[expect(clippy::cast_precision_loss, reason = "small synthetic dimensions")]
fn build_dense_model(recipe: &MistralRecipe, q: QuantType, seed: u32) -> CountingWeights {
    let mut w = CountingWeights::default();
    let h = recipe.hidden_dim;
    w.insert(
        "token_embd.weight",
        q,
        vec![h, recipe.vocab_size],
        encode_matrix(q, recipe.vocab_size, h, seed + 1),
    );
    for l in 0..recipe.num_layers {
        let sb = seed + u32::try_from(l * 20).unwrap_or(0);
        w.insert_f32(
            &format!("blk.{l}.attn_norm.weight"),
            vec![h],
            &(0..h)
                .map(|i| 0.85 + ((i % 9) as f32) * 0.04)
                .collect::<Vec<_>>(),
        );
        w.insert_f32(
            &format!("blk.{l}.ffn_norm.weight"),
            vec![h],
            &(0..h)
                .map(|i| 0.90 + ((i % 7) as f32) * 0.03)
                .collect::<Vec<_>>(),
        );
        let q_out = recipe.n_heads_q * recipe.head_dim;
        let kv_out = recipe.n_heads_kv * recipe.head_dim;
        w.insert(
            &format!("blk.{l}.attn_q.weight"),
            q,
            vec![h, q_out],
            encode_matrix(q, q_out, h, sb + 1),
        );
        w.insert(
            &format!("blk.{l}.attn_k.weight"),
            q,
            vec![h, kv_out],
            encode_matrix(q, kv_out, h, sb + 2),
        );
        w.insert(
            &format!("blk.{l}.attn_v.weight"),
            q,
            vec![h, kv_out],
            encode_matrix(q, kv_out, h, sb + 3),
        );
        w.insert(
            &format!("blk.{l}.attn_output.weight"),
            q,
            vec![q_out, h],
            encode_matrix(q, h, q_out, sb + 4),
        );
        w.insert(
            &format!("blk.{l}.ffn_gate.weight"),
            q,
            vec![h, recipe.ffn_dim],
            encode_matrix(q, recipe.ffn_dim, h, sb + 5),
        );
        w.insert(
            &format!("blk.{l}.ffn_up.weight"),
            q,
            vec![h, recipe.ffn_dim],
            encode_matrix(q, recipe.ffn_dim, h, sb + 6),
        );
        w.insert(
            &format!("blk.{l}.ffn_down.weight"),
            q,
            vec![recipe.ffn_dim, h],
            encode_matrix(q, h, recipe.ffn_dim, sb + 7),
        );
    }
    w.insert_f32(
        "output_norm.weight",
        vec![h],
        &(0..h)
            .map(|i| 0.95 + ((i % 11) as f32) * 0.02)
            .collect::<Vec<_>>(),
    );
    w.insert(
        "output.weight",
        q,
        vec![h, recipe.vocab_size],
        encode_matrix(q, recipe.vocab_size, h, seed + 99),
    );
    w
}

fn assert_max_abs_diff(label: &str, got: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(got.len(), expected.len());
    for (idx, (&g, &e)) in got.iter().zip(expected).enumerate() {
        assert!(
            g.is_finite() && e.is_finite(),
            "{label}[{idx}] non-finite: {g} vs {e}"
        );
        let diff = (g - e).abs();
        assert!(
            diff < tol,
            "{label}[{idx}] diff {diff} >= {tol}: got={g}, expected={e}"
        );
    }
}

fn assert_cosine_and_l2(label: &str, got: &[f32], expected: &[f32], max_l2_rel: f64) {
    assert_eq!(got.len(), expected.len());
    let mut dot = 0.0_f64;
    let mut norm_g = 0.0_f64;
    let mut norm_e = 0.0_f64;
    let mut diff_sq = 0.0_f64;
    for (&g, &e) in got.iter().zip(expected) {
        assert!(g.is_finite() && e.is_finite());
        let gf = f64::from(g);
        let ef = f64::from(e);
        dot += gf * ef;
        norm_g += gf * gf;
        norm_e += ef * ef;
        diff_sq += (gf - ef) * (gf - ef);
    }
    let cosine = if norm_g > 1e-9 && norm_e > 1e-9 {
        dot / (norm_g.sqrt() * norm_e.sqrt())
    } else {
        1.0
    };
    let l2_rel = diff_sq.sqrt() / norm_e.sqrt().max(1.0);
    assert!(
        cosine > 0.999 && l2_rel < max_l2_rel,
        "{label} parity failure: cosine={cosine}, l2_rel={l2_rel}"
    );
}

fn run_dense_sequential_ref(
    recipe: &MistralRecipe,
    weights: &CountingWeights,
    prompt: &[u32],
    n_ctx: usize,
    kv_len: usize,
) -> Vec<f32> {
    let mut kv_seq = vec![0.0_f32; kv_len];
    let mut logits_seq = vec![0.0_f32; 8];
    let mut cache_seq = MistralLayerCache::new();
    for (pos, &tok) in prompt.iter().enumerate() {
        let t_slice = [tok];
        let mut ctx_s = ExecutionContext::new_prefill_chunk(
            &t_slice,
            n_ctx,
            pos,
            pos + 1 == prompt.len(),
            &mut logits_seq,
            &mut kv_seq,
        )
        .unwrap();
        recipe
            .prefill_step_with_cache(&mut ctx_s, weights, &mut cache_seq)
            .expect("seq prefill");
    }
    let mut ctx_seq_dec = ExecutionContext::new_prefill_chunk(
        &[0_u32],
        n_ctx,
        prompt.len(),
        false,
        &mut logits_seq,
        &mut kv_seq,
    )
    .unwrap();
    ctx_seq_dec.prepare_decode(6).unwrap();
    recipe
        .decode_step_with_cache(&mut ctx_seq_dec, weights, &mut cache_seq)
        .expect("seq decode");
    logits_seq
}

fn run_dense_quant_case(recipe: &MistralRecipe, quant: QuantType, prompt: &[u32], n_ctx: usize) {
    let kv_len = 2 * 2 * n_ctx * 128;
    let weights = build_dense_model(recipe, quant, 17);

    // 1. Batched M=6 prefill over warm cache executes 14 dense lookups (not 6*14=84).
    let mut warm_cache = MistralLayerCache::new();
    recipe.warm_layer_cache(&weights, &mut warm_cache).unwrap();
    let mut kv_batch = vec![0.0_f32; kv_len];
    let mut logits_batch = vec![0.0_f32; 8];
    let mut ctx_batch =
        ExecutionContext::new(prompt, n_ctx, &mut logits_batch, &mut kv_batch).unwrap();
    recipe
        .prefill_step_with_cache(&mut ctx_batch, &weights, &mut warm_cache)
        .expect("batched prefill");
    assert_eq!(warm_cache.stats().dense_hits, 14);
    assert_eq!(warm_cache.stats().norm_hits, 5);
    ctx_batch.prepare_decode(6).unwrap();
    recipe
        .decode_step_with_cache(&mut ctx_batch, &weights, &mut warm_cache)
        .expect("batched post-prefill decode");

    // 2. Chunked prefill: first chunk suppresses LM-head (`compute_logits=false`), second emits.
    weights.reset_calls();
    let mut kv_chunk = vec![0.0_f32; kv_len];
    let mut cache_chunk = MistralLayerCache::new();
    let mut no_logits: [f32; 0] = [];
    let mut ctx_c0 = ExecutionContext::new_prefill_chunk(
        &prompt[..3],
        n_ctx,
        0,
        false,
        &mut no_logits,
        &mut kv_chunk,
    )
    .unwrap();
    recipe
        .prefill_step_with_cache(&mut ctx_c0, &weights, &mut cache_chunk)
        .expect("chunk 0");
    assert_eq!(weights.call_count("output_norm.weight"), 0);
    assert_eq!(weights.call_count("output.weight"), 0);

    let mut logits_chunk = vec![0.0_f32; 8];
    let mut ctx_c1 = ExecutionContext::new_prefill_chunk(
        &prompt[3..],
        n_ctx,
        3,
        true,
        &mut logits_chunk,
        &mut kv_chunk,
    )
    .unwrap();
    recipe
        .prefill_step_with_cache(&mut ctx_c1, &weights, &mut cache_chunk)
        .expect("chunk 1");
    ctx_c1.prepare_decode(6).unwrap();
    recipe
        .decode_step_with_cache(&mut ctx_c1, &weights, &mut cache_chunk)
        .expect("chunked decode");

    // 3. Compare batched & chunked paths against strictly sequential reference.
    let logits_seq = run_dense_sequential_ref(recipe, &weights, prompt, n_ctx, kv_len);
    assert_max_abs_diff(
        &format!("batch vs seq {quant:?}"),
        ctx_batch.logits(),
        &logits_seq,
        1e-4,
    );
    assert_max_abs_diff(
        &format!("chunk vs seq {quant:?}"),
        ctx_c1.logits(),
        &logits_seq,
        1e-4,
    );

    // 4. Zero-budget DirectFallback parity check.
    let mut fb_cache = MistralLayerCache::new().with_max_repacked_bytes(0);
    let mut kv_fb = vec![0.0_f32; kv_len];
    let mut logits_fb = vec![0.0_f32; 8];
    let mut ctx_fb = ExecutionContext::new(prompt, n_ctx, &mut logits_fb, &mut kv_fb).unwrap();
    recipe
        .prefill_step_with_cache(&mut ctx_fb, &weights, &mut fb_cache)
        .expect("fb prefill");
    ctx_fb.prepare_decode(6).unwrap();
    recipe
        .decode_step_with_cache(&mut ctx_fb, &weights, &mut fb_cache)
        .expect("fb decode");
    assert_eq!(fb_cache.resident_repacked_bytes(), 0);
    let max_rel = if matches!(quant, QuantType::F32 | QuantType::F16 | QuantType::BF16) {
        1e-4
    } else {
        2e-2
    };
    assert_cosine_and_l2(
        &format!("256x256 batch vs fallback {quant:?}"),
        ctx_batch.logits(),
        ctx_fb.logits(),
        max_rel,
    );
}

#[test]
fn test_wave3_256x256_all_quants_batched_prefill_chunked_and_swa_parity() {
    clear_global_layer_cache();

    let mut recipe = MistralRecipe::new().with_sliding_window(Some(4));
    recipe.num_layers = 2;
    recipe.hidden_dim = 256;
    recipe.ffn_dim = 256;
    recipe.n_heads_q = 4;
    recipe.n_heads_kv = 2;
    recipe.head_dim = 64;
    recipe.vocab_size = 8;

    let prompt = [0_u32, 1, 2, 3, 4, 5]; // M = 6 > window 4
    let n_ctx = 12;

    for quant in [
        QuantType::F32,
        QuantType::F16,
        QuantType::BF16,
        QuantType::Q4_0,
        QuantType::Q4_K_M,
        QuantType::Q5_K,
        QuantType::Q6_K,
        QuantType::Q8_0,
    ] {
        run_dense_quant_case(&recipe, quant, &prompt, n_ctx);
    }
}

fn attach_mixtral_3d_experts(weights: &mut CountingWeights, quant: QuantType, n_exp: usize) {
    weights.remove("blk.0.ffn_gate.weight");
    weights.remove("blk.0.ffn_up.weight");
    weights.remove("blk.0.ffn_down.weight");
    weights.insert(
        "blk.0.ffn_gate_inp.weight",
        quant,
        vec![256, n_exp],
        encode_matrix(quant, n_exp, 256, 77),
    );
    for proj in ["gate", "up", "down"] {
        let mut stacked = Vec::new();
        for e in 0..n_exp {
            stacked.extend_from_slice(&encode_matrix(
                quant,
                256,
                256,
                200 + u32::try_from(e).unwrap_or(0) * 9,
            ));
        }
        weights.insert(
            &format!("blk.0.ffn_{proj}_exps.weight"),
            quant,
            vec![256, 256, n_exp],
            stacked,
        );
    }
}

fn verify_split_2d_expert_bucketing(recipe: &MistralRecipe, prompt: &[u32], n_ctx: usize) {
    let kv_len = 2 * n_ctx * 128;
    let mut split_w = build_dense_model(recipe, QuantType::F32, 5);
    split_w.remove("blk.0.ffn_gate.weight");
    split_w.remove("blk.0.ffn_up.weight");
    split_w.remove("blk.0.ffn_down.weight");

    let mut emb = vec![0.01_f32; 8 * 256];
    for t in 0..8 {
        emb[t * 256 + (t % 3)] = 10.0;
    }
    split_w.insert_f32("token_embd.weight", vec![256, 8], &emb);
    // Zero V projection keeps `ffn_norm` equal to normalized embeddings for deterministic routing.
    split_w.insert_f32(
        "blk.0.attn_v.weight",
        vec![256, 128],
        &vec![0.0_f32; 256 * 128],
    );

    let mut gate_inp = vec![0.0_f32; 4 * 256];
    gate_inp[0] = 6.0;
    gate_inp[1] = 4.0;
    gate_inp[2] = 5.0;
    gate_inp[256] = 4.0;
    gate_inp[256 + 1] = 2.0;
    gate_inp[256 + 2] = 6.0;
    gate_inp[512] = 2.0;
    gate_inp[512 + 1] = 6.0;
    gate_inp[512 + 2] = 2.0;
    split_w.insert_f32("blk.0.ffn_gate_inp.weight", vec![256, 4], &gate_inp);

    for e in 0..4 {
        let e_val = f32::from(u8::try_from(e).unwrap_or(0));
        for proj in ["gate", "up", "down"] {
            split_w.insert_f32(
                &format!("blk.0.ffn_{proj}.{e}.weight"),
                vec![256, 256],
                &vec![0.02 + e_val * 0.003; 256 * 256],
            );
        }
    }

    split_w.reset_calls();
    let mut kv_split = vec![0.0_f32; kv_len];
    let mut log_split = vec![0.0_f32; 8];
    let mut cache_split =
        MistralLayerCache::with_memlock_headroom(DEFAULT_MEMLOCK_LIMIT_BYTES, 26_300_000_000);
    let mut ctx_split =
        ExecutionContext::new(prompt, n_ctx, &mut log_split, &mut kv_split).unwrap();
    recipe
        .prefill_step_with_cache(&mut ctx_split, &split_w, &mut cache_split)
        .expect("split bucketed mixtral");
    for e in 0..3 {
        assert_eq!(split_w.call_count(&format!("blk.0.ffn_gate.{e}.weight")), 1);
        assert_eq!(split_w.call_count(&format!("blk.0.ffn_up.{e}.weight")), 1);
        assert_eq!(split_w.call_count(&format!("blk.0.ffn_down.{e}.weight")), 1);
    }
    assert_eq!(split_w.call_count("blk.0.ffn_gate.3.weight"), 0);
    assert_eq!(cache_split.stats().expert_zero_copy_slices, 9);
    assert_eq!(cache_split.cached_dense_count(), 5);
}

fn verify_packed_3d_mixtral_quant(
    recipe: &MistralRecipe,
    quant: QuantType,
    prompt: &[u32],
    n_ctx: usize,
) {
    let kv_len = 2 * n_ctx * 128;
    let mut weights = build_dense_model(recipe, quant, 31);
    attach_mixtral_3d_experts(&mut weights, quant, 4);

    let mut kv_batch = vec![0.0_f32; kv_len];
    let mut log_batch = vec![0.0_f32; 8];
    let mut cache_batch =
        MistralLayerCache::with_memlock_headroom(DEFAULT_MEMLOCK_LIMIT_BYTES, 26_300_000_000);
    let mut ctx_batch =
        ExecutionContext::new(prompt, n_ctx, &mut log_batch, &mut kv_batch).unwrap();
    recipe
        .prefill_step_with_cache(&mut ctx_batch, &weights, &mut cache_batch)
        .expect("3d batched mixtral");
    assert!(cache_batch.stats().expert_zero_copy_slices <= 12);
    assert_eq!(cache_batch.cached_dense_count(), 5);

    let mut kv_seq = vec![0.0_f32; kv_len];
    let mut log_seq = vec![0.0_f32; 8];
    let mut cache_seq =
        MistralLayerCache::with_memlock_headroom(DEFAULT_MEMLOCK_LIMIT_BYTES, 26_300_000_000);
    for (pos, &tok) in prompt.iter().enumerate() {
        let t_slice = [tok];
        let mut ctx_s = ExecutionContext::new_prefill_chunk(
            &t_slice,
            n_ctx,
            pos,
            pos + 1 == prompt.len(),
            &mut log_seq,
            &mut kv_seq,
        )
        .unwrap();
        recipe
            .prefill_step_with_cache(&mut ctx_s, &weights, &mut cache_seq)
            .expect("3d seq mixtral");
    }
    assert_max_abs_diff(
        &format!("mixtral 3d batch vs seq {quant:?}"),
        &log_batch,
        &log_seq,
        1e-4,
    );

    if matches!(quant, QuantType::F32 | QuantType::F16 | QuantType::BF16) {
        let mut kv_dec = vec![0.0_f32; kv_len];
        let mut log_dec = vec![0.0_f32; 8];
        let mut cache_dec = MistralLayerCache::new();
        let mut ctx_dec =
            ExecutionContext::new(&prompt[..1], n_ctx, &mut log_dec, &mut kv_dec).unwrap();
        recipe
            .prefill_step_with_cache(&mut ctx_dec, &weights, &mut cache_dec)
            .expect("mixtral decode baseline[0]");
        for &tok in &prompt[1..] {
            ctx_dec.prepare_decode(tok).unwrap();
            recipe
                .decode_step_with_cache(&mut ctx_dec, &weights, &mut cache_dec)
                .expect("mixtral decode step");
        }
        assert_max_abs_diff(
            &format!("mixtral batch vs decode {quant:?}"),
            &log_batch,
            &log_dec,
            1e-4,
        );
    }

    let mut kv_fb = vec![0.0_f32; kv_len];
    let mut log_fb = vec![0.0_f32; 8];
    let mut cache_fb = MistralLayerCache::new().with_max_repacked_bytes(0);
    let mut ctx_fb = ExecutionContext::new(prompt, n_ctx, &mut log_fb, &mut kv_fb).unwrap();
    recipe
        .prefill_step_with_cache(&mut ctx_fb, &weights, &mut cache_fb)
        .expect("mixtral fallback prefill");
    let max_rel = if matches!(quant, QuantType::F32 | QuantType::F16 | QuantType::BF16) {
        1e-4
    } else {
        2e-2
    };
    assert_cosine_and_l2(
        &format!("mixtral batch vs fallback {quant:?}"),
        &log_batch,
        &log_fb,
        max_rel,
    );
}

#[test]
fn test_wave3_mixtral_256x256_token_bucketed_moe_prefill_parity() {
    clear_global_layer_cache();

    let mut recipe = MistralRecipe::new_mixtral()
        .with_sliding_window(Some(4))
        .with_experts(4, 2);
    recipe.num_layers = 1;
    recipe.hidden_dim = 256;
    recipe.ffn_dim = 256;
    recipe.n_heads_q = 4;
    recipe.n_heads_kv = 2;
    recipe.head_dim = 64;
    recipe.vocab_size = 8;

    let prompt = [0_u32, 1, 2, 3, 4, 5];
    let n_ctx = 12;

    verify_split_2d_expert_bucketing(&recipe, &prompt, n_ctx);

    for quant in [
        QuantType::F32,
        QuantType::F16,
        QuantType::BF16,
        QuantType::Q4_0,
        QuantType::Q4_K_M,
        QuantType::Q5_K,
        QuantType::Q6_K,
        QuantType::Q8_0,
    ] {
        verify_packed_3d_mixtral_quant(&recipe, quant, &prompt, n_ctx);
    }
}

#[test]
fn test_wave3_tekken_131072_vocab_lm_head_projects_only_final_row() {
    clear_global_layer_cache();

    let mut recipe = MistralRecipe::new_mistral3().with_sliding_window(Some(4));
    recipe.num_layers = 1;
    recipe.hidden_dim = 64;
    recipe.ffn_dim = 64;
    recipe.n_heads_q = 2;
    recipe.n_heads_kv = 1;
    recipe.head_dim = 32;
    recipe.vocab_size = TEKKEN_VOCAB_SIZE;

    let weights = build_dense_model(&recipe, QuantType::Q4_0, 99);
    let prompt = [10_u32, 20, 30, 40, 50, 60, 70, 80];
    let n_ctx = 16;
    let kv_len = 2 * n_ctx * 32;

    // 1. `compute_logits=false` skips `output_norm.weight` and `output.weight` completely.
    weights.reset_calls();
    let mut kv_chunk = vec![0.0_f32; kv_len];
    let mut cache_chunk = MistralLayerCache::new();
    let mut empty_logits: [f32; 0] = [];
    let mut ctx_c0 = ExecutionContext::new_prefill_chunk(
        &prompt[..4],
        n_ctx,
        0,
        false,
        &mut empty_logits,
        &mut kv_chunk,
    )
    .unwrap();
    recipe
        .prefill_step_with_cache(&mut ctx_c0, &weights, &mut cache_chunk)
        .expect("tekken chunk 0");
    assert_eq!(weights.call_count("output_norm.weight"), 0);
    assert_eq!(weights.call_count("output.weight"), 0);

    let mut logits_chunk = vec![0.0_f32; TEKKEN_VOCAB_SIZE];
    let mut ctx_c1 = ExecutionContext::new_prefill_chunk(
        &prompt[4..],
        n_ctx,
        4,
        true,
        &mut logits_chunk,
        &mut kv_chunk,
    )
    .unwrap();
    recipe
        .prefill_step_with_cache(&mut ctx_c1, &weights, &mut cache_chunk)
        .expect("tekken chunk 1");

    // 2. Full M=8 batched prefill projects ONLY final row M-1 (`output.weight` read once).
    weights.reset_calls();
    let mut kv_batch = vec![0.0_f32; kv_len];
    let mut logits_batch = vec![0.0_f32; TEKKEN_VOCAB_SIZE];
    let mut cache_batch = MistralLayerCache::new();
    let mut ctx_batch =
        ExecutionContext::new(&prompt, n_ctx, &mut logits_batch, &mut kv_batch).unwrap();
    recipe
        .prefill_step_with_cache(&mut ctx_batch, &weights, &mut cache_batch)
        .expect("tekken full batch");
    assert_eq!(weights.call_count("output_norm.weight"), 1);
    assert_eq!(weights.call_count("output.weight"), 1);

    assert_max_abs_diff(
        "tekken 131k full vs chunked",
        &logits_batch,
        &logits_chunk,
        1e-4,
    );
}
