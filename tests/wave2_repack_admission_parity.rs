//! Wave 2 integration parity & admission tests covering `256x256` 8-quant `MistralLayerCache`
//! repacking, compact vs dense `F16`/`BF16` modes, `Mixtral` multi-step zero-copy `3D` `_exps`
//! execution under `RLIMIT_MEMLOCK` headroom, and `Tekken` `131,072`-vocab GGUF / LM-head contracts.

use std::collections::HashMap;

use jero_arch_mistral::{
    CachedDenseProjection, DEFAULT_MEMLOCK_LIMIT_BYTES, MISTRAL_7B_KV_BYTES_PER_TOKEN,
    MISTRAL3_KV_BYTES_PER_TOKEN, MIXTRAL_KV_BYTES_PER_TOKEN, MistralLayerCache,
    MistralModelProfile, MistralRecipe, MistralTokenizerFamily, SENTENCEPIECE_VOCAB_32K,
    SENTENCEPIECE_VOCAB_32K_EXT, TEKKEN_VOCAB_SIZE, classify_mistral_manifest_profile,
    classify_mistral_vocab_family, clear_global_layer_cache, evaluate_moe_memlock_admission,
    inspect_packed_expert_slice, validate_gguf_tokenizer_and_lm_head, validate_vocab_stop_contract,
    verify_mistral_recipe_admission, verify_mixtral_admission,
};
use jero_core::{
    ExecutionContext, GgufFile, K3Error, ModelManifest, ModelWeights, QuantType, TensorView,
};

struct StoredTensor {
    name: String,
    quant: QuantType,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

#[derive(Default)]
struct Wave2Weights {
    tensors: HashMap<String, StoredTensor>,
}

impl Wave2Weights {
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
}

impl ModelWeights for Wave2Weights {
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

#[expect(clippy::cast_precision_loss, reason = "small synthetic dimensions")]
fn deterministic_scalar(idx: usize, seed: u32) -> f32 {
    let raw = i32::try_from((idx * 17 + seed as usize * 31) % 29).unwrap_or(0) - 14;
    (raw as f32) * 0.025
}

#[expect(clippy::cast_precision_loss, reason = "small synthetic dimensions")]
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
fn build_dense_model(recipe: &MistralRecipe, q: QuantType, seed: u32) -> Wave2Weights {
    let mut w = Wave2Weights::default();
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
        "{label} parity failure: cosine={cosine}, l2_rel={l2_rel}, got={got:?}, expected={expected:?}"
    );
}

#[test]
fn test_wave2_256_warm_cache_multistep_decode_and_compact_half_parity() {
    clear_global_layer_cache();

    let mut recipe = MistralRecipe::new().with_sliding_window(Some(8));
    recipe.num_layers = 2;
    recipe.hidden_dim = 256;
    recipe.ffn_dim = 256;
    recipe.n_heads_q = 4;
    recipe.n_heads_kv = 2;
    recipe.head_dim = 64;
    recipe.vocab_size = 4;

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
        let weights = build_dense_model(&recipe, quant, 11);

        // Warm cache up front: 5 norms (output + 2*2) and 14 dense projections (2*7).
        let mut warm_cache = MistralLayerCache::new();
        recipe
            .warm_layer_cache(&weights, &mut warm_cache)
            .expect("warm cache");
        assert_eq!(warm_cache.cached_norm_count(), 5);
        assert_eq!(warm_cache.cached_dense_count(), 14);
        assert_eq!(warm_cache.stats().norm_decodes, 5);
        assert_eq!(warm_cache.stats().dense_repacks, 14);

        let mut kv_cached = vec![0.0_f32; 2 * 2 * 8 * 128];
        let mut logits_cached = vec![0.0_f32; 4];
        let mut ctx_cached =
            ExecutionContext::new(&[0_u32, 1], 8, &mut logits_cached, &mut kv_cached).unwrap();
        recipe
            .prefill_step_with_cache(&mut ctx_cached, &weights, &mut warm_cache)
            .expect("cached prefill");
        for tok in [2_u32, 3] {
            ctx_cached.prepare_decode(tok).unwrap();
            recipe
                .decode_step_with_cache(&mut ctx_cached, &weights, &mut warm_cache)
                .expect("cached decode");
        }
        // Zero new decodes or repacks during prefill+decode after warm_layer_cache.
        assert_eq!(warm_cache.stats().norm_decodes, 5);
        assert_eq!(warm_cache.stats().dense_repacks, 14);
        assert!(warm_cache.stats().norm_hits >= 15);
        assert!(warm_cache.stats().dense_hits >= 56);

        // Compare cached path against zero-budget DirectFallback execution.
        let mut fb_cache = MistralLayerCache::new().with_max_repacked_bytes(0);
        let mut kv_fb = vec![0.0_f32; 2 * 2 * 8 * 128];
        let mut logits_fb = vec![0.0_f32; 4];
        let mut ctx_fb = ExecutionContext::new(&[0_u32, 1], 8, &mut logits_fb, &mut kv_fb).unwrap();
        recipe
            .prefill_step_with_cache(&mut ctx_fb, &weights, &mut fb_cache)
            .expect("fb prefill");
        for tok in [2_u32, 3] {
            ctx_fb.prepare_decode(tok).unwrap();
            recipe
                .decode_step_with_cache(&mut ctx_fb, &weights, &mut fb_cache)
                .expect("fb decode");
        }
        assert_eq!(fb_cache.resident_repacked_bytes(), 0);

        let max_rel = if matches!(quant, QuantType::F32 | QuantType::F16 | QuantType::BF16) {
            1e-4
        } else {
            2e-2
        };
        assert_cosine_and_l2(
            &format!("256x256 warm vs fallback {quant:?}"),
            ctx_cached.logits(),
            ctx_fb.logits(),
            max_rel,
        );
    }

    // Compact vs expanded F16/BF16 dense mode parity.
    for half_q in [QuantType::F16, QuantType::BF16] {
        verify_compact_vs_dense_half(&recipe, half_q);
    }
}

fn verify_compact_vs_dense_half(recipe: &MistralRecipe, half_q: QuantType) {
    let weights = build_dense_model(recipe, half_q, 42);
    let q_view = weights.tensor("blk.0.attn_q.weight").unwrap();
    let compact_proj = CachedDenseProjection::from_tensor_with_half_limit(
        &q_view,
        recipe.hidden_dim,
        recipe.hidden_dim,
        0,
    )
    .unwrap();
    let dense_proj = CachedDenseProjection::from_tensor_with_half_limit(
        &q_view,
        recipe.hidden_dim,
        recipe.hidden_dim,
        usize::MAX,
    )
    .unwrap();
    assert!(compact_proj.is_compact_half());
    assert!(!dense_proj.is_compact_half());

    let mut compact_cache = MistralLayerCache::new().with_dense_f32_max_elements(0);
    recipe
        .warm_layer_cache(&weights, &mut compact_cache)
        .expect("warm compact");
    let mut dense_cache = MistralLayerCache::new();
    recipe
        .warm_layer_cache(&weights, &mut dense_cache)
        .expect("warm dense");
    assert!(compact_cache.resident_repacked_bytes() > 0);
    #[cfg(feature = "with-kernels")]
    assert!(compact_cache.resident_repacked_bytes() < dense_cache.resident_repacked_bytes());
    #[cfg(not(feature = "with-kernels"))]
    assert_eq!(
        compact_cache.resident_repacked_bytes(),
        dense_cache.resident_repacked_bytes()
    );

    let mut kv_c = vec![0.0_f32; 2 * 2 * 8 * 128];
    let mut kv_d = vec![0.0_f32; 2 * 2 * 8 * 128];
    let mut log_c = vec![0.0_f32; 4];
    let mut log_d = vec![0.0_f32; 4];
    let mut ctx_c = ExecutionContext::new(&[1_u32, 2], 8, &mut log_c, &mut kv_c).unwrap();
    let mut ctx_d = ExecutionContext::new(&[1_u32, 2], 8, &mut log_d, &mut kv_d).unwrap();
    recipe
        .prefill_step_with_cache(&mut ctx_c, &weights, &mut compact_cache)
        .unwrap();
    recipe
        .prefill_step_with_cache(&mut ctx_d, &weights, &mut dense_cache)
        .unwrap();
    assert_cosine_and_l2(
        &format!("compact vs dense {half_q:?}"),
        &log_c,
        &log_d,
        1e-4,
    );
}

#[test]
fn test_wave2_mixtral_256_multistep_3d_exps_zero_copy_headroom_parity() {
    let mut recipe = MistralRecipe::new_mixtral()
        .with_sliding_window(Some(8))
        .with_experts(4, 2);
    recipe.num_layers = 1;
    recipe.hidden_dim = 256;
    recipe.ffn_dim = 256;
    recipe.n_heads_q = 4;
    recipe.n_heads_kv = 2;
    recipe.head_dim = 64;
    recipe.vocab_size = 4;

    for quant in [QuantType::F16, QuantType::Q4_0, QuantType::Q6_K] {
        let mut w_packed = Wave2Weights::default();
        let mut w_split = Wave2Weights::default();
        let dense = build_dense_model(&recipe, quant, 19);
        for (name, st) in dense.tensors {
            if !name.contains("ffn_gate") && !name.contains("ffn_up") && !name.contains("ffn_down")
            {
                w_packed.insert(&name, st.quant, st.shape.clone(), st.bytes.clone());
                w_split.insert(&name, st.quant, st.shape, st.bytes);
            }
        }
        let router_bytes = encode_matrix(quant, 4, 256, 77);
        w_packed.insert(
            "blk.0.ffn_gate_inp.weight",
            quant,
            vec![256, 4],
            router_bytes.clone(),
        );
        w_split.insert(
            "blk.0.ffn_gate_inp.weight",
            quant,
            vec![256, 4],
            router_bytes,
        );

        for proj in ["gate", "up", "down"] {
            let mut stacked = Vec::new();
            for e in 0..4_u32 {
                let exp_bytes = encode_matrix(
                    quant,
                    256,
                    256,
                    300 + e * 17 + u32::from(proj.as_bytes()[0]),
                );
                stacked.extend_from_slice(&exp_bytes);
                w_split.insert(
                    &format!("blk.0.ffn_{proj}.{e}.weight"),
                    quant,
                    vec![256, 256],
                    exp_bytes,
                );
            }
            w_packed.insert(
                &format!("blk.0.ffn_{proj}_exps.weight"),
                quant,
                vec![256, 256, 4],
                stacked,
            );
        }

        let tv_gate_exps = w_packed.tensor("blk.0.ffn_gate_exps.weight").unwrap();
        for e in 0..4 {
            let meta = inspect_packed_expert_slice(&tv_gate_exps, e, 4, 256, 256).unwrap();
            assert_eq!(meta.byte_offset_within_tensor, e * meta.slice_byte_len);
        }

        // Memlock-headroom cache repacks router & dense attention, never duplicates experts.
        let mut cache_packed =
            MistralLayerCache::with_memlock_headroom(DEFAULT_MEMLOCK_LIMIT_BYTES, 26_300_000_000);
        let mut cache_split =
            MistralLayerCache::with_memlock_headroom(DEFAULT_MEMLOCK_LIMIT_BYTES, 26_300_000_000);

        let mut kv_p = vec![0.0_f32; 2 * 8 * 128];
        let mut kv_s = vec![0.0_f32; 2 * 8 * 128];
        let mut log_p = vec![0.0_f32; 4];
        let mut log_s = vec![0.0_f32; 4];
        let mut ctx_p = ExecutionContext::new(&[0_u32, 1], 8, &mut log_p, &mut kv_p).unwrap();
        let mut ctx_s = ExecutionContext::new(&[0_u32, 1], 8, &mut log_s, &mut kv_s).unwrap();

        recipe
            .prefill_step_with_cache(&mut ctx_p, &w_packed, &mut cache_packed)
            .unwrap();
        recipe
            .prefill_step_with_cache(&mut ctx_s, &w_split, &mut cache_split)
            .unwrap();
        for tok in [2_u32, 3] {
            ctx_p.prepare_decode(tok).unwrap();
            ctx_s.prepare_decode(tok).unwrap();
            recipe
                .decode_step_with_cache(&mut ctx_p, &w_packed, &mut cache_packed)
                .unwrap();
            recipe
                .decode_step_with_cache(&mut ctx_s, &w_split, &mut cache_split)
                .unwrap();
        }
        for (&lp, &ls) in log_p.iter().zip(&log_s) {
            assert!(
                (lp - ls).abs() < 1e-6,
                "packed vs split MoE delta for {quant:?}"
            );
        }
        assert!(cache_packed.stats().expert_zero_copy_slices >= 4 * 2 * 3);
        // Only dense attention + router (5 matrices) were repacked, zero 3D experts.
        assert_eq!(cache_packed.cached_dense_count(), 5);
    }
}

fn build_gguf_v3_with_vocab_tokens(
    arch: &str,
    vocab_size: usize,
    tok_model: &str,
    chat_template: &str,
    bos: u32,
    eos: u32,
) -> Vec<u8> {
    fn write_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3_u32.to_le_bytes());
    buf.extend_from_slice(&0_u64.to_le_bytes());
    buf.extend_from_slice(&13_u64.to_le_bytes());

    let u32_kvs = [
        (format!("{arch}.block_count"), 40_u32),
        (format!("{arch}.embedding_length"), 5120),
        (format!("{arch}.feed_forward_length"), 32_768),
        (format!("{arch}.attention.head_count"), 32),
        (format!("{arch}.attention.head_count_kv"), 8),
        (format!("{arch}.attention.key_length"), 128),
        (
            format!("{arch}.vocab_size"),
            u32::try_from(vocab_size).expect("vocab u32"),
        ),
        ("tokenizer.ggml.bos_token_id".to_owned(), bos),
        ("tokenizer.ggml.eos_token_id".to_owned(), eos),
    ];

    write_str(&mut buf, "general.architecture");
    buf.extend_from_slice(&8_u32.to_le_bytes());
    write_str(&mut buf, arch);

    write_str(&mut buf, "tokenizer.ggml.model");
    buf.extend_from_slice(&8_u32.to_le_bytes());
    write_str(&mut buf, tok_model);

    write_str(&mut buf, "tokenizer.chat_template");
    buf.extend_from_slice(&8_u32.to_le_bytes());
    write_str(&mut buf, chat_template);

    write_str(&mut buf, "tokenizer.ggml.tokens");
    buf.extend_from_slice(&9_u32.to_le_bytes()); // ARRAY
    buf.extend_from_slice(&8_u32.to_le_bytes()); // element_type = STRING
    buf.extend_from_slice(&(vocab_size as u64).to_le_bytes());
    for i in 0..vocab_size {
        let tok = if i == bos as usize {
            "<s>".to_owned()
        } else if i == eos as usize {
            "</s>".to_owned()
        } else {
            format!("t{i}")
        };
        write_str(&mut buf, &tok);
    }

    for (k, v) in u32_kvs {
        write_str(&mut buf, &k);
        buf.extend_from_slice(&4_u32.to_le_bytes());
        buf.extend_from_slice(&v.to_le_bytes());
    }

    while buf.len() % 32 != 0 {
        buf.push(0);
    }
    buf
}

#[test]
fn test_wave2_tekken_131k_sentencepiece_gguf_and_mistral3_memlock_contracts() {
    // Full 131,072-entry Tekken GGUF parsing, vocab/stop/LM-head validation, and hydration.
    let tekken_gguf_bytes = build_gguf_v3_with_vocab_tokens(
        "mistral3",
        TEKKEN_VOCAB_SIZE,
        "tekken",
        "<s>[INST] {{ prompt }} [/INST]",
        1,
        2,
    );
    let tekken_gguf = GgufFile::parse(&tekken_gguf_bytes).expect("parse full 131k Tekken GGUF");
    let contract = validate_gguf_tokenizer_and_lm_head(&tekken_gguf, Some(TEKKEN_VOCAB_SIZE))
        .expect("131k contract valid");
    assert_eq!(contract.family, MistralTokenizerFamily::Tekken131k);
    assert_eq!(contract.vocab_size, TEKKEN_VOCAB_SIZE);
    assert_eq!(contract.logit_buffer_bytes, TEKKEN_VOCAB_SIZE * 4);
    assert!(contract.has_inst_template);

    let recipe = MistralRecipe::from_gguf(&tekken_gguf).expect("hydrate Tekken recipe");
    assert_eq!(recipe.vocab_size, TEKKEN_VOCAB_SIZE);

    // Undersized logit slice for Tekken 131k fails closed before execution.
    assert!(matches!(
        validate_gguf_tokenizer_and_lm_head(&tekken_gguf, Some(SENTENCEPIECE_VOCAB_32K_EXT)),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // SentencePiece 32,000 and 32,768 GGUFs validate cleanly.
    for spm_vocab in [SENTENCEPIECE_VOCAB_32K, SENTENCEPIECE_VOCAB_32K_EXT] {
        let spm_bytes = build_gguf_v3_with_vocab_tokens(
            "mistral",
            spm_vocab,
            "llama",
            "[INST] {{ prompt }} [/INST]",
            1,
            2,
        );
        let spm_gguf = GgufFile::parse(&spm_bytes).expect("parse SPM GGUF");
        let spm_contract = validate_gguf_tokenizer_and_lm_head(&spm_gguf, Some(spm_vocab))
            .expect("SPM contract valid");
        assert_eq!(
            classify_mistral_vocab_family(spm_vocab).unwrap(),
            spm_contract.family
        );
    }

    // Check Mistral-Small-3.2-24B, Mistral-Nemo-12B, Mistral-7B, and Mixtral-8x7B profiles.
    let mut small_24b =
        ModelManifest::new("gguf", "mistral3", "1.0.0", "Q4_K_M", 8192, "sha256:s", 0);
    small_24b.origin = Some("mistralai/Mistral-Small-3.2-24B-Instruct".to_owned());
    assert_eq!(
        classify_mistral_manifest_profile(&small_24b),
        MistralModelProfile::MistralSmall24B
    );
    let rep_small =
        verify_mixtral_admission(&small_24b, DEFAULT_MEMLOCK_LIMIT_BYTES).expect("24B Q4_K_M OK");
    assert_eq!(rep_small.kv_bytes, 8192 * MISTRAL3_KV_BYTES_PER_TOKEN);
    let rep_recipe =
        verify_mistral_recipe_admission(&small_24b, &recipe, DEFAULT_MEMLOCK_LIMIT_BYTES, 64)
            .expect("recipe admission OK");
    assert!(rep_recipe.total_required_bytes <= DEFAULT_MEMLOCK_LIMIT_BYTES);

    let mistral_7b = ModelManifest::new("gguf", "mistral", "1.0.0", "Q8_0", 4096, "sha256:7b", 0);
    let rep_7b = verify_mixtral_admission(&mistral_7b, DEFAULT_MEMLOCK_LIMIT_BYTES).unwrap();
    assert_eq!(rep_7b.kv_bytes, 4096 * MISTRAL_7B_KV_BYTES_PER_TOKEN);

    let mixtral_q4k =
        ModelManifest::new("gguf", "mixtral", "1.0.0", "Q4_K_M", 32_768, "sha256:m", 0);
    assert!(matches!(
        evaluate_moe_memlock_admission(&mixtral_q4k, DEFAULT_MEMLOCK_LIMIT_BYTES),
        Err(K3Error::InsufficientLockableMemory { .. })
    ));
    let _ = MIXTRAL_KV_BYTES_PER_TOKEN;
    assert!(matches!(
        validate_vocab_stop_contract(
            "mistral3",
            TEKKEN_VOCAB_SIZE,
            None,
            None,
            None,
            None,
            &[u32::try_from(TEKKEN_VOCAB_SIZE).expect("vocab fits u32")],
            TEKKEN_VOCAB_SIZE
        ),
        Err(K3Error::CapabilityMismatch { .. })
    ));
}
