---
name: jero-arch-mistral
description: "Implement and validate Mistral and Mixtral ArchRecipe execution, Sliding Window Attention (SWA), and Sparse Mixture-of-Experts (MoE) top-k routing in jero-arch-mistral."
---

# `jero-arch-mistral` Sliding Window Attention & Sparse MoE Invariants

- Implement `jero_core::ArchRecipe` supporting dense `mistral`/`mistral3` (Sliding Window Attention `sliding_window` and decoupled GQA head dimensions) and `mixtral` Sparse MoE (`expert_count`, `expert_used_count` top-k softmax gating over expert FFN weights).
- Execute batched `[M, K]` dense prefill (`Q`/`K`/`V`/`O` and dense `SwiGLU`) via `gemm_repacked_rows`, batched `Norm` RoPE across `start_position..end_position`, causal + `SWA` `gqa_causal_attention_prefill_f32`, and final-row-only (`M - 1`) LM-head projection gated on `ExecutionContext::compute_logits()`.
- Execute `mixtral` prefill via token-bucketed active-expert `[M_e, hidden_dim]` GEMMs (`repack_transient_projection` within memlock headroom or zero-copy fallback), loading each active expert at most once per layer batch without mutating resident expert cache bytes.
- Cache `128-byte`-aligned dense projections and once-decoded `f32` norm vectors in `MistralLayerCache` with content-verified anti-ABA invalidation and bounded `RLIMIT_MEMLOCK` headroom, while slicing `3D` `_exps` `[n_embd, ffn_dim, n_expert]` tensors zero-copy per active expert without duplicating expert RAM.
- Validate `SentencePiece` (`32,000` / `32,768`) and `Tekken` (`131,072`) tokenizer metadata, `[INST]` chat framing, special/stop token bounds, LM-head logit buffer sizing, and `weights + KV + scratch <= RLIMIT_MEMLOCK` admission before page locking.
- Preserve deterministic expert selection and delegate to `jero-kernels-rvv` when the `with-kernels` feature is enabled.
