---
name: jero-arch-mistral
description: "Implement and validate Mistral and Mixtral ArchRecipe execution, Sliding Window Attention (SWA), and Sparse Mixture-of-Experts (MoE) top-k routing in jero-arch-mistral."
---

# `jero-arch-mistral` Sliding Window Attention & Sparse MoE Invariants

- Implement `jero_core::ArchRecipe` supporting dense `mistral`/`mistral3` (Sliding Window Attention `sliding_window` and decoupled GQA head dimensions) and `mixtral` Sparse MoE (`expert_count`, `expert_used_count` top-k softmax gating over expert FFN weights).
- Cache `128-byte`-aligned dense projections and once-decoded `f32` norm vectors in `MistralLayerCache` with content-verified anti-ABA invalidation and bounded `RLIMIT_MEMLOCK` headroom, while slicing `3D` `_exps` `[n_embd, ffn_dim, n_expert]` tensors zero-copy per active expert without duplicating expert RAM.
- Validate `SentencePiece` (`32,000` / `32,768`) and `Tekken` (`131,072`) tokenizer metadata, `[INST]` chat framing, special/stop token bounds, LM-head logit buffer sizing, and `weights + KV + scratch <= RLIMIT_MEMLOCK` admission before page locking.
- Preserve deterministic expert selection and delegate to `jero-kernels-rvv` when the `with-kernels` feature is enabled.
