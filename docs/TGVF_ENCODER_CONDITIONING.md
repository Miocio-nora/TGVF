# TGVF Encoder-Inside Conditioning Notes

This note records candidate designs for moving TGVF target conditioning from the
current post-visual-encoder path into the Qwen2-VL visual encoder path.

## Current State

The current TGVF v2 variants are post-encoder visual re-encoding modules:

```text
image
-> Qwen2-VL visual encoder
-> pre-merge visual tokens
-> target-conditioned TGVF module
-> grouped visual merger
-> FVT/readout
```

This is close to the VisualPerceptionToken project's projector-level
conditioning: a hidden/action token is used as a condition vector for visual
features after a visual encoder has produced features. It is not the same as
classic visual prompt tuning where prompt tokens are inserted into vision
self-attention layers.

The current `tgvf_v2_vpt_gating` variant uses one global target condition:

```text
target tokens [T, d_lm]
-> mean pool over T
-> linear projection to d_v
-> per-vision-token dot score
-> residual update on pre-merge visual tokens
```

So `tgvf_v2_vpt_gating` is a simple global-condition baseline. The
token-level conditioning variants are closer to `tgvf_v2_cross_attention` and
`tgvf_v2_bidirectional`.

## Why Move Conditioning Into the Encoder

Target-conditioned visual encoding could let the target affect how image tokens
are refined before the final visual representation is produced. This may be
useful when the target should change local feature extraction rather than only
post-process already-frozen visual features.

The main cost risk is that Qwen2-VL-2B's visual encoder is not cheap. For common
image sizes, a full visual encoder pass can be comparable to or larger than an
LLM prefill pass. A design that reruns all visual layers for every target can be
very expensive for same-image multi-target batches.

## Candidate A: Cross-Attention Adapter Inside Vision Blocks

Insert a target-to-vision adapter after selected visual blocks:

```text
vision tokens
-> original visual self-attention / MLP block
-> cross-attn(query=vision tokens, key/value=target tokens)
-> residual update
-> next visual block
```

Properties:

- Keeps image token count unchanged.
- Keeps image token order unchanged.
- Keeps real image grid and image mRoPE semantics.
- Does not require target tokens to have fake image positions.
- Lets each visual token attend to the full target-token sequence.

This is the recommended first encoder-inside design because it gives true
token-level conditioning without changing the visual token layout.

Implementation sketch:

```python
vision_context = cross_attn(
    query=vision_tokens,
    key=target_tokens_projected,
    value=target_tokens_projected,
)
vision_tokens = vision_tokens + gate * out_proj(vision_context)
```

Recommended details:

- Add adapters only to the last few visual blocks first.
- Initialize residual gate near zero.
- Keep the original visual encoder weights frozen initially.
- Train only adapter, projections, and TGVF/readout heads.

## Candidate B: Prefix K/V Prompt for Vision Self-Attention

Do not concatenate condition tokens to the vision token sequence. Instead, add
target-derived prefix keys and values inside vision self-attention:

```text
Q = image tokens
K,V = concat(target-derived prefix K/V, image K/V)
output = image-token outputs only
```

Properties:

- Keeps image token count unchanged.
- Keeps output layout compatible with real image grid.
- Avoids assigning image positions to condition tokens.
- More similar to prompt tuning than Candidate A, but less invasive than
actually inserting prompt tokens into the token sequence.

Tradeoffs:

- Requires modifying attention internals or wrapping visual attention modules.
- Needs careful handling of rotary embeddings: prefix K/V should either bypass
  image rotary positions or use a separate position treatment.
- Harder to implement cleanly than Candidate A in stock Qwen2-VL code.

## Candidate C: Shallow or Deep Visual Prompt Tokens

Construct target-derived prompt tokens and concatenate them with image tokens
inside the visual encoder:

```text
[condition prompt tokens; image tokens]
-> visual encoder block(s)
-> discard condition prompt tokens
-> keep image tokens
-> merger/readout
```

Variants:

- Shallow prompt: insert only at the visual encoder input.
- Deep prompt: insert fresh prompt tokens at multiple layers.

Properties:

- Closest to classic visual prompt tuning.
- Lets image tokens and condition tokens interact through self-attention.

Risks:

- Qwen2-VL visual tokens use image-grid rotary positions.
- Condition tokens do not naturally have image-grid positions.
- Assigning fake image positions to condition tokens may harm geometry.
- Attention masks and token slicing must ensure prompt tokens are not passed to
  the visual merger as image tokens.

This is expressive but should not be the first implementation unless the
position treatment is designed carefully.

## Candidate D: FiLM or Gated Modulation Inside Vision Blocks

Generate per-layer scale/shift/gate values from target tokens and apply them to
vision hidden states:

```text
gamma, beta = f(target tokens)
vision_tokens = vision_tokens * (1 + gamma) + beta
```

Properties:

- Simple and memory efficient.
- Keeps image token count, order, and grid unchanged.
- Easy to initialize as near-identity.

Tradeoffs:

- Weaker than token-level cross-attention.
- A global target summary can repeat the current `vpt_gating` limitation.
- Token-level FiLM is possible but more complex.

This is a good low-risk ablation, but it may underperform cross-attention
adapters when the target requires local, token-specific visual selection.

## Top-K Selection With Real Grid

Top-k can be combined with encoder-inside conditioning in two different ways.

Dense-grid top-k gating:

```text
compute relevance score per image grid group
select top-k groups
apply target-conditioned update only to selected groups
keep all image tokens
```

This preserves real grid and real position ids because no image tokens are
removed. It does not reduce LLM/readout token count.

Sparse-position top-k:

```text
select top-k image grid groups
append only selected tokens
use their original sparse image mRoPE positions
```

This can reduce token count, but it no longer uses a dense source grid span. It
requires custom sparse image position ids and must be validated empirically,
because the base model usually sees dense image grids.

## Recommended Roadmap

1. Implement late-block cross-attention adapters inside the visual encoder.
2. Keep visual token count unchanged and continue using real source grid
   metadata.
3. Start with the last 2-4 visual blocks only.
4. Initialize adapter residual gates near zero.
5. Compare against current post-encoder v2 under the same readout losses.
6. Add dense-grid top-k gating as an ablation after the adapter path is stable.
7. Consider sparse-position top-k only after dense-grid conditioning shows a
   clear benefit.

## Open Questions

- Which visual layers should receive conditioning?
- Should target tokens come from the original question prompt, generated target
  tokens, or a separate target encoder?
- Should the visual encoder be fully frozen, partially unfrozen, or adapter-only?
- Does dense-grid top-k improve target specificity without reducing readout
  cost?
- Can sparse-position top-k work without hurting Qwen2-VL's image-position
  assumptions?
