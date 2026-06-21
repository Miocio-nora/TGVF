# TGVF Patent-Oriented Executive Summary

## Purpose

This is the short decision brief for the full technical disclosure:

`docs/tgvf_v3_patent_technical_report_en.tex`

Use this file for first-pass review. Use the LaTeX report as the detailed technical record.

## Invention in one sentence

Target-Guided Visual Foveation (TGVF) lets a vision-language model generate a sample-specific visual focus descriptor during reasoning, captures the descriptor hidden states, uses them to condition image-derived visual evidence generation or re-encoding, appends the resulting visual evidence tokens, and resumes answering from the same reasoning trajectory.

## Core mechanism

1. Image and task are processed by a VLM.
2. The model emits either a direct answer or a bounded focus descriptor.
3. Descriptor-token hidden states, excluding boundary tokens, form `H_q`.
4. A TGVF module combines `H_q` with visual features to produce target-conditioned evidence tokens `D`.
5. `D` is appended as a new visual/tool observation.
6. The model continues evidence readout and final answer generation.

## Current preferred embodiment

| Area | Current position |
|---|---|
| Backbone | Qwen3-VL-Thinking style local VLM runtime |
| Protocol | Protocol C tool-observation format |
| Data | V4 trace-style teacher data |
| Stage1 | Bidirectional TGVF training for readable, target-specific `D` |
| Stage2 | LoRA + trainable TGVF trajectory training |
| Controls | correct-D, no-D, random-D, wrong-same-image-D, wrong-different-image-D |

## Key differentiation

- Not crop/zoom: the core operation is descriptor-hidden-state-conditioned visual evidence generation or re-encoding.
- Not VPT only: the control signal is visible, generated at inference, sample-specific, and auditable.
- Not prompt engineering: correct-D is evaluated against no-D, random-D, and wrong-D controls.
- Not ordinary tool use: the tool input is a descriptor-hidden-state query that conditions image-derived visual evidence.
- Not second-pass VQA: the preferred runtime appends evidence while preserving the language-model cache.

## Evidence pattern

The current prototype supports the mechanism through a chain of evidence: valid focus actions, marker-excluded descriptor hidden-state capture, readable `D`, correct-D outperforming no-D/random-D/wrong-D controls, and working focus/tool/answer trajectories. Benchmark claims should still be separated from mechanism evidence until official evaluation code and settings are used.

## Claim themes to review

1. Model-generated focus descriptor controlling visual evidence generation.
2. Descriptor hidden states as a foveation query.
3. Target-conditioned visual evidence tokens derived from image features.
4. Append-only evidence insertion without replacing original image tokens.
5. KV-preserving continuation after TGVF evidence insertion.
6. Stage1/Stage2 training for evidence quality and trajectory behavior.
7. V4 traces for no-refocus, single-refocus, and multi-refocus behavior.

## Immediate next steps

1. Compile the LaTeX report and fix layout issues.
2. Render the core figures: system loop, focus capture, bidirectional module, encoder-reencode, and KV-preserving runtime.
3. Attach representative traces and Stage1/Stage2 configuration/evaluation bundles.
4. Ask patent counsel to decide claim-family split and convert the technical draft language into formal claims.
