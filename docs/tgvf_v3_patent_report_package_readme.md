# TGVF Patent-Oriented Report Package

## Purpose

This package collects the current English technical disclosure materials for Target-Guided Visual Foveation (TGVF). It is intended for patent-counsel review, internal research review, and engineering handoff.

The package is a technical disclosure package, not a finalized patent application and not a legal claim set.

## Recommended minimal review path

For a first-pass review, read only:

1. `docs/tgvf_v3_patent_executive_summary_en.md`
2. `docs/tgvf_v3_patent_technical_report_en.tex`
3. `docs/tgvf_v3_patent_report_package_readme.md`

The build notes are optional support material for producing the PDF. They do not need to be read in the first pass.

## Core documents

| File | Purpose | Primary reader |
|---|---|---|
| `docs/tgvf_v3_patent_technical_report_en.tex` | Full patent-oriented technical disclosure and engineering report. | Patent counsel, technical reviewers, engineering maintainers |
| `docs/tgvf_v3_patent_executive_summary_en.md` | Concise executive summary and decision brief. | Counsel, PI, project leads |
| `docs/tgvf_v3_patent_report_package_readme.md` | Package manifest and handoff guide. | Anyone receiving the report package |

## Optional support documents

This file is useful for PDF production, but it is not required for a first technical read.

| File | Purpose | Primary reader |
|---|---|---|
| `docs/tgvf_v3_patent_report_build_notes.md` | Build commands and PDF production-check guidance. | Engineering maintainers |

## Full report scope

The full LaTeX report is intentionally comprehensive, but its content falls into five main groups:

- invention overview: problem, solution, key contributions, technical effects, and preferred embodiment;
- technical implementation: protocol, mathematical formulation, renderer, training stages, inference controller, interfaces, and audit logs;
- evidence: metrics, controls, mechanism proof chain, experimental summaries, ablations, and interpretation boundaries;
- patent support: novelty rationale, claim skeletons, claim scope ladder, prosecution notes, figure specifications, and non-limiting variations;
- handoff material: reproducibility requirements, open filing items, build notes, and recommended appendices.

## Current preferred technical position

The current preferred embodiment is:

- TGVF-v3 method;
- Protocol C tool-observation runtime;
- V4 trace-style teacher data;
- Stage1 bidirectional TGVF visual evidence training;
- Stage2 LoRA plus trainable TGVF trajectory training;
- focus-imend action termination before tool insertion;
- correct-D, no-D, random-D, wrong-same-image-D, and wrong-different-image-D controls.

The broader invention should not be limited to this exact embodiment. The report explicitly preserves alternative implementations including encoder-reencode TGVF, native visual-token insertion, temporal descriptor reuse, and equivalent focus/evidence boundary mechanisms.

## Main invention thesis

TGVF is a reasoning-time active visual evidence mechanism for VLMs:

1. the model emits a visible, sample-specific visual focus descriptor;
2. descriptor hidden states, excluding boundary markers, become the foveation query;
3. the foveation query conditions image-derived visual features;
4. the system generates or re-encodes target-specific visual evidence tokens `D`;
5. `D` is appended as a new visual/tool observation;
6. the model continues reasoning and answers from the updated context.

## Key differentiation points

TGVF should be distinguished from:

- crop/zoom methods, because the core mechanism is hidden-state-conditioned visual evidence generation or re-encoding;
- VPT/soft-prompt methods, because the control signal is visible, generated, sample-specific, and auditable;
- prompt engineering, because correct-D is compared against no-D, random-D, and wrong-D controls;
- ordinary tool use, because the tool input is a descriptor-hidden-state query that conditions visual evidence;
- second-pass VQA, because the preferred runtime preserves the LLM cache and avoids a second full language-prefix forward.

## Evidence package to attach later

For formal review or filing, attach:

1. compiled PDF of the full LaTeX report;
2. rendered schematic figures;
3. representative raw traces for direct, force, free, correct-D, no-D, random-D, wrong-D;
4. Stage1 and Stage2 configuration files;
5. checkpoint identifiers and tokenizer versions;
6. V4 schema and accepted/rejected teacher-trace examples;
7. readout, query-sensitivity, FVT distribution, force, teacher-forced, and free-router evaluation reports;
8. parser/controller source excerpts if counsel requests implementation detail.

## Remaining gates

The main remaining gate is production cleanup:

- compile the LaTeX source;
- fix syntax errors if any;
- inspect PDF layout;
- fix table widths and float placement;
- resolve overfull boxes;
- optionally split long tables into appendix format;
- render final patent-style figures from the included TikZ/specification drafts.

Until this gate is complete, the content is a strong technical draft but not a production-ready PDF package.
