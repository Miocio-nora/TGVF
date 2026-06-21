# TGVF Patent Report Build Notes

## Purpose

These notes describe how to compile and production-check the English TGVF patent-oriented technical report.

This file is a build guide only. It does not replace the full report or the executive summary.

## Source files

Primary source:

```text
docs/tgvf_v3_patent_technical_report_en.tex
```

Companion documents:

```text
docs/tgvf_v3_patent_executive_summary_en.md
docs/tgvf_v3_patent_report_package_readme.md
```

Expected PDF output:

```text
docs/tgvf_v3_patent_technical_report_en.pdf
```

## Recommended compile command

From the repository root:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=docs \
  docs/tgvf_v3_patent_technical_report_en.tex
```

If `latexmk` is unavailable, use two passes of `pdflatex` from the repository root:

```bash
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=docs \
  docs/tgvf_v3_patent_technical_report_en.tex

pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=docs \
  docs/tgvf_v3_patent_technical_report_en.tex
```

## Expected dependencies

The report uses standard LaTeX packages:

- `geometry`
- `booktabs`
- `array`
- `enumitem`
- `hyperref`
- `xcolor`
- `graphicx`
- `amsmath`
- `amssymb`
- `microtype`
- `tikz`

TikZ libraries used:

- `arrows.meta`
- `positioning`
- `fit`
- `shapes.geometric`

A reasonably complete TeX Live installation should be sufficient.

## Production checks after compile

After generating the PDF, inspect the following manually:

1. Title and abstract fit cleanly on the first page.
2. Two-column flow is readable and not overly dense.
3. Large tables do not overflow page margins.
4. TikZ figures fit within page width.
5. `figure*` and `table*` floats appear near their first references.
6. Verbatim algorithm blocks do not overflow or collide with floats.
7. Hyperlinks and table/figure references resolve.
8. No visible placeholder markers remain unintentionally.
9. Executive and patent-facing sections appear before deep implementation detail.
10. Conclusion and open-items sections appear near the end.

## Common fixes if compile/layout issues appear

If a table is too wide:

- reduce `\small` to `\scriptsize` for that table;
- shorten column text;
- convert `table` to `table*`;
- split the table into two smaller tables;
- move long implementation tables into an appendix.

If a TikZ figure is too wide:

- reduce node width or spacing;
- wrap the figure in `\resizebox{\textwidth}{!}{...}` for `figure*`;
- split the figure into two diagrams.

If floats move too far:

- move the source block earlier;
- convert some long tables to appendix-style sections;
- reduce the number of consecutive `table*` / `figure*` floats.

If overfull boxes appear in monospace strings:

- shorten file paths;
- replace long paths with representative artifact labels;
- use `\scriptsize` in the affected table;
- move exact paths to a Markdown appendix instead of the LaTeX body.

## Suggested final package after successful compile

A production-ready handoff should contain:

```text
docs/tgvf_v3_patent_technical_report_en.pdf
docs/tgvf_v3_patent_technical_report_en.tex
docs/tgvf_v3_patent_executive_summary_en.md
docs/tgvf_v3_patent_report_package_readme.md
docs/tgvf_v3_patent_report_build_notes.md
```

Optional appendices:

```text
representative_traces.jsonl
stage1_stage2_config_bundle/
evaluation_report_bundle/
figure_source_bundle/
```

## Current status

As of this note, the LaTeX source has not yet been compiled in this workflow. The content is a strong technical draft, but PDF syntax/layout status remains unverified until the compile gate is run.
