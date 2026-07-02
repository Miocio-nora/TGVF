# TGVF 50K Source and Question Examples

This document gives compact examples of the image sources and question types in
the current Stage2 open-answer 50K training split.

Source split:

```text
data/tgvf_teacher/generated/runs/tgvf_v4_teacher_50k_clean_imend_open_answer/splits/tgvf_v4_teacher_stage2_protocol_c.train.jsonl
```

## Dataset Mix

Image source distribution:

| Source | Rows | Source profile |
|---|---:|---|
| `visual_genome` | 19,625 | `natural_image` |
| `docvqa` | 9,150 | `document` |
| `textvqa` | 7,240 | `scene_text` |
| `textocr` | 6,652 | `scene_text` |
| `chartqa` | 4,216 | `chart` |

Major question types:

| Question type | Rows |
|---|---:|
| `color_attribute` | 6,552 |
| `spatial_relation` | 5,750 |
| `whole_image_obvious` | 5,620 |
| `chart_table` | 4,615 |
| `ocr_text` | 4,584 |
| `object_part` | 4,259 |
| `counting` | 3,441 |
| `texture_material` | 3,243 |
| `shape_boundary` | 2,787 |
| `pattern` | 1,772 |
| `state_action` | 1,642 |
| `math_reasoning` | 1,435 |
| `number_reading` | 1,141 |

## Source Examples

### Visual Genome: Natural Image

Image id: `visual_genome:2363183`

![Visual Genome example](assets/tgvf_50k_data_examples/single_visual_genome_2363183.jpg)

| Question type | Focus? | Question | Target when focused | Answer |
|---|---|---|---|---|
| `color_attribute` | yes | What color is the bird's beak? | close-up of the bird's head, beak surface, and surrounding face feathers | orange |
| `spatial_relation` | yes | Where is the bird positioned relative to the surrounding reeds? | wide shared view containing the bird, nearby reeds, and the space around its perch | perched among them |
| `shape_boundary` | yes | What is the general shape of the bird's tail? | outer contour and full boundary of the bird's tail against the reeds | long and narrow |
| `whole_image_obvious` | no | What kind of animal is shown in the image? | none | a bird |

### TextOCR: Scene Text and Object Detail

Image id: `textocr:849b83e927a81ef6`

![TextOCR example](assets/tgvf_50k_data_examples/single_textocr_849b83e927a81ef6.jpg)

| Question type | Focus? | Question | Target when focused | Answer |
|---|---|---|---|---|
| `color_attribute` | yes | What color is the main body of the remote control? | broad front casing surface of the remote around the buttons and lower edge | silver |
| `pattern` | yes | What kind of pattern is visible on the chair fabric beneath the remote? | chair upholstery surface showing repeated stitched lines and small floral motifs | diagonal diamond grid with small motifs |
| `object_part` | yes | Which part of the remote is colored red? | upper end of the remote with colored buttons and surrounding top edge | a top corner button |
| `ocr_text` | yes | What device is named in the English text on the label near the bottom of the remote? | high-resolution black label near the remote bottom with the English text lines | TV set |
| `spatial_relation` | no | Where is the remote placed relative to the chair seat? | none | On the chair seat |

### TextVQA: Scene Text and Label Appearance

Image id: `textvqa:62b7ef0cb4a96f93`

![TextVQA example](assets/tgvf_50k_data_examples/source_textvqa_62b7ef0cb4a96f93.jpg)

| Question type | Focus? | Question | Target when focused | Answer |
|---|---|---|---|---|
| `color_attribute` | yes | What color are the large letters spelling the beer name in the middle of the label? | large central title letters and their colored fill against the dark label | gold |
| `shape_boundary` | yes | What is the overall shape of the main label on the bottle? | outer contour and full border of the main bottle label | shield-like with curved sides |
| `pattern` | yes | What kind of lettering style is used for the brand name near the top, 'The Bruery'? | top brand name letterforms with loops, strokes, and connected script styling | flowing cursive script |
| `ocr_text` | yes | What word appears below the brand name near the top center of the label? | small centered printed word below the script brand name near the top | UNFILTERED |
| `whole_image_obvious` | no | What object is shown in the image? | none | a beer bottle |

### DocVQA: Document Layout and Reading

Image id: `docvqa:ktdw0079_1`

![DocVQA organization chart](assets/tgvf_50k_data_examples/single_docvqa_ktdw0079_1.png)

| Question type | Focus? | Question | Target when focused | Answer |
|---|---|---|---|---|
| `chart_table` | yes | Who is listed as the manager in the organizational chart? | upper central boxed name and italic job title beneath it | Dr. D. J. Doolittle |
| `chart_table` | yes | What job title is shown for C. W. Butner? | upper right name box with the italic role line below the name | Administrative Secretary |
| `counting` | yes | How many people are listed under Dr. C. K. Lee's branch? | entire left subordinate list below Dr. C. K. Lee with all connected names | 4 |
| `spatial_relation` | yes | Which person is directly below Dr. D. W. Bombick in the chart? | right branch box for Dr. D. W. Bombick and first connected name below it | C. McKarns |
| `whole_image_obvious` | no | What is the main title of this document? | none | Cellular/Molecular Biology |

### ChartQA: Chart Values and Arithmetic

Image id: `chartqa:train_004281`

![ChartQA bar chart](assets/tgvf_50k_data_examples/multi_chartqa_train_004281.png)

| Question type | Focus? | Question | Target when focused | Answer |
|---|---|---|---|---|
| `chart_table` | yes | Which category has the tallest bar in the chart? | the tallest blue bar with its category label beneath and nearby value label above | Direct tourism contribution |
| `chart_table` | yes | Which category is the only one shown with a negative value? | the bar crossing below the zero line with its category label and value annotation | Imported goods from indirect spending |
| `number_reading` | yes | What value is shown above the Domestic supply chain bar? | the Domestic supply chain bar with the numeric label directly above its top | 278.23 |
| `math_reasoning` | yes, multi-focus | Which category is closest in value to 100 trillion Iranian rial? | mid-to-lower bars around the 100 mark, then the bar labeled 93.47 | Capital investment |
| `math_reasoning` | no | How much larger is Direct tourism contribution than Induced impact? | none | 174.35 |

## Notes

- `Focus? yes` means the row uses `need_focus=true` and has a local visual
  target.
- `Focus? no` means the row is a direct-answer example with
  `sufficient_visual_evidence`.
- Multi-focus rows contain two focus steps in the current training path; the
  table condenses those targets into one readable cell.

