# TGVF Teacher Data Generation and FVT Training

This document describes the current data-generation and training flow for TGVF.

TGVF here means Target-Guided Visual Foveation. The current training target is the newly added FVT module, not Qwen2-VL itself.

## Current Scope

Implemented:

- teacher image-pool preparation
- fixed source mix selection for teacher generation
- OpenAI GPT vision teacher-guide data generation
- strict JSON schema validation and local filtering
- append-only ledger and resumable generation
- W&B logging for data generation
- FVT module training with frozen Qwen2-VL
- W&B logging for training

Out of scope in this code path:

- teacher-guide prompt generation from another model family
- Qwen2-VL finetuning or LoRA
- visual encoder training
- crop/OCR pipeline
- benchmark evaluation

## Raw Dataset Root

Raw external datasets live outside the project:

```text
/home/dredvpn009/Flash_Storage/datasets
```

Current required sources:

```text
visual_genome
textvqa
textocr
docvqa
chartqa
```

Project-generated manifests and teacher data live inside the repository:

```text
data/tgvf_teacher/
```

Do not write generated teacher JSONL under `/home/dredvpn009/Flash_Storage/datasets`.

## Source Mix Requirement

All teacher-generation selections should keep this source mix:

```text
Visual Genome      40%
TextVQA + TextOCR  30%
DocVQA             20%
ChartQA            10%
```

The current 20k teacher run selection is:

```text
data/tgvf_teacher/preparation/selections/selected_images_20k_samples_v0.jsonl
```

It contains 7000 selected images:

```text
visual_genome:      2800
textvqa + textocr:  2100
docvqa:             1400
chartqa:             700
```

Teacher generation now tracks accepted-item source quotas. The 20k target is interpreted as approximately 8000 Visual Genome, 6000 TextVQA/TextOCR, 4000 DocVQA, and 2000 ChartQA accepted items. The selection pool is larger than strictly needed because some images may fail validation or produce fewer accepted items.

## Rebuilding Manifest and Selection

If datasets change, rebuild the image manifest:

```bash
python -m tgvf_data.prepare build-manifest \
  --dataset-root /home/dredvpn009/Flash_Storage/datasets \
  --project-root /home/dredvpn009/Flash_Storage/projects/r-vlm/revisit_vlm \
  --registry data/tgvf_teacher/preparation/dataset_registry.yaml \
  --output data/tgvf_teacher/preparation/image_pool_manifest.latest.jsonl \
  --append-only \
  --no-inspect-images
```

Then create the 20k selection:

```bash
python -m tgvf_data.prepare select \
  --manifest data/tgvf_teacher/preparation/image_pool_manifest.latest.jsonl \
  --target-samples 20000 \
  --selected-images 7000 \
  --output data/tgvf_teacher/preparation/selections/selected_images_20k_samples_v0.jsonl \
  --seed 20260525 \
  --selection-run-id selection_20k_v0 \
  --planned-teacher-run-id teacher_run_000001
```

The selection logic is source-dataset based, not only source-profile based.

## Teacher Data Generation

Main module:

```text
src/tgvf_data/generate_teacher.py
```

It reads a selected-image manifest and writes a run directory:

```text
data/tgvf_teacher/generated/runs/<run_id>/
```

Important outputs:

```text
final/tgvf_teacher_items.accepted.jsonl
final/tgvf_teacher_items.rejected.jsonl
reports/generation_summary.json
reports/quality_filter_report.json
reports/cost_usage_report.json
```

Global ledger:

```text
data/tgvf_teacher/generated/teacher_generation_ledger.jsonl
```

The ledger enables resume and avoids reprocessing successful images.

### API Key

Preferred safe pattern:

```bash
export OPENAI_API_KEY='your_key_here'
```

Then run commands without putting the key in the command history.

A one-line command with `OPENAI_API_KEY=... command` also works, but shell history may capture the key.

### Smoke Test

Run a small smoke before the 20k generation:

```bash
python -m tgvf_data.generate_teacher smoke-sync \
  --selection data/tgvf_teacher/preparation/selections/selected_images_20k_samples_v0.jsonl \
  --project-root /home/dredvpn009/Flash_Storage/projects/r-vlm/revisit_vlm \
  --run-id teacher_run_000001_smoke \
  --limit-images 100 \
  --target-accepted-samples 200 \
  --model gpt-5.4 \
  --detail original \
  --concurrency 4 \
  --max-retries 5
```

If smoke shows zero requests and zero tokens, it probably skipped images already marked `succeeded` in the ledger. For a true rerun smoke, add:

```bash
--allow-regenerate
```

Do not use `--allow-regenerate` for the normal 20k run unless you intentionally want to pay for already-processed images again.

### Full 20k Generation

```bash
python -m tgvf_data.generate_teacher resume-sync \
  --selection data/tgvf_teacher/preparation/selections/selected_images_20k_samples_v0.jsonl \
  --project-root /home/dredvpn009/Flash_Storage/projects/r-vlm/revisit_vlm \
  --run-id teacher_run_000001 \
  --target-accepted-samples 20000 \
  --model gpt-5.4 \
  --detail original \
  --concurrency 4 \
  --max-retries 5
```

`resume-sync` is resumable. If interrupted, rerun the same command. Images with `succeeded` ledger entries are skipped. Failed images are candidates for retry on the next resume. The stopping condition is source-quota aware, so one high-yield source cannot consume the full 20k budget before DocVQA or ChartQA is reached.

If a run has extra accepted items, export an exact balanced training file:

```bash
python -m tgvf_data.generate_teacher export-balanced \
  --accepted data/tgvf_teacher/generated/runs/teacher_run_000001/final/tgvf_teacher_items.accepted.jsonl \
  --output data/tgvf_teacher/generated/runs/teacher_run_000001/final/tgvf_teacher_items.accepted_balanced_20k.jsonl \
  --target-samples 20000
```

Use the balanced export for training when the raw accepted file has overshoot.

### Progress Output

During generation, progress is printed to stderr:

```text
[tgvf-teacher] ... progress images=... ok=... failed=... accepted=... remaining_accept=... rate=... eta_by_images=...
```

This is separate from the final JSON summary printed at the end.

### W&B Logging for Data Generation

W&B is optional. It is enabled only when `--wandb-project` is supplied.

```bash
python -m tgvf_data.generate_teacher resume-sync \
  --selection data/tgvf_teacher/preparation/selections/selected_images_20k_samples_v0.jsonl \
  --project-root /home/dredvpn009/Flash_Storage/projects/r-vlm/revisit_vlm \
  --run-id teacher_run_000001 \
  --target-accepted-samples 20000 \
  --model gpt-5.4 \
  --detail original \
  --concurrency 4 \
  --max-retries 5 \
  --wandb-project tgvf \
  --wandb-run-name teacher_run_000001 \
  --wandb-tags teacher,data-generation,20k \
  --wandb-log-artifacts
```

Logged metrics include:

- candidate count
- skipped existing images
- processed images
- successful and failed calls
- accepted and rejected items
- remaining accepted item target
- token usage
- source dataset counters
- final quality and usage summary

When `--wandb-log-artifacts` is enabled, the code uploads:

- config
- prompt
- schema
- accepted JSONL
- rejected JSONL
- summary reports

It does not upload raw images, base64 image payloads, or raw OpenAI responses.

## Clearing Generated Data

To restart generation from scratch, remove only generated outputs:

```bash
rm -rf data/tgvf_teacher/generated
mkdir -p data/tgvf_teacher/generated/runs
```

Do not delete:

```text
data/tgvf_teacher/preparation/
/home/dredvpn009/Flash_Storage/datasets/
```

## Accepted Teacher JSONL Schema

Training requires these fields:

```text
image
question
target
evidence_description
```

Teacher generation also stores useful optional fields:

```text
stable_image_uid
source_dataset
source_profile
short_answer
evidence_type
locality
answer_type
visual_difficulty
visibility
confidence
item_content_hash
```

The target should be a neutral visual pointer and should not leak the answer.

## FVT Training

Main training script:

```text
scripts/train_tgvf_fvt.py
```

The training code freezes Qwen2-VL and trains only the TGVF/FVT module.

Frozen:

- Qwen2-VL LLM
- Qwen2-VL visual encoder
- original Qwen visual merger
- tokenizer/processor

Trainable:

- selected TGVF FVT module

Supported variants:

```text
token_direct
pooled
foveal_cross_merger
```

Variant meanings:

- `token_direct`: token-level direct cross-attention. It uses the captured target-token hidden states directly as queries and is useful as an ablation.
- `pooled`: pooled target-query cross-attention. It mean-pools the target hidden states into a global foveation query and emits a fixed number of FVTs.
- `foveal_cross_merger`: pooled query + M x R sub-slots + Qwen-style merger. This is the main/default structural variant.

Default losses:

```text
L_gen = enabled, weight 1.0; readout target dropout defaults to 0.3
L_visual_token_manifold = enabled, weight 0.01
L_same_image_negative = implemented, disabled by default; modes: cyclic_margin or matrix_ce
L_contrastive_alignment = implemented, disabled by default
```

### Training Smoke

After teacher generation creates the accepted JSONL, run a tiny smoke first:

```bash
python scripts/train_tgvf_fvt.py \
  --train-file data/tgvf_teacher/generated/runs/teacher_run_000001/final/tgvf_teacher_items.accepted_balanced_20k.jsonl \
  --output-dir outputs/tgvf_fvt/smoke_foveal_cross_merger \
  --model-name-or-path Qwen/Qwen2-VL-2B-Instruct \
  --variant foveal_cross_merger \
  --num-foveated-tokens 16 \
  --device cuda:0 \
  --torch-dtype bfloat16 \
  --attn-implementation flash_attention_2 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --max-steps 2 \
  --save-every 2 \
  --log-every 1
```

To smoke all supported variants on the balanced 20k file:

```bash
for variant in token_direct pooled foveal_cross_merger; do
  python scripts/train_tgvf_fvt.py \
    --train-file data/tgvf_teacher/generated/runs/teacher_run_000001/final/tgvf_teacher_items.accepted_balanced_20k.jsonl \
    --output-dir outputs/tgvf_fvt/smoke_b200_balanced_20k_${variant} \
    --model-name-or-path Qwen/Qwen2-VL-2B-Instruct \
    --variant "${variant}" \
    --num-foveated-tokens 16 \
    --device cuda:0 \
    --torch-dtype bfloat16 \
    --attn-implementation flash_attention_2 \
    --batch-size 1 \
    --gradient-accumulation-steps 1 \
    --max-steps 2 \
    --save-every 2 \
    --log-every 1
done
```

Current B200 smoke results on `tgvf_teacher_items.accepted_balanced_20k.jsonl`:

```text
token_direct:
  completed 2 steps
  FVT shape: [16, 1536]
  checkpoint: outputs/tgvf_fvt/smoke_b200_balanced_20k_token_direct/checkpoint_step_2.pt

pooled:
  completed 2 steps
  FVT shape: [16, 1536]
  checkpoint: outputs/tgvf_fvt/smoke_b200_balanced_20k_pooled/checkpoint_step_2.pt

foveal_cross_merger:
  completed 2 steps
  FVT shape: [16, 1536]
  checkpoint: outputs/tgvf_fvt/smoke_b200_balanced_20k_foveal_cross_merger/checkpoint_step_2.pt
```

All three checkpoints contain only TGVF module state plus config/optimizer metadata, not Qwen2-VL weights.

### Full Training Example

One full pass over the balanced 20k file uses `--max-steps 20000` with `--batch-size 1`. With `--gradient-accumulation-steps 8`, this is 20,000 sample steps and 2,500 optimizer updates.

```bash
python scripts/train_tgvf_fvt.py \
  --train-file data/tgvf_teacher/generated/runs/teacher_run_000001/final/tgvf_teacher_items.accepted_balanced_20k.jsonl \
  --output-dir outputs/tgvf_fvt/teacher_run_000001_foveal_cross_merger \
  --model-name-or-path Qwen/Qwen2-VL-2B-Instruct \
  --variant foveal_cross_merger \
  --num-foveated-tokens 16 \
  --device cuda:0 \
  --torch-dtype bfloat16 \
  --attn-implementation flash_attention_2 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --lr-scheduler warmup_cosine \
  --warmup-ratio 0.03 \
  --warmup-steps 0 \
  --min-lr-ratio 0.1 \
  --max-steps 20000 \
  --save-every 1000 \
  --log-every 50
```

Current B200 timing check:

```text
variant: foveal_cross_merger
steps: 10
device: cuda:0, NVIDIA B200
model: Qwen/Qwen2-VL-2B-Instruct
attention: flash_attention_2
total wall time including model load: 29.10 seconds
```

For a full 20k single-pass run, use a practical estimate of about 14 to 18 hours on one B200 for `foveal_cross_merger`, assuming similar image sizes and no storage stalls. The timing depends strongly on image resolution because `pre_merge_visual_shape` varies per sample.

### 4-GPU DDP Training

The training script supports DDP under `torchrun`. In DDP mode each rank loads one frozen Qwen2-VL copy, the TGVF module is wrapped with `DistributedDataParallel`, the dataset is sharded with `DistributedSampler`, and only rank 0 writes W&B/checkpoints.

Smoke test on GPUs 0,1,2,3:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  scripts/train_tgvf_fvt.py \
  --train-file data/tgvf_teacher/generated/runs/teacher_run_000001/final/tgvf_teacher_items.accepted_balanced_20k.jsonl \
  --output-dir outputs/tgvf_fvt/ddp_smoke_0123_foveal_cross_merger \
  --model-name-or-path Qwen/Qwen2-VL-2B-Instruct \
  --variant foveal_cross_merger \
  --num-foveated-tokens 16 \
  --device auto \
  --torch-dtype bfloat16 \
  --attn-implementation flash_attention_2 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --max-steps 2 \
  --save-every 2 \
  --log-every 1 \
  --wandb-mode disabled
```

Full 20k run on GPUs 0,1,2,3:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  scripts/train_tgvf_fvt.py \
  --train-file data/tgvf_teacher/generated/runs/teacher_run_000001/final/tgvf_teacher_items.accepted_balanced_20k.jsonl \
  --output-dir outputs/tgvf_fvt/teacher_run_000001_foveal_cross_merger_ddp0123 \
  --model-name-or-path Qwen/Qwen2-VL-2B-Instruct \
  --variant foveal_cross_merger \
  --num-foveated-tokens 16 \
  --device auto \
  --torch-dtype bfloat16 \
  --attn-implementation flash_attention_2 \
  --batch-size 1 \
  --gradient-accumulation-steps 2 \
  --learning-rate 1e-4 \
  --max-steps 5000 \
  --save-every 5000 \
  --log-every 25 \
  --progress \
  --wandb-project tgvf \
  --wandb-run-name fvt_train_teacher_run_000001_ddp0123 \
  --wandb-tags train,fvt,foveal_cross_merger,balanced_20k,ddp0123
```

For DDP, `--max-steps` is per-rank loop steps. With 4 ranks and `--batch-size 1`, `--max-steps 5000` processes about 20,000 samples total. With `--gradient-accumulation-steps 2`, the effective global batch is 8 samples per optimizer update, matching the single-card command that uses `--gradient-accumulation-steps 8`. If you intentionally want a larger global batch of 32, use `--gradient-accumulation-steps 8`.

Current 4-GPU smoke result on GPUs 0,1,2,3:

```text
completed: yes
world_size: 4
rank-0 checkpoint: outputs/tgvf_fvt/ddp_smoke_0123_foveal_cross_merger/checkpoint_step_2.pt
checkpoint contains: tgvf_module, config, optimizer, global_step
Qwen weights saved: no
qwen_frozen: true
second_full_forward_used: false
```

### W&B Logging for Training

```bash
python scripts/train_tgvf_fvt.py \
  --train-file data/tgvf_teacher/generated/runs/teacher_run_000001/final/tgvf_teacher_items.accepted_balanced_20k.jsonl \
  --output-dir outputs/tgvf_fvt/teacher_run_000001_foveal_cross_merger \
  --model-name-or-path Qwen/Qwen2-VL-2B-Instruct \
  --variant foveal_cross_merger \
  --num-foveated-tokens 16 \
  --device cuda:0 \
  --torch-dtype bfloat16 \
  --attn-implementation flash_attention_2 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --lr-scheduler warmup_cosine \
  --warmup-ratio 0.03 \
  --warmup-steps 0 \
  --min-lr-ratio 0.1 \
  --max-steps 20000 \
  --save-every 1000 \
  --log-every 50 \
  --wandb-project tgvf \
  --wandb-run-name fvt_train_teacher_run_000001 \
  --wandb-tags train,fvt,foveal_cross_merger \
  --wandb-log-checkpoints
```

Logged metrics include:

- total loss
- generation loss
- visual token manifold loss
- optional loss values
- gradient norm
- learning rate
- target hidden-state shape
- pre-merge visual feature shape
- merged visual token shape
- FVT shape
- whether Qwen is frozen
- whether a second full forward was used

When `--wandb-log-checkpoints` is enabled, checkpoint files and `config.json` are uploaded as W&B artifacts.

Optional model watching:

```bash
--wandb-watch gradients
```

Use this sparingly because it can add overhead.

## Environment Notes

The current environment has:

```text
CUDA available
8 GPUs
NVIDIA B200
flash_attn installed
wandb installed
```

The training script defaults to:

```text
--attn-implementation flash_attention_2
--torch-dtype bfloat16
```

If flash-attn is unavailable in another environment, use a supported Transformers attention implementation instead.

## Validation Commands

Current affected tests:

```bash
python -m pytest tests/test_tgvf_generate_teacher.py tests/test_tgvf_training.py -q
```

Lint for touched files:

```bash
python -m ruff check src/revisit_vlm/wandb_logging.py src/tgvf_data/generate_teacher.py scripts/train_tgvf_fvt.py
```

## Safety Notes

- Do not commit or log OpenAI API keys.
- Do not upload raw images or base64 payloads to W&B unless explicitly intended and allowed by dataset licenses.
- Keep generated teacher JSONL inside the project under `data/tgvf_teacher/generated/`.
- Keep raw datasets under `/home/dredvpn009/Flash_Storage/datasets/`.
- For production generation, rely on the ledger rather than `--allow-regenerate`.
