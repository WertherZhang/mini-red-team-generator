# Red-Team Generator (HateXplain + Structured Synthesis + LoRA)

This repository provides end-to-end pipelines for red-team text generation and evaluation.

## What is implemented

1. Data strategy
- Seed data: `HateXplain`
- Structured synthesis: long-tail pattern expansion using templates, paraphrases, and obfuscation mechanisms

2. Conditional generation
- Decoder-only LM generation with control codes
- LoRA fine-tuning via `transformers + peft`

3. Evaluation
- Coverage: `category/style/context/intensity/obf`
- Diversity: `distinct-1/2/3`, `Self-BLEU`, `Embedding Diversity`

In addition, the repo also includes a lightweight baseline pipeline in:
- `/Users/admin/Desktop/redteam_generator/src/safe_redteam_pipeline.py`

## Project structure

- `/Users/admin/Desktop/redteam_generator/src/redteam_generator.py`: unified CLI entry for the LoRA pipeline
- `/Users/admin/Desktop/redteam_generator/src/safe_redteam_pipeline.py`: lightweight category-conditioned generation + baseline comparison
- `/Users/admin/Desktop/redteam_generator/data/`: seed and synthetic training data
- `/Users/admin/Desktop/redteam_generator/artifacts/`: LoRA artifacts (adapter/tokenizer/checkpoints)
- `/Users/admin/Desktop/redteam_generator/outputs/`: generated outputs and evaluation reports
- `/Users/admin/Desktop/redteam_generator/conditioned_wordcloud.ipynb`: notebook for word cloud and metric display

## Setup

```bash
cd /Users/admin/Desktop/redteam_generator
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline A: Build HateXplain seed

Option A (recommended): load from Hugging Face `datasets`.

```bash
python3 src/redteam_generator.py build-seed \
  --source hf \
  --out_seed data/hatexplain_seed.jsonl
```

Option B: use local HateXplain JSON.

```bash
python3 src/redteam_generator.py build-seed \
  --source local_json \
  --local_json_path /path/to/HateXplain/dataset.json \
  --out_seed data/hatexplain_seed.jsonl
```

## Pipeline A: Structured synthesis (long-tail attack patterns)

```bash
python3 src/redteam_generator.py synthesize \
  --seed_file data/hatexplain_seed.jsonl \
  --max_total 12000 \
  --samples_per_seed 6 \
  --out_file data/structured_synth.jsonl \
  --summary_file outputs/synth_summary.json
```

Generated records include:
- `category/style/context/intensity/obf`
- `control_code`
- `text`
- `train_text` (for LoRA training)

## Pipeline A: Train conditional model with LoRA

```bash
python3 src/redteam_generator.py train-lora \
  --train_file data/hatexplain_seed.jsonl \
  --base_model Qwen/Qwen3-0.6B \
  --output_dir artifacts/lora_redteam \
  --num_train_epochs 1.0 \
  --batch_size 2 \
  --grad_accum_steps 8
```

Notes:
- Control code format:
  - `<|category:...|><|style:...|><|ctx:...|><|intensity:...|><|obf:...|>`
- Training objective:
  - Learn natural syntax from seed/template text
  - Learn diverse expression from paraphrase/control-code combinations
  - Learn obfuscation mechanisms from synthetic data

## Pipeline A: Generate samples

```bash
python3 src/redteam_generator.py generate \
  --base_model Qwen/Qwen3-0.6B \
  --adapter_dir artifacts/lora_redteam/adapter \
  --tokenizer_dir artifacts/lora_redteam/tokenizer \
  --num_samples 600 \
  --temperature 0.35 \
  --top_p 0.75 \
  --progress_every 10 \
  --out_file outputs/generated_outputs.jsonl \
  --summary_file outputs/generation_summary.json
```

## Pipeline A: Evaluate (coverage + diversity)

```bash
python3 src/redteam_generator.py evaluate \
  --generated_file outputs/generated_outputs.jsonl \
  --min_count_per_value 20 \
  --out_report outputs/eval_report.json
```

Report highlights:
- `coverage`: counts and distributions by dimension
- `coverage_gate`: pass/fail against minimum count threshold
- `mechanism_coverage`: configured obfuscation vs observed mechanism hits
- `diversity`:
  - `distinct_1/2/3` (higher = more diverse)
  - `self_bleu` (lower = more diverse)
  - `embedding_diversity.mean_cosine_distance` (higher = more diverse)

## Recommended minimal full flow (Pipeline A)

```bash
python3 src/redteam_generator.py build-seed --source hf
python3 src/redteam_generator.py synthesize --max_total 12000
python3 src/redteam_generator.py train-lora --base_model Qwen/Qwen3-0.6B --output_dir artifacts/lora_redteam
python3 src/redteam_generator.py generate --base_model Qwen/Qwen3-0.6B --adapter_dir artifacts/lora_redteam/adapter --tokenizer_dir artifacts/lora_redteam/tokenizer --num_samples 600 --temperature 0.35 --top_p 0.75
python3 src/redteam_generator.py evaluate --generated_file outputs/generated_outputs.jsonl --out_report outputs/eval_report.json
```

## One-command run (Pipeline A)

```bash
python3 src/redteam_generator.py run-all \
  --source hf \
  --base_model Qwen/Qwen3-0.6B \
  --max_total 12000 \
  --num_samples 600 \
  --temperature 0.35 \
  --top_p 0.75 \
  --progress_every 10
```

## Pipeline B: Lightweight conditioned generator (fast baseline)

This path uses a category-conditioned Markov + template-mutation approach and writes:
- `outputs/conditioned_outputs.json`
- `outputs/generic_outputs.json`
- `outputs/metrics_report.json`

Run:

```bash
python3 src/safe_redteam_pipeline.py --count 90 --seed_value 13
```

## Notes

- This project is intended for safety red-teaming and robustness evaluation.
- Do not use it for harmful real-world misuse.
- Depending on the selected pipeline/config, output text may use abstract placeholders.
