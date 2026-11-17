# Orange Jordan eShop FAQ Assistant — Phase 2

**CPT Model:** https://huggingface.co/orange-eshop-team/orange-eshop-team-cpt \\
**SFT Model:** https://huggingface.co/orange-eshop-team/orange-eshop-team-sft

Two-stage fine-tuning of **GPT-2 small (124M)** on the Orange Jordan eShop FAQ dataset:
1. **Continued Pretraining (CPT)** on 42 cleaned FAQ documents (`2,670` train / `260` validation tokens).
2. **Supervised Fine-Tuning (SFT)** on `488` curated question-answer pairs (`438` train / `50` validation, including `50` refusal pairs and `280` LLM-generated paraphrases) with answer-only loss and a custom `<|user|> ... <|assistant|> ... <|end|>` chat template.

---

## Model Checkpoints (Stored Outside Git)

Trained model weights are hosted externally at:
- **Checkpoint Link:** `<https://drive.google.com/drive/folders/1wEj0GBlAGYZFZf2b15wN-T6xqLkdIuDs?usp=sharing>`
  - `checkpoints/cpt/final` — Continued-pretrained GPT-2 small (`val perplexity: 16.53`)
  - `checkpoints/sft/final` — Supervised fine-tuned GPT-2 small (`test accuracy: 53.3%`, `refusal accuracy: 100.0%`)

Download both folders into `checkpoints/cpt/final` and `checkpoints/sft/final` before running inference.

---

## Setup

```bash
pip install -r requirements.txt
```

*(Optional)* Copy `.env.example` to `.env` if you regenerate synthetic SFT questions via an API:
```bash
cp .env.example .env
```

---

## How to Train

All hyperparameters (`model_name: gpt2`, `seed: 42`, `learning_rate: 5e-5`, `epochs: 3`, `max_length: 512`) are configured in `config.yaml`:

```bash
# Step 1: Continued Pretraining (CPT)
python train_cpt.py

# Step 2: Supervised Fine-Tuning (SFT)
python train_sft.py
```

---

## How to Score

View the graded Phase 2 scoreboard summary directly from `results/`:
```bash
python score.py
```

Re-run greedy inference (`do_sample=False`, `max_new_tokens=200`) across all three models (`base`, `+CPT`, `+SFT`) using the shared `generate_answer()` function in `src/utils.py`:
```bash
python score.py --run
```

---

## Repository Structure

```text
├── README.md
├── SCOREBOARD.md
├── config.yaml
├── requirements.txt
├── train_cpt.py
├── train_sft.py
├── score.py
├── src/
│   ├── __init__.py
│   └── utils.py
├── data/
│   ├── cpt/
│   │   ├── train.csv
│   │   └── val.csv
│   ├── sft_pairs.jsonl
│   └── sft_hand_check.csv
├── tests/
│   └── test_questions.csv
├── results/
│   ├── base_results.csv
│   ├── cpt_results.csv
│   └── sft_results.csv
├── .env.example
└── .gitignore
```
