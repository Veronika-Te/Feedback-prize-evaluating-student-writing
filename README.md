# Feedback Prize — Evaluating Student Writing

Token Classification / NER project for the [Feedback Prize - Evaluating Student Writing](https://www.kaggle.com/competitions/feedback-prize-2021) Kaggle competition.

> **Status: work in progress.** Baselines 1–4 below are initial results, not final numbers. Baseline 1 in particular is still being tuned for better performance, and the comparison table will be updated as that work continues.

## Task

Given argumentative essays written by students, build an NLP pipeline that segments the text and classifies each section into one of 7 discourse elements: **Lead, Position, Claim, Counterclaim, Rebuttal, Evidence, Concluding Statement**.

This is implemented as a **Token Classification (NER)** task using a BIO tagging scheme (15 total labels: `O` + `B-`/`I-` for each of the 7 discourse types).

## Baselines Overview

Four baselines were built, exploring different backbones and strategies for handling essays longer than standard transformer context windows (23% of essays exceed 512 words).

**Compute environment:** all baselines are designed to run on **Google Colab**, except **baseline 3**, which was trained **locally on Apple Silicon (Mac, MPS backend)**.

| Baseline | Notebook | Backbone | Long-context strategy | Environment | Validation Macro-F1 |
|---|---|---|---|---|---|
| 1 | `baseline1.ipynb` | `microsoft/deberta-v3-base` | Truncation @ 1024 tokens | Colab | **0.5893** _(pre-tuning; further tuning in progress)_ |
| 2 | `baseline2.ipynb` | `microsoft/deberta-v3-small` | Sliding window | Colab | **_ADJUST: fill in score_** |
| 3 | `baseline3_mps.ipynb` | `microsoft/deberta-v3-base` | Sliding window | Local (Mac/MPS) | **~0.46–0.47** |
| 4 | `baseline4_longform.ipynb` | `allenai/longformer-base-4096` | Full-length, no truncation (sliding-window attention, up to 2048 tokens) | Colab | **0.5793** |

### Key takeaways so far

- **Baseline 1 (DeBERTa-v3-base, truncated)** is the strongest result to date (0.5893), despite discarding the tail of ~23% of essays. This is the current focus for further tuning — see *Status* below.
- **Baselines 2 & 3 both use a sliding window over `deberta-v3-small`/`deberta-v3-base`** to see the full essay instead of truncating, at the cost of extra compute and potential boundary artifacts between chunks. Baseline 3 (base model, local) currently sits around 0.46–0.47; baseline 2 (small model) has not yet been fully evaluated.
- **Baseline 4 (Longformer-4096)** natively handles full essays without chunking, using global attention on the `<s>` token and local sliding-window attention elsewhere. It reaches 0.5793 — close to baseline 1, and the strongest of the long-context approaches so far.
- **Rebuttal and Counterclaim are the weakest classes across every baseline evaluated so far**, consistent with them being the rarest discourse types in training data (~3% and ~4% respectively). This appears to be a data-imbalance problem more than an architecture problem, since it persists regardless of backbone or context strategy.
- None of the sliding-window/long-context approaches have yet surpassed baseline 1's truncation-based score — this is an open question the project is still investigating, not a settled conclusion.

## Pipeline Architecture

### Baseline 1 (DeBERTa-v3-base, truncation)
- **Preprocessing:** Word-level BIO tagging and subword alignment using `DeBERTaV2TokenizerFast`.
- **Model:** `microsoft/deberta-v3-base` for token-level feature extraction.
- **Training:** PyTorch AMP (Mixed Precision), AdamW + Linear Warmup. Runs on Colab.
- **Context handling:** Truncation at 1024 tokens.
- **Status:** currently being tuned further for better results.

### Baseline 2 & 3 (DeBERTa-v3, sliding window)
- **Model:** `microsoft/deberta-v3-small` (baseline 2) / `microsoft/deberta-v3-base` (baseline 3) for token-level feature extraction.
- **Context handling:** essays longer than the max sequence length are split into overlapping chunks (sliding window) instead of truncated, so the full essay is seen across multiple forward passes.
- **Environment:** baseline 2 runs on Colab; baseline 3 runs **locally on Apple Silicon (Mac, MPS backend)**.
- Predicted BIO tags are merged back into spans and converted into Kaggle's word-index `predictionstring` format for evaluation/submission.

### Baseline 4 (Longformer-4096)
- **Backbone:** `allenai/longformer-base-4096` with full-length sequence processing (up to 4096 tokens; configured here at 2048).
- **Attention Configuration:** Sliding-window localized attention with global attention placed on `<s>` (token index 0).
- **Optimization:** Native PyTorch mixed precision (`torch.amp`), gradient accumulation (effective batch size = 16), gradient clipping, linear warmup learning rate decay.
- **Evaluation:** Official Kaggle Feedback Prize Macro-F1 metric, word-level IoU ≥ 0.5 overlap matching across all 7 discourse classes.
- **Post-processing:** `DiscourseSpanDecoder` aggregates subword BIO predictions into valid spans, with per-class minimum length constraints (e.g. `Evidence` ≥ 5 words, `Lead`/`Position`/`Claim`/`Counterclaim`/`Rebuttal` ≥ 3 words, `Concluding Statement` ≥ 4 words) and an `O`-class decision threshold (0.45) to reduce spurious non-entity predictions.

## Setup & Environment Support

The project is designed to be environment-agnostic, working seamlessly on both Google Colab and local environments (Mac/Linux) without any code changes — the notebooks detect the platform automatically.

### 1. Credentials (Kaggle API)

The data pipeline supports credentials from three sources, checked in order:

- **Colab Secrets** (on Colab) — set `KAGGLE_USERNAME` and `KAGGLE_API_TOKEN` once via the 🔑 icon in Colab, reused across sessions and notebooks.
- **Local `.env`** (on Mac/local) — place a `.env` file in the project root with `KAGGLE_USERNAME` and `KAGGLE_API_TOKEN` (see `.env.example`).
- **Interactive fallback** — if neither is found, the notebook securely prompts for credentials via `getpass` (input is hidden, nothing is written to disk).

### 2. Installation

```bash
pip install -r requirements.txt
```

### 3. Automatic Data Download

Running any of the notebooks (`eda.ipynb`, `baseline1.ipynb`, `baseline2.ipynb`, `baseline3_mps.ipynb`, or `baseline4_longform.ipynb`) will automatically:

- Detect the environment root (`/content` on Colab, project root on local).
- Check whether data already exists in `data/raw/`.
- Download and extract the dataset via `kagglehub` if missing.

**Note:** on Colab, `/content` is wiped when the runtime disconnects — data and installed packages need to be re-fetched on each new session (this happens automatically on the next run). On local, `data/raw/` persists on disk between sessions.

## Project Structure

```
.
├── data/
│   └── raw/
│       ├── train.csv
│       ├── sample_submission.csv
│       ├── train/                 # Essay .txt files (train) — gitignored
│       └── test/                  # Essay .txt files (test) — gitignored
├── notebooks/
│   ├── eda.ipynb                  # Exploratory Data Analysis
│   ├── baseline1.ipynb            # DeBERTa-v3-base, truncation @ 1024 tokens
│   ├── baseline2.ipynb            # DeBERTa-v3-small, sliding window
│   ├── baseline3_mps.ipynb        # DeBERTa-v3-base (local/MPS only)
│   └── baseline4_longform.ipynb   # Longformer
├── main.py
├── requirements.txt
├── .env                           # Kaggle credentials (gitignored, local only)
├── .env.example
└── README.md
```

## Key Insights & Decisions

### Handling Long Context

- **The Problem:** EDA showed 22.9% of essays exceed 512 words. Standard BERT-style truncation loses the end of the text, where the Concluding Statement is usually located.
- **Approaches tried:**
  - Baseline 1/2: truncate at 1024 tokens — captures the full structure of most essays, but still loses the tail on the longest ones.
  - Baseline 3: sliding window over the essay at inference/training time — sees the full essay, at the cost of extra compute and potential boundary artifacts between chunks.
  - Baseline 4: Longformer's native long-context attention (up to 4096 tokens, configured at 2048 here) — processes the full essay in a single forward pass without chunking.
- **Current finding:** despite the added complexity, baselines 3 and 4 have not yet surpassed baseline 1's truncation-based Macro-F1 (0.5893). This is worth further investigation — see *Recommended next steps*.

### Handling Class Imbalance

- EDA found `Claim` (~35%) and `Evidence` (~32%) dominate the dataset, while `Rebuttal` (~3%) and `Counterclaim` (~4%) are underrepresented.
- The baselines downweight the `O` (Outside) class in the loss function (`o_class_weight: 0.30–0.40`) to reduce the dominant "non-labeled" signal.
- **Remaining gap:** per-class loss weighting specifically for `Rebuttal`/`Counterclaim` has not yet been tuned in any baseline — this is consistently the weakest class across all four approaches.

## Results Summary

**Baseline 1 — DeBERTa-v3-base (truncated)** — Validation Macro-F1: **0.5893** (200 validation essays, IoU ≥ 0.5)

| Discourse Type | F1-Score |
|---|---|
| Concluding Statement | 0.7512 |
| Lead | 0.7110 |
| Position | 0.6300 |
| Counterclaim | 0.5755 |
| Evidence | 0.5750 |
| Claim | 0.5297 |
| Rebuttal | 0.3529 |

**Baseline 4 — Longformer-base-4096** — Validation Macro-F1: **0.5793** (200 validation essays, IoU ≥ 0.5)

| Discourse Type | Precision | Recall | F1-Score |
|---|---|---|---|
| Concluding Statement | 0.6681 | 0.8933 | 0.7644 |
| Lead | 0.6011 | 0.8359 | 0.6993 |
| Position | 0.5947 | 0.6818 | 0.6353 |
| Counterclaim | 0.4576 | 0.6207 | 0.5268 |
| Evidence | 0.4945 | 0.5918 | 0.5388 |
| Claim | 0.4772 | 0.5138 | 0.4948 |
| Rebuttal | 0.3273 | 0.5000 | 0.3956 |

**Baseline 3 — DeBERTa-v3-base (sliding window, local)** — Validation Macro-F1: **~0.46–0.47** (390 validation essays)

| Discourse Type | F1-Score |
|---|---|
| Concluding Statement | 0.638 |
| Lead | 0.633 |
| Position | 0.551 |
| Evidence | 0.511 |
| Counterclaim | 0.310 |
| Claim | 0.297 |
| Rebuttal | 0.225 |

_ADJUST: add baseline 2 (DeBERTa-v3-small, sliding window) results table here once evaluation is finished._

## Recommended Next Steps

- **Finish tuning baseline 1** (DeBERTa-v3-base, truncation) — currently the strongest baseline and the active focus of the project.
- **Finish evaluating baseline 2** (DeBERTa-v3-small, sliding window) so all four baselines can be compared on equal footing.
- Evaluate baseline 3 on its **full validation set** rather than the small sample, and confirm whether overfitting (training loss vs. validation F1 divergence after epoch 3) is the main cause of its gap vs. baseline 1.
- Add **per-class loss weighting** targeting `Rebuttal`/`Counterclaim` specifically, since both remain the weakest class in every baseline regardless of backbone or context strategy.
- Investigate why the sliding-window/long-context approaches (2, 3 & 4) haven't yet outperformed truncation-based baseline 1 — possibly the added chunking/windowing complexity introduces its own error sources that offset the context advantage.

## Status

This project is **still in progress** — treat all results above as intermediate, not final.

- [x] Universal data pipeline (Colab + local support)
- [x] Exploratory Data Analysis
- [x] BIO subword alignment strategy
- [x] DeBERTa-v3-base baseline (truncation) — initial result, tuning in progress
- [ ] DeBERTa-v3-small baseline (sliding window) — training done, full evaluation pending
- [x] DeBERTa-v3-base baseline (sliding window, local/MPS)
- [x] Longformer-4096 baseline (full-length context)
- [x] Mixed precision training (AMP)
- [x] Official Macro-F1 evaluation
- [x] Interactive HTML visualizer (ground truth vs. predicted spans)
- [x] Standalone inference script (`predict.py`)
- [ ] Finish tuning baseline 1 for improved results
- [ ] Finish evaluating baseline 2
- [ ] Per-class loss weighting for Rebuttal / Counterclaim
- [ ] Full validation set evaluation for baseline 3 (currently partial)
- [ ] Investigate why sliding-window/long-context baselines underperform truncation baseline
