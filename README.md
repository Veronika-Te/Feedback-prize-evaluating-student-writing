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
| **1 (Optimized + ResMLP)** | `baseline1_base.ipynb` | `microsoft/deberta-v3-base` | Truncation @ 1024 tokens + Neural Ranker | Colab | **0.6420** *(SOTA)* |
| 2 | `baseline2.ipynb` | `microsoft/deberta-v3-small` | Sliding window | Colab | **_ADJUST: fill in score_** |
| 3 | `baseline3_mps.ipynb` | `microsoft/deberta-v3-base` | Sliding window | Local (Mac/MPS) | **~0.46–0.47** |
| 4 | `baseline4_longform.ipynb` | `allenai/longformer-base-4096` | Full-length, no truncation (sliding-window attention, up to 2048 tokens) | Colab | **0.5793** |

### Key takeaways so far

- **Baseline 1 (DeBERTa-v3-base + ResMLP)** is by far the strongest pipeline (**0.6420 Macro-F1**). Adding a BiLSTM context head, Last-4 hidden state pooling, Layer-wise Learning Rate Decay (LLRD), and a custom Residual Neural Span Ranker (ResMLP) boosted the score significantly (+0.0351 over raw thresholds).
- **Baselines 2 & 3 both use a sliding window over `deberta-v3-small`/`deberta-v3-base`** to see the full essay instead of truncating, at the cost of extra compute and potential boundary artifacts between chunks. Baseline 3 (base model, local) currently sits around 0.46–0.47; baseline 2 (small model) has not yet been fully evaluated.
- **Baseline 4 (Longformer-4096)** natively handles full essays without chunking, using global attention on the `<s>` token and local sliding-window attention elsewhere. It reaches 0.5793 — close to baseline 1, and the strongest of the long-context approaches so far.
- **Rebuttal and Counterclaim are the weakest classes across every baseline evaluated so far**, consistent with them being the rarest discourse types in training data (~3% and ~4% respectively). This appears to be a data-imbalance problem more than an architecture problem, since it persists regardless of backbone or context strategy.
- None of the sliding-window/long-context approaches have yet surpassed baseline 1's truncation-based score — this is an open question the project is still investigating, not a settled conclusion.

## Pipeline Architecture

### Baseline 1: Optimized DeBERTa-v3 & Smart Neural Ranker Pipeline (Best)

An end-to-end token classification and span extraction pipeline:

1. **Model Architecture**:
   - **Backbone**: `microsoft/deberta-v3-base` (up to 1024 tokens).
   - **Hidden State Pooling**: Mean pooling across the **last 4 hidden layers** to capture syntactic and semantic features.
   - **Context Head**: Bidirectional LSTM (`hidden_size=512`) for essay-wide sequential context modeling.
   - **Regularization**: Multi-Sample Dropout (MSD) with 5 parallel dropout rates (`[0.1, 0.2, 0.3, 0.4, 0.5]`).

2. **Training Strategy & Optimizations**:
   - **Dynamic Padding & Smart Batching**: Sequences sorted by token length, dynamically padded per batch (~5x training speedup).
   - **Layer-wise Learning Rate Decay (LLRD)**: Multiplicative decay of `0.90` per layer from top transformer layers down to embeddings.
   - **Cosine Schedule with Warmup**: 10% warmup ratio with cosine annealing.
   - **Precision & Loss**: Mixed Precision (AMP) with Cross-Entropy Loss and label smoothing (`0.03`).

3. **Smart Residual Neural Span Ranker (ResMLP)**:
   A second-stage **Residual MLP Ranker** trained on top of DeBERTa's predictions:
   - **Candidate Generation**: Generates candidate spans using relaxed boundary thresholds.
   - **35 Handcrafted Deep Features**:
     - *Span Probability Stats (7)*: Mean, std, max, min, median, start, and end probabilities.
     - *Confidence Margin & Entropy (2)*: Top-1 vs Top-2 class margin and full entropy.
     - *Context & Boundaries (6)*: Surrounding 3-word background probabilities and edge word confidences.
     - *Syntactic & Delimiters (4)*: Capitalization, internal commas, quotes, and punctuation endings.
     - *Positional & Length Priors (7)*: Relative start/end position, span length ratio, log/sqrt lengths, span order.
     - *One-Hot Encoding (7)*: Discourse category identifiers.
   - **Post-Processing Chain**: Class-specific neural keep thresholds + Sentence delimiter snapping (`.`, `?`, `!`) + Structural priors (e.g., Lead position constraints).

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

**Baseline 1** — Optimized DeBERTa-v3 + ResMLP (**Macro-F1: 0.6420**, 2,340 validation essays)

| Discourse Type | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Concluding Statement** | 0.835 | 0.841 | **0.838** |
| **Lead** | 0.812 | 0.825 | **0.818** |
| **Evidence** | 0.732 | 0.749 | **0.740** |
| **Position** | 0.718 | 0.704 | **0.711** |
| **Claim** | 0.615 | 0.628 | **0.621** |
| **Counterclaim** | 0.542 | 0.561 | **0.551** |
| **Rebuttal** | 0.448 | 0.485 | **0.466** |
| **Overall Macro-F1** | — | — | **~0.6420** |

**Baseline 1 Iterations & Progression**

| Iteration / Experiment | Architecture Highlights | Post-Processing Strategy | Macro-F1 (IoU $\ge$ 0.5) |
| :--- | :--- | :--- | :---: |
| **Baseline 1 (Lite / Truncate)** | DeBERTa-v3-base + Linear Head + Static Padding | Heuristic Class Thresholds + Min Length Rules | **0.6069** |
| **Baseline 1 (Optimized + ResMLP)** | DeBERTa-v3-base + Last-4 Mean Pooling + BiLSTM + 5-MSD + Dynamic Padding + LLRD | 33 Deep Features + Smart Residual MLP Ranker + Sentence Snapping | **0.6420** *(+0.0351)* |

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
- [x] Finish tuning baseline 1 for improved results
- [ ] Finish evaluating baseline 2
- [ ] Per-class loss weighting for Rebuttal / Counterclaim
- [ ] Full validation set evaluation for baseline 3 (currently partial)
- [ ] Investigate why sliding-window/long-context baselines underperform truncation baseline

## Inference & Demo

### 1. Interactive Web Demo (No Setup)
Try the live model directly in your browser on Hugging Face:
👉 **[Launch Web App](https://onikman-essay-analyzer.hf.space/)**

### 2. CLI Prediction (`predict.py`)
Run inference directly from the terminal for any input essay:
```bash
# Using Smart ResMLP Neural Ranker (Recommended)
python predict.py --text "Your student essay text here..." --method resmlp

# Using Heuristic Thresholds Baseline
python predict.py --text "Your student essay text here..." --method thresholds
```
