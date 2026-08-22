# Feedback Prize — Evaluating Student Writing

Token Classification / NER project for the [Feedback Prize - Evaluating Student Writing](https://www.kaggle.com/competitions/feedback-prize-2021) Kaggle competition.

## Task

Given argumentative essays written by students, build an NLP pipeline that segments the text and classifies each section into one of 7 discourse elements:
`Lead`, `Position`, `Claim`, `Counterclaim`, `Rebuttal`, `Evidence`, `Concluding Statement`.

This is implemented as a **Token Classification (NER)** task using a **BIO tagging** scheme (15 total labels).

## Pipeline Architecture

1. **Preprocessing:** Word-level BIO tagging and subword alignment using `DeBERTaV2TokenizerFast`.
2. **Model:** `microsoft/deberta-v3-base` for token-level feature extraction.
3. **Training:** PyTorch AMP (Mixed Precision) on Tesla T4, AdamW + Linear Warmup.
4. **Post-Processing:** Aggregating subword predictions into spans with length filtering.
5. **Evaluation:** Official competition metric (Macro-F1 based on Span Overlap IoU ≥ 0.5).

## Setup & Environment Support

The project is designed to be **environment-agnostic**, working seamlessly on both **Google Colab** and **local environments** (Mac/Linux) without any code changes — the notebooks detect the platform automatically.

### 1. Credentials (Kaggle API)

The data pipeline supports credentials from three sources, checked in order:

1. **Colab Secrets** (on Colab) — set `KAGGLE_USERNAME` and `KAGGLE_API_TOKEN` once via the 🔑 icon in Colab, reused across sessions and notebooks.
2. **Local `.env`** (on Mac/local) — place a `.env` file in the project root with `KAGGLE_USERNAME` and `KAGGLE_API_TOKEN` (see `.env.example`).
3. **Interactive fallback** — if neither is found, the notebook securely prompts for credentials via `getpass` (input is hidden, nothing is written to disk).

### 2. Installation

```bash
pip install -r requirements.txt
```

### 3. Automatic Data Download

Running `notebooks/eda.ipynb` or `notebooks/baseline1.ipynb` will automatically:

1. Detect the environment root (`/content` on Colab, project root on local).
2. Check whether data already exists in `data/raw/`.
3. Download and extract the dataset via `kagglehub` if missing.

**Note:** on Colab, `/content` is wiped when the runtime disconnects — data and installed packages need to be re-fetched on each new session (this happens automatically on the next run). On local, `data/raw/` persists on disk between sessions.

## Project Structure

```
.
├── data/
│   └── raw/            # Dataset (train.csv and essay .txt files) — gitignored
├── models/              # Saved model_weights.pt — gitignored
├── notebooks/
│   ├── eda.ipynb        # Exploratory Data Analysis
│   └── baseline1.ipynb  # DeBERTa-v3 training & validation
├── main.py
├── predict.py            # Standalone CLI inference script
├── requirements.txt
├── .env                  # Kaggle credentials (gitignored, local only)
├── .env.example
└── README.md
```

## Key Insights & Decisions

### Handling Long Context

- **The Problem:** EDA showed **22.9% of essays exceed 512 words**. Standard BERT truncation would lose the end of the text, where the `Concluding Statement` is usually located.
- **The Solution:** The baseline uses a **max length of 1024 tokens**, which captures the full structure of the vast majority of essays in the dataset.

### Handling Class Imbalance

- EDA found `Claim` (~35%) and `Evidence` (~32%) dominate the dataset, while `Rebuttal` (~3%) and `Counterclaim` (~4%) are underrepresented.
- The baseline downweights the `O` (Outside) class in the loss function (`o_class_weight: 0.30`) to reduce the dominant "non-labeled" signal.
- **Remaining gap:** per-class weighting for `Rebuttal`/`Counterclaim` specifically has not yet been tuned — see results below, where `Rebuttal` is the weakest class.

## Results (Baseline DeBERTa-v3)

The model achieves a **Validation Macro-F1 Score of 0.5893** (200 validation essays, IoU ≥ 0.5).

| Discourse Type       | F1-Score |
|-----------------------|----------|
| Concluding Statement | 0.7512   |
| Lead                  | 0.7110   |
| Position              | 0.6300   |
| Counterclaim          | 0.5755   |
| Evidence              | 0.5750   |
| Claim                 | 0.5297   |
| Rebuttal              | 0.3529   |

`Rebuttal` is the weakest-performing class, consistent with it being the rarest discourse type in training data (~3%). Improving its weighting is a natural next step.

## Status

- [x] Universal data pipeline (Colab + local support)
- [x] Exploratory Data Analysis
- [x] BIO subword alignment strategy
- [x] DeBERTa-v3 baseline implementation
- [x] Mixed precision training (AMP)
- [x] Official Macro-F1 evaluation
- [x] Interactive HTML visualizer
- [x] Standalone inference script (`predict.py`)
- [ ] Per-class loss weighting for `Rebuttal` / `Counterclaim`
- [ ] Full validation set evaluation (currently 200/2,340 essays)