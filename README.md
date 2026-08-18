# Feedback Prize — Evaluating Student Writing

Token Classification / NER project for the [Feedback Prize - Evaluating Student Writing](https://www.kaggle.com/competitions/feedback-prize-2021) Kaggle competition.

## Task

Given full argumentative essays written by students, build an NLP pipeline that segments the text and classifies each section into a discourse element:

`Lead`, `Position`, `Claim`, `Counterclaim`, `Rebuttal`, `Evidence`, `Concluding Statement`

This is framed as **token classification (NER)**, not standard document classification — the model must locate segment boundaries within the essay and label each one.

## Project Structure

```
.
├── data/
│   ├── raw/            # downloaded competition data (train.csv, test/, sample_submission.csv)
│   └── processed/      # cleaned / feature-engineered data (generated)
├── models/              # saved model artifacts (generated)
├── notebooks/
│   └── eda_and_baseline.ipynb
├── main.py
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

1. Clone the repository and create a virtual environment.
2. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
3. Copy `.env.example` to `.env` and fill in your Kaggle credentials:
  ```bash
  cp .env.example .env
  ```
  ```
  KAGGLE_USERNAME=your_kaggle_username
  KAGGLE_API_TOKEN=your_kaggle_api_key
  ```
4. Run `notebooks/eda.ipynb` — the data download is automatic and cached (skipped on repeat runs).

## EDA Summary

Exploratory analysis was done at three levels: segment level, essay level, and data-quality checks.

### 1. Class balance & segment length (segment level)
- Strong class imbalance: `Claim` (~50k) and `Evidence` (~46k) dominate; `Rebuttal` and `Counterclaim` are underrepresented (~5–8k).
- Segment length varies a lot by type: `Evidence` and `Concluding Statement` run longest (median ~50–70 words, outliers up to 300), while `Position`/`Claim` are short and consistent (median ~15–20 words).
- **Implication:** class weighting or a similar imbalance-handling strategy will likely be needed during training.

### 2. Essay-level aggregation
- **15,594** unique essays, **144,293** total segments, average **9.3 segments per essay**.
- Segment count per essay is right-skewed but centered around 7–11 segments.

### 3. Essay length & token limits
- Median essay length: **365 words**, mean **~402 words**, max **1,359 words**.
- **22.9% of essays (3,577) exceed 512 words** — the typical context limit for standard transformer models (BERT-like).
- **Implication:** naive truncation at 512 tokens would cut off a meaningful share of essays, likely losing late-essay content such as `Concluding Statement`. Candidate strategies: sliding window with stride, or a long-context model (Longformer / BigBird).

### 4. Data quality checks
- No missing values in any column.
- No duplicate rows.
- All 7 expected discourse types present, no unexpected labels.
- **Implication:** no cleaning step required before modeling.

### 5. Qualitative check
- Visualized one full essay with its segments in order — confirmed segments are sequential and span the text contiguously, consistent with a BIO-tagging / token classification setup rather than multi-label document classification.

## Status

- [x] Data pipeline (Kaggle download + local caching)
- [x] Exploratory Data Analysis
- [ ] Long-essay handling strategy (sliding window vs. long-context model)
- [ ] Baseline model
- [ ] Evaluation metric implementation
- [ ] Model training & tuning

## Notes

- `data/` and `.env` are gitignored — data is re-downloaded via `kagglehub` on first run, credentials are never committed.
- See `.env.example` for the required environment variables.