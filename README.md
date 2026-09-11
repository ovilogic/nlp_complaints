# Customer Complaints NLP

Analyzing ~160K CFPB consumer financial complaint narratives to surface the terms, themes, and language patterns that distinguish one financial product category from another — built as an analyst-employability portfolio project, prioritizing interpretable, well-justified methodology over model complexity.

**Status: work in progress.** Preprocessing, TF-IDF term extraction, and sentiment scoring are done. The classifier tab and dashboard frontend are not yet built. See [Roadmap](#roadmap) below.

---

## What this project does (so far)

Given a complaint narrative and its labeled product category (`credit_card`, `retail_banking`, `credit_reporting`, `mortgages_and_loans`, `debt_collection`), the pipeline:

1. Loads and audits the raw data
2. Lemmatizes narrative text with spaCy
3. Fits TF-IDF globally across the full corpus, then computes the top distinctive terms per product category

Planned next: a static frontend dashboard using the sentiment JSON export and an optional Naive Bayes classifier tab.

Full reasoning behind each methodological choice — why TF-IDF is fit globally rather than per-category, how the per-category mean is computed and why, the stopword and cross-category filtering applied — is documented in [`docs/methodology.md`](docs/methodology.md).

---

## Dataset

- **Source:** [CFPB Consumer Complaints (Kaggle)](https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp)
- **Rows:** 162,421
- **Columns:** `product`, `narrative`
- **Categories:** `credit_card`, `retail_banking`, `credit_reporting`, `mortgages_and_loans`, `debt_collection`

Not included in this repo — download the CSV from Kaggle and place it at `data/complaints_processed.csv`.

---

## Pipeline

```
Load data
    ↓
Audit / inspect               ✅ done
    ↓
Lemmatize (spaCy)             ✅ done
    ↓
TF-IDF vectorization          ✅ done
    ↓
Sentiment scoring (VADER)     ✅ done
    ↓
JSON export                   ✅ done
    ↓
Naive Bayes (optional)        ⏳ planned
    ↓
Static JS dashboard           ⏳ planned
```

---

## Setup

```bash
git clone https://github.com/ovilogic/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Place the Kaggle CSV at `data/complaints_processed.csv`.

## Usage

Run the pipeline end to end:

```bash
python load_data.py
```

On first run this will lemmatize the full corpus with spaCy and cache the result to `df_lemmatized.parquet` (this step is slow — expect it to take a while on ~160K rows). Subsequent runs load the cached Parquet file instead of re-lemmatizing.

The script prints, for each product category, the top 20 TF-IDF terms and their category-mean scores, along with a cross-category check flagging terms that score higher in a *different* category than the one they're listed under.

---

## Roadmap

- [x] Data loading + statistical audit (stratified sample, stopword/punctuation/token-length checks)
- [x] spaCy lemmatization with Parquet checkpointing
- [x] Global TF-IDF fitting + per-category mean term extraction
- [x] VADER sentiment ("frustration intensity") scoring on raw narrative text
- [x] JSON export for frontend consumption
- [ ] Plain JS dashboard (audit panel, complaints-by-product, sentiment distribution, top themes, insights)
- [ ] Optional Naive Bayes classifier tab
- [ ] Deployment (Raspberry Pi + AWS Route 53 - Cloudflare, static frontend)

---

## Tools

| Tool | Purpose |
|---|---|
| `pandas` | Data loading, aggregation, JSON export |
| `spaCy` | Lemmatization |
| `scikit-learn` | TF-IDF vectorization, Naive Bayes classifier |
| `nltk` / VADER | Sentiment scoring |

## Documentation

- [`docs/methodology.md`](docs/methodology.md) — detailed reasoning behind preprocessing and TF-IDF design decisions
- Full backend spec: `NLP_Complaints_Project.md` (this repo's original planning doc)

## License

_Add a license (e.g. MIT) before making the repo public, if you haven't already._
