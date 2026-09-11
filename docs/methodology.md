# Methodology

Detailed reasoning behind preprocessing and feature-extraction decisions in this project. For the high-level pipeline and usage, see the main [README](../README.md).

---

## Preprocessing audit

Rather than eyeballing individual complaint rows, preprocessing decisions are based on a stratified random sample of ~300 rows (statistically sufficient for a 162K population at 95% confidence). The sample is measured against concrete, countable criteria: percentage of tokens that are stopwords, percentage of documents containing punctuation, percentage containing digits, percentage of short (≤2 character) tokens, and percentage of tokens that are already uninflected. If an issue appears in fewer than a defined threshold of documents, that preprocessing step is skipped rather than applied blindly. This audit table is intended to be the first panel on the eventual dashboard, making the methodology transparent to anyone reviewing the project.

## Lemmatization

Narrative text is lemmatized with spaCy's `en_core_web_sm` model, reducing inflected forms (*payments*, *charged*, *charges*) to a single root token. This is run once and cached to Parquet (`df_lemmatized.parquet`) rather than re-run on every pipeline execution, since it's the most expensive preprocessing step at this row count.

## TF-IDF: global fitting, not per-category

The `TfidfVectorizer` is fit once across the entire lemmatized corpus — all five product categories combined — rather than fit separately within each category.

This was tested both ways. Per-category fitting was rejected: when IDF is computed only within a single category's own documents, a genuinely hallmark term for that category (e.g. "overdraft" within retail banking) can appear in most of that category's documents, driving its IDF — and therefore its TF-IDF score — toward zero. Fitting IDF globally, across all categories, preserves the contrast that makes a term distinctive to one category *relative to the others*, which is the actual goal.

The vectorizer is fit on the lemmatized `Lemmatized` column, not raw narrative text — consolidating inflected variants onto one token gives that token a more accurate, concentrated IDF weight than if its frequency were split across surface forms.

## Per-category term ranking

After the single global fit produces one sparse TF-IDF matrix for the whole corpus:

- Each product category's rows are sliced out via a boolean mask on the `product` column.
- Column-wise means are computed over that subset (`product_tfidf.mean(axis=0)`, flattened with `np.asarray(...).ravel()` for performance — a per-row Python loop over a sparse matrix at this scale was prohibitively slow).
- The mean is taken over the **total number of documents in the category**, including documents where the term never appears, rather than only documents containing the term.

That denominator choice is deliberate. Dividing by total category size lets a term that's genuinely common across many of a category's documents accumulate a meaningful score from many small, nonzero contributions, while diluting toward zero a term that only spikes in a handful of unusually long or repetitive complaints. Dividing only by documents-containing-the-term instead would let a small number of outlier rants dominate the ranking — a moderately rare term appearing intensely in, say, 50–100 out of 30,000 documents could survive averaging and look "representative" when it's really an idiosyncratic writing-style artifact rather than a category-wide signal.

## Filtering

- **Custom stopwords:** a small manually curated list of generic, cross-category verbs (*make, say, get, send, tell, call, ask, use, day*) is removed after ranking, since these surface near the top of every category's list without carrying topical meaning.

## Known limitation — not yet addressed

Mean TF-IDF alone can still be misleading: it doesn't distinguish a term that's broadly representative of a category from one driven by a small number of documents written in an unusually intense style. A companion statistic — **within-category document frequency** (the share of a category's documents that contain the term at all) — has been identified as necessary to make this distinction visible to a dashboard reader, and is planned as a second field alongside the mean score, so the two numbers are read together rather than the mean being taken at face value.

## Sentiment scoring (planned)

VADER will be run on **raw**, non-lemmatized narrative text — lemmatization doesn't meaningfully affect VADER's sentiment lexicon lookup, and normalizing tokens could interfere with how VADER weights punctuation/capitalization intensity cues. All complaints are inherently negative in tone, so compound scores will be framed as *frustration intensity* rather than positive/negative sentiment.
