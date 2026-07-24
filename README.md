# Yelp Veterinary Data Pipeline and NLP Sentiment Mining

A memory-efficient data engineering, text preprocessing, and sentiment analysis pipeline built to ingest large-scale datasets, extract industry-specific subsets, and isolate structural drivers of customer satisfaction.

![Sentiment Divergence Chart](https://github.com/MLuftig/yelp-veterinary-sentiment-pipeline/raw/main/tfidf_diverging_chart.png)

---

## Key Findings

An initial TF-IDF sentiment-delta pass surfaced mostly narrative-scaffolding words (`told`, `said`, `could`, `another`, `still`, `later`) on the negative-review side rather than complaint content — a known artifact of unigram TF-IDF picking up how people structure a story rather than what the story is about. After extending the stopword list to remove narrative-reporting verbs and connectors, the corrected delta analysis surfaces substantially cleaner, higher-signal terms:

**Terms most associated with low-star reviews** (delta values shown): `told` (0.0351), `money` (0.0175), `rude` (0.0163), `pay` (0.0097), `horrible` (0.0093), `blood` (0.0092), `charged` (0.0089), `worst` (0.0088), `business` (0.0087), `appointment` (0.0085), `phone` (0.0084), `give` (0.0084), `wanted` (0.0082), `bill` (0.0082), `wrong` (0.0081), `hours` (0.0079), `want` (0.0078), `went` (0.0076), `charge` (0.0074), `please` (0.0074)

**Terms most associated with high-star reviews** (delta values shown): `great` (-0.0259), `love` (-0.0177), `friendly` (-0.0174), `best` (-0.0161), `care` (-0.0147), `highly` (-0.0143), `amazing` (-0.0143), `wonderful` (-0.0142), `thank` (-0.0135), `highly recommend` (-0.013), `recommend` (-0.0129), `helpful` (-0.012), `kind` (-0.0119), `knowledgeable` (-0.0107), `compassionate` (-0.0098), `everyone` (-0.0095), `excellent` (-0.0094), `professional` (-0.0092), `happy` (-0.0089), `thorough` (-0.008)

The corrected negative-review list clusters tightly around **billing and administrative friction**: `money`, `pay`, `charged`, `bill`, `charge`, `appointment`, `phone`, `hours`, `business`. `told` remains the single strongest signal by a wide margin — it was deliberately kept in this text rather than filtered, since it also drives a separate rule-based `communication` category flag earlier in the pipeline, and it plausibly reflects genuine content here too (reviewers relaying what staff told them, often in a complaint framing). The high-star list is consistent with the earlier pass and centers on **staff temperament and competence**: `compassionate`, `knowledgeable`, `kind`, `professional`, `thorough`, `helpful`. Together, this points toward front-of-house communication, billing transparency, and appointment/scheduling handling as the highest-leverage areas for improving review sentiment — clinical trust already appears to be a given for reviewers who leave positive feedback.

---

## Technical Stack and Dependencies

- **Language:** Python 3.x
- **Core Libraries:** pandas, numpy, matplotlib, re, nltk, scikit-learn
- **Environment:** Optimized for Kaggle and Jupyter Notebook environments

---

## Core Project Architecture

- **Memory-Optimized Data Pipeline:** Utilizes chunked JSON streaming (`chunksize=100000`) to parse and filter millions of rows without triggering Out-Of-Memory (OOM) errors.
- **Relational ID Slicing:** Leverages high-performance set-based indexing for O(1) lookups to dynamically isolate user interactions (reviews, tips, check-ins) tied directly to targeted businesses.
- **Text Preprocessing Pipeline:** Normalizes text data by eliminating character noise (punctuation, case, numbers) and removing standard English and domain-specific stopwords.
- **N-Gram TF-IDF Vectorization:** Constructs high-dimensional sparse matrices using unigrams and bigrams (`ngram_range=(1,2)`) to preserve multi-word context.
- **Statistical Sentiment Differencing:** Computes a custom mathematical delta vector between average high-star and low-star text reviews.
- **Analytical Visualizations:** Generates a custom horizontal diverging bar chart plotting statistical keyword variance to isolate operational vulnerabilities from consumer praise.

---

## End-to-End Pipeline Workflow

### Phase 1: Data Isolation and Ingestion — `yelp-reviews-data-extraction.ipynb`

1. **Target Identification:** Scans the core Yelp business dataset to isolate entities explicitly categorized under 'Veterinarian'.
2. **Relational Extraction:** Uses the unique target IDs to slice secondary datasets (`review.json`, `user.json`, `checkin.json`, `tip.json`), filtering out irrelevant entries stream-by-stream.
3. **Structured Storage:** Exports five localized, small-footprint CSV files representing the targeted business network.

### Phase 2: Feature Engineering and NLP Analytics — `yelp-reviews-model-preperation.ipynb`

4. **Vocabulary Capping:** Instantiates a 5,000-feature `TfidfVectorizer` mapping high-signal unigrams and bigrams.
5. **Frequency Boundaries:** Implements strict data pruning boundaries (`min_df=3`, `max_df=0.5`) to eliminate rare typographical noise and ubiquitous terms. Stopword filtering was further extended to remove narrative-reporting verbs and connectors (`said`, `asked`, `could`, `another`, `still`, `later`, etc.) that otherwise dominated the low-rating term list without carrying complaint-specific content — an artifact of unigram TF-IDF capturing narrative structure rather than subject matter. `told` was deliberately retained, since a separate rule-based category flag in the extraction phase depends on it.
6. **Sentiment Delta Computation:** Splits the dataset into poor (<= 2 stars) and excellent (>= 4 stars) cohorts, evaluating feature metrics with a custom delta curve:

   Δ = Mean<sub>Low Rating</sub>(TF-IDF) − Mean<sub>High Rating</sub>(TF-IDF)

### Phase 3: Analytical Visualization — `yelp-reviews-visualization.ipynb`

7. **Vector Alignment:** Sorts the resulting vocabulary matrix by its sentiment delta (Δ).
8. **Diverging Chart Generation:** Plots a dual-color horizontal bar chart map to contrast positive and negative sentiment coefficients.
9. **Asset Generation:** Saves high-resolution, presentation-ready visualizations (`tfidf_diverging_chart.png`) directly to the workspace.

---

## Local Replication and Setup

### Dataset Access

The raw source data used in this pipeline is hosted on Kaggle. Due to GitHub file size limits and repository hygiene, the raw multi-gigabyte files are excluded from this repository.

To run or replicate this pipeline:

1. Authenticate your Kaggle account and navigate to the official [Kaggle Yelp Dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset) page.
2. Download the uncompressed source archive containing the five core `yelp_academic_dataset_*.json` files.
3. Place your raw files into the local workspace using the exact path structure detailed below.

### Prerequisites

Install the required data engineering, analytical, and visualization dependencies:

```
pip install pandas numpy matplotlib scikit-learn nltk
```

### Expected Directory Architecture

Configure your codebase and input files inside your working environment as follows:

```
├── input/
│   └── datasets/
│       ├── organizations/yelp-dataset/
│       │   ├── yelp_academic_dataset_business.json
│       │   ├── yelp_academic_dataset_review.json
│       │   ├── yelp_academic_dataset_user.json
│       │   ├── yelp_academic_dataset_checkin.json
│       │   └── yelp_academic_dataset_tip.json
│       └── micahluftig/stop-words/
│           └── stop_words.csv
├── src/
│   ├── yelp-reviews-data-extraction.ipynb
│   ├── yelp-reviews-model-preperation.ipynb
│   └── yelp-reviews-visualization.ipynb
├── .gitignore
├── README.md
└── tfidf_diverging_chart.png
```

---

## Key NLP Metrics and Discoveries

- **Positive Delta (Δ > 0):** Highlights operational, managerial, or structural failure points. These keywords explicitly expose what triggers negative reviews (e.g., billing disputes, communication breakdowns, long wait times).
- **Negative Delta (Δ < 0):** Isolates key drivers of patient retention and positive reviews, pointing directly to clinical excellence, compassionate staff behaviors, and successful medical outcomes.
