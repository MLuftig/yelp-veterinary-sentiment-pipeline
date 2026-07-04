# Yelp Veterinary Data Pipeline and NLP Sentiment Mining

A memory-efficient data engineering, text preprocessing, and sentiment analysis pipeline built to ingest large-scale datasets, extract industry-specific subsets, and isolate structural drivers of customer satisfaction.

![Sentiment Divergence Chart](tfidf_diverging_chart.png)

---

## Executive Summary

This project applies NLP and sentiment analysis to over 33,000 veterinary-specific reviews extracted from the multi-gigabyte Yelp Open Dataset, using TF-IDF differencing to isolate the specific language that separates satisfied clients from dissatisfied ones.

- **Business Goal:** Give veterinary practices a data-driven view of what actually drives client satisfaction and complaints, beyond star ratings alone.
- **Key Result:** Billing and communication breakdowns (e.g., terms like "charged," "rude," "told," "money") were the dominant drivers of negative reviews, while staff compassion and expertise ("knowledgeable," "compassionate," "highly recommend") drove positive ones.
- **Actionable Recommendation:** Practices should prioritize front-desk communication and billing transparency as retention levers — these surfaced as stronger differentiators than clinical outcomes alone.

## Technical Stack and Dependencies

* **Language:** Python 3.x
* **Core Libraries:** pandas, numpy, matplotlib, re, nltk, scikit-learn
* **Environment:** Optimized for Kaggle and Jupyter Notebook environments

---

## Core Project Architecture

* **Memory-Optimized Data Pipeline:** Utilizes chunked JSON streaming (`chunksize=100000`) to parse and filter millions of rows without triggering Out-Of-Memory (OOM) errors.
* **Relational ID Slicing:** Leverages high-performance set-based indexing for O(1) lookups to dynamically isolate user interactions (reviews, tips, check-ins) tied directly to targeted businesses.
* **Text Preprocessing Pipeline:** Normalizes text data by eliminating character noise (punctuation, case, numbers) and removing standard English and domain-specific stopwords.
* **N-Gram TF-IDF Vectorization:** Constructs high-dimensional sparse matrices using unigrams and bigrams (`ngram_range=(1,2)`) to preserve multi-word context.
* **Statistical Sentiment Differencing:** Computes a custom mathematical delta vector between average high-star and low-star text reviews.
* **Analytical Visualizations:** Generates a custom horizontal diverging bar chart plotting statistical keyword variance to isolate operational vulnerabilities from consumer praise.

---

## End-to-End Pipeline Workflow

### Phase 1: Data Isolation and Ingestion
1. **Target Identification:** Scans the core Yelp business dataset to isolate entities explicitly categorized under 'Veterinarian'.
2. **Relational Extraction:** Uses the unique target IDs to slice secondary datasets (`review.json`, `user.json`, `checkin.json`, `tip.json`), filtering out irrelevant entries stream-by-stream.
3. **Structured Storage:** Exports five localized, small-footprint CSV files representing the targeted business network.

### Phase 2: Feature Engineering and NLP Analytics
4. **Vocabulary Capping:** Instantiates a 5,000-feature `TfidfVectorizer` mapping high-signal unigrams and bigrams.
5. **Frequency Boundaries:** Implements strict data pruning boundaries (`min_df=3`, `max_df=0.5`) to eliminate rare typographical noise and ubiquitous terms.
6. **Sentiment Delta Computation:** Splits the dataset into poor (≤ 2 stars) and excellent (≥ 4 stars) cohorts, then computes a delta score for each term:

   `Δ = (avg. TF-IDF score in low-rating reviews) − (avg. TF-IDF score in high-rating reviews)`

   A positive Δ means the term appears more in negative reviews; a negative Δ means it appears more in positive reviews.

### Phase 3: Analytical Visualization
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
```bash
pip install pandas numpy matplotlib scikit-learn nltk
```

### Expected Directory Architecture
Configure your codebase and input files inside your working environment as follows:
```text
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

* **Positive Delta ($\Delta > 0$):** Highlights operational, managerial, or structural failure points. These keywords explicitly expose what triggers negative reviews (e.g., long wait times, billing discrepancies).
* **Negative Delta ($\Delta < 0$):** Isolates key drivers of patient retention and positive reviews, pointing directly to clinical excellence, compassionate staff behaviors, and successful medical outcomes.
