# Corpus Analysis
4NL3 – Assignment 2

Author: Mahad Ahmed

## Overview

This project analyzes lexical and thematic differences between 19th and 20th century English literature using classical Natural Language Processing techniques. The corpus is composed of novels sourced from Project Gutenberg and segmented into chapters to satisfy the assignment requirement of 100+ documents per category.

The analysis includes:
- Dataset statistics
- Bag-of-Words representations (raw counts and TF-IDF)
- Naive Bayes word association analysis
- Topic modeling using Latent Dirichlet Allocation (LDA)

## Setup

Create and activate a virtual environment, then install dependencies.

```
python -m venv .venv
```
```
source .venv/bin/activate    (Linux/macOS)
```
```
.venv\Scripts\activate       (Windows)
```

```
pip install numpy scikit-learn gensim
```

Ensure that raw novels are placed in the data directory before running the scripts.

## Running the Project

All scripts should be run from the project root directory.

### Step 1: Split novels into chapters (only needed once)

```
python split_chapters.py
```

This populates the chapters/19th and chapters/20th directories.

### Step 2: Compute dataset statistics

```
python dataset_stats.py
```

Outputs the number of documents and average token counts per century.

### Step 3: Bag-of-Words analysis (raw counts)

```
python run_bow_counts.py
```

Prints the most frequent words for each century using raw counts.

### Step 4: Naive Bayes analysis (TF-IDF)

```
python run_naive_bayes.py
```

Computes log-likelihood ratios to identify words most strongly associated with each century.

### Step 5: Topic modeling with LDA

```
python run_topic_modeling.py
```

Generates topic modeling outputs in the outputs directory:
- lda_topics.txt
- lda_topics.csv
- lda_avg_topic_dist.csv

### Step 6: Experimental comparisons

```
python run_experiments.py
```

Runs multiple preprocessing and feature representation configurations for comparison.

## Notes on Preprocessing

The preprocessing pipeline supports:
- Stopword removal using scikit-learn’s English stopword list
- Removal of common character names
- Removal of contraction fragments (e.g., t, ll, ve)

These options are configurable through the Preprocessor class and were used to study their impact on downstream analysis.

## Attribution

AI assistance was used during the development of this project in the following ways:

- Assistance with implementing topic modeling using gensim, as this library was new to me
- Assistance with designing and implementing split_chapters.py
- Assistance with structuring and writing this README file

Number of Prompts used: 12
4.32g*12  = 51.84g
