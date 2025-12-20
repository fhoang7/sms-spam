# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SMS spam detection using modular architecture. Compares traditional ML (TF-IDF) against embedding-based classification, with an optimized 2-stage hybrid cascade system.

## Architecture

**Modular Package Structure** (`src/sms_spam/`):
- **data/**: Dataset loading, preprocessing, embedding generation
- **features/**: TF-IDF extraction, ChromaDB vector storage
- **classifiers/**: Traditional, embedding, and 2-stage hybrid classifiers
- **optimization/**: Optuna tuning and threshold optimization
- **evaluation/**: Metrics calculation and visualization
- **utils/**: Constants and I/O utilities

**Entry Point Scripts** (`scripts/`):
- 01_preprocess.py → 02_train_traditional.py → 03_train_embeddings.py
- 04_optimize.py (optional) → 05_train_hybrid.py → 06_evaluate.py

## Environment Setup

```bash
# Create and activate conda environment
conda env create -f ml.yml
conda activate sms_ml
```

Environment includes: Python 3.10, scikit-learn, optuna, pandas, sentence-transformers, chromadb, lazypredict, pyyaml

## Data Architecture

**Dataset**: `spam.csv` (gitignored, must be provided)
- **CRITICAL**: Must be read with `encoding='latin'` (latin-1), NOT UTF-8
- Contains columns: `v1` (label: ham/spam), `v2` (message text)

**Directory Structure**:
```
data/
├── raw/spam.csv              # Original dataset
├── processed/                # Preprocessed text (CSV, NPY)
├── features/                 # TF-IDF (NPZ), embeddings (NPY)
├── vectorizers/              # Fitted TF-IDF vectorizer (PKL)
└── chromadb/                 # Vector storage (optional)

models/
├── traditional/              # TF-IDF-based models
│   ├── baseline/
│   └── optimized/
├── embedding/                # Embedding-based models
│   ├── baseline/
│   └── optimized/
└── hybrid/                   # 2-stage cascade

results/
├── metrics/                  # CSV, JSON performance results
├── visualizations/           # PNG plots
└── reports/                  # Text reports
```

## Key Code Patterns

### 1. Data Loading
```python
from sms_spam.data.loader import load_spam_dataset, validate_dataset

data = load_spam_dataset("spam.csv")  # Uses latin-1 encoding
data = validate_dataset(data)
```

### 2. Preprocessing
```python
from sms_spam.data.preprocessing import (
    preprocess_dataset,
    create_train_test_split
)

data = preprocess_dataset(data, verbose=True)
X_train, X_test, y_train, y_test = create_train_test_split(data)
```

### 3. Feature Extraction
```python
from sms_spam.features.tfidf import extract_and_save_tfidf_features
from sms_spam.data.embeddings import generate_and_save_embeddings

# TF-IDF
X_train_tfidf, X_test_tfidf, featurizer = extract_and_save_tfidf_features(
    X_train, X_test
)

# Embeddings
train_embeddings, test_embeddings = generate_and_save_embeddings(
    X_train, X_test
)
```

### 4. Training Classifiers
```python
from sms_spam.classifiers.traditional import (
    run_lazypredict_comparison,
    TraditionalMLClassifier
)

# Compare models
models_df, predictions = run_lazypredict_comparison(
    X_train_tfidf, X_test_tfidf, y_train, y_test
)

# Train best model
classifier = TraditionalMLClassifier(best_sklearn_model, model_name)
classifier.fit(X_train_tfidf, y_train)
classifier.evaluate(X_test_tfidf, y_test)
```

### 5. 2-Stage Hybrid Cascade
```python
from sms_spam.classifiers.hybrid import HybridCascadeClassifier

cascade = HybridCascadeClassifier(
    stage1_model=traditional_classifier,
    stage2_model=embedding_classifier,
    stage1_low_threshold=0.1,
    stage1_high_threshold=0.9,
    stage2_threshold=0.5
)

# Evaluate with detailed breakdown
results = cascade.evaluate(X_test_tfidf, test_embeddings, y_test)
```

## Configuration

Edit `configs/default.yaml` to customize:
- TF-IDF parameters (max_features, ngram_range, etc.)
- Embedding model selection
- Cascade thresholds
- Optimization settings (n_trials, metric)

## Testing

Minimal test structure in `tests/`:
- Basic unit tests for core utilities
- Integration test for full pipeline
- Run with: `pytest tests/`

## Important Notes

1. **No LLM/API Dependencies**: This project uses only local ML models
2. **ChromaDB is Optional**: Used for semantic search demos, not required for classification
3. **Encoding is Critical**: Always use `encoding='latin'` for spam.csv
4. **2-Stage Cascade**: Stage 1 (TF-IDF) → Stage 2 (Embeddings), no third stage
5. **Modular Design**: All logic in src/sms_spam/, scripts are thin orchestration

## Common Tasks

**Add a new classifier**:
1. Inherit from `sms_spam.classifiers.base.BaseClassifier`
2. Implement `fit()`, `predict()`, `predict_proba()`
3. Add to appropriate module (traditional.py or embedding.py)

**Modify cascade thresholds**:
1. Edit `configs/default.yaml` cascade section
2. Or use `optimization.threshold_tuner.optimize_cascade_thresholds()`

**Add new features**:
1. Create new module in `features/`
2. Follow pattern of `tfidf.py` (class with fit/transform/save/load)

## Workflow Summary

```bash
# Full pipeline
scripts/01_preprocess.py       # Data → Features
scripts/02_train_traditional.py # TF-IDF models
scripts/03_train_embeddings.py  # Embedding models
scripts/04_optimize.py          # Optuna tuning (optional)
scripts/05_train_hybrid.py      # 2-stage cascade
scripts/06_evaluate.py          # Final comparison
```

Each script is self-contained and can be run independently after preprocessing.
