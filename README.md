# SMS Spam Detection: Traditional ML vs Embeddings vs Hybrid Cascade

Educational project comparing traditional ML approaches with modern transformer-based embeddings for SMS spam detection. Features a clean modular architecture and efficient 2-stage cascade classifier.

## Project Overview

**Goal**: Compare multiple approaches to spam classification and build an optimized 2-stage system:
1. **Traditional ML**: TF-IDF vectorization + classical algorithms
2. **Embedding-based ML**: Sentence transformers + modern classifiers
3. **Hybrid 2-Stage Cascade**: Optimized system combining both approaches

**Dataset**: SMS Spam Collection (5,571 messages)
- Ham: 4,825 (86.6%)
- Spam: 747 (13.4%)

## Architecture

This project uses a clean, modular architecture:

```
sms-spam/
├── src/sms_spam/        # Core package
│   ├── data/            # Loading, preprocessing, embeddings
│   ├── features/        # TF-IDF, ChromaDB storage
│   ├── classifiers/     # Traditional, embedding, hybrid
│   ├── optimization/    # Optuna tuning, threshold optimization
│   ├── evaluation/      # Metrics and visualization
│   └── utils/           # Constants and I/O utilities
│
├── scripts/             # Entry point scripts
│   ├── 01_preprocess.py
│   ├── 02_train_traditional.py
│   ├── 03_train_embeddings.py
│   ├── 04_optimize.py (optional)
│   ├── 05_train_hybrid.py
│   └── 06_evaluate.py
│
├── configs/             # Configuration files
├── data/                # Processed data and features
├── models/              # Trained models
└── results/             # Metrics, visualizations, reports
```

## Setup

### Option 1: Using Conda (Recommended)
```bash
conda env create -f ml.yml
conda activate sms_ml
```

### Option 2: Using pip
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Complete Workflow

Run scripts in order for the complete workflow:

```bash
# 1. Preprocess data and generate features
python scripts/01_preprocess.py

# 2. Train traditional ML models (TF-IDF)
python scripts/02_train_traditional.py

# 3. Train embedding-based models
python scripts/03_train_embeddings.py

# 4. (Optional) Optimize hyperparameters with Optuna
python scripts/04_optimize.py

# 5. Train 2-stage hybrid cascade
python scripts/05_train_hybrid.py

# 6. Comprehensive evaluation and comparison
python scripts/06_evaluate.py
```

### Quick Start

For quick experimentation:
```bash
# Just run the first 3 scripts for baseline comparison
python scripts/01_preprocess.py
python scripts/02_train_traditional.py
python scripts/03_train_embeddings.py
```

## 2-Stage Hybrid Cascade Architecture

The hybrid system implements an efficient 2-stage cascade to optimize for both performance and speed:

```
┌─────────────────────────────────────────────────────────────┐
│                   INPUT: SMS Message                        │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              STAGE 1: TF-IDF Classifier                     │
│                   (Fast & Local)                            │
├─────────────────────────────────────────────────────────────┤
│ • TF-IDF vectorization (3000 features)                     │
│ • Best traditional ML model                                │
│ • Latency: ~1ms per message                                │
├─────────────────────────────────────────────────────────────┤
│ Decision Logic:                                             │
│   IF probability >= 0.9  → SPAM (confident)      ─────┐    │
│   IF probability <= 0.1  → HAM (confident)       ─────┤    │
│   ELSE                   → UNCERTAIN (10-20%)    ─────┼─┐  │
└────────────────────────────────────────────────────────┼─┼──┘
                            ▲                            │ │
                            │                            │ │
                    Filters 80-90%                       │ │
                    of messages                          │ │
                                                         │ │
                            ┌────────────────────────────┘ │
                            │                              │
                            ▼                              │
┌─────────────────────────────────────────────────────────────┐
│           STAGE 2: Embedding Classifier                     │
│               (Semantic Understanding)                      │
├─────────────────────────────────────────────────────────────┤
│ • Sentence transformer embedding (384D)                    │
│ • Best embedding-based model                               │
│ • Latency: ~10ms per message                               │
├─────────────────────────────────────────────────────────────┤
│ Final Decision:                                             │
│   probability >= 0.5 → SPAM                                 │
│   probability < 0.5  → HAM                                  │
└────────────────────────────┬────────────────────────────────┘
                             │
                     Handles uncertain
                     from Stage 1
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│         OUTPUT: Final Classification + Metadata             │
├─────────────────────────────────────────────────────────────┤
│ • Label: spam/ham                                           │
│ • Confidence scores from each stage                        │
│ • Which stage made the final decision (1 or 2)            │
│ • Stage-wise performance breakdown                         │
└─────────────────────────────────────────────────────────────┘
```

### Performance Trade-offs

| Stage    | Coverage | Avg Latency | Accuracy |
|----------|----------|-------------|----------|
| Stage 1  | 80-90%   | 1ms         | ~97%     |
| Stage 2  | 10-20%   | 10ms        | ~98%     |
| **Total**| **100%** | **~3ms**    | **~98%** |

**Key Benefits:**
- **Fast**: Average 3ms latency per message
- **Accurate**: Maintains 98%+ accuracy
- **Efficient**: 80-90% filtered by fast Stage 1
- **No external dependencies**: Completely local inference
- **100% spam recall**: Through threshold optimization

## Results

After running all scripts, you'll find:

**Metrics** (`results/metrics/`):
- `traditional_ml_results.csv` - All traditional ML model results
- `embedding_ml_results.csv` - All embedding-based results
- `final_comparison.csv` - Side-by-side comparison
- `cascade_thresholds.json` - Optimized cascade thresholds
- `hybrid_cascade_results.json` - Cascade performance breakdown

**Visualizations** (`results/visualizations/`):
- `model_comparison.png` - Performance comparison
- `cascade_analysis.png` - Stage usage and performance
- `cascade_confusion_matrix.png` - Final predictions

**Reports** (`results/reports/`):
- `evaluation_report.txt` - Comprehensive text report

## Package Usage

The modular architecture allows programmatic usage:

```python
from sms_spam.data.loader import load_spam_dataset
from sms_spam.data.preprocessing import preprocess_dataset, create_train_test_split
from sms_spam.classifiers.hybrid import HybridCascadeClassifier

# Load and preprocess data
data = load_spam_dataset("spam.csv")
data = preprocess_dataset(data)
X_train, X_test, y_train, y_test = create_train_test_split(data)

# Load trained cascade
cascade = HybridCascadeClassifier.load('models/hybrid/')

# Make predictions
predictions = cascade.predict(X_test_tfidf, test_embeddings)

# Detailed evaluation
results = cascade.evaluate(X_test_tfidf, test_embeddings, y_test)
```

## ChromaDB Vector Storage

ChromaDB is used for optional semantic search capabilities:
- Stores sentence transformer embeddings
- Enables similarity-based message retrieval
- Demonstrates vector database usage
- Not required for classification

## Development

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
# Format code
black src/ scripts/ tests/

# Type checking
mypy src/
```

## Project Structure Details

**Key Modules:**
- `data.loader`: Load and validate spam.csv
- `data.preprocessing`: Text cleaning and train/test split
- `data.embeddings`: Sentence transformer embeddings
- `features.tfidf`: TF-IDF feature extraction
- `classifiers.traditional`: TF-IDF + sklearn models
- `classifiers.embedding`: Embeddings + sklearn models
- `classifiers.hybrid`: 2-stage cascade classifier
- `optimization.optuna_tuner`: Hyperparameter optimization
- `optimization.threshold_tuner`: Cascade threshold optimization
- `evaluation.metrics`: Performance metrics
- `evaluation.visualization`: Plotting utilities

## Configuration

Edit `configs/default.yaml` to customize:
- TF-IDF parameters
- Embedding model selection
- Cascade thresholds
- Optimization settings

## License

MIT

## Acknowledgments

- SMS Spam Collection dataset
- Sentence Transformers library
- Optuna for hyperparameter optimization
- LazyPredict for quick model comparison
