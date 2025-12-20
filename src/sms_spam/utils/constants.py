"""Constants used throughout the SMS spam detection project."""

# Data constants
RANDOM_STATE = 42
TEST_SIZE = 0.2

# TF-IDF defaults
TFIDF_MAX_FEATURES = 3000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95

# Embedding defaults
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32

# Optimization defaults
OPTUNA_N_TRIALS = 100
OPTUNA_METRIC = "recall"

# Cascade thresholds
CASCADE_STAGE1_LOW_THRESHOLD = 0.1  # Below this -> HAM
CASCADE_STAGE1_HIGH_THRESHOLD = 0.9  # Above this -> SPAM
CASCADE_STAGE2_THRESHOLD = 0.5  # Default for Stage 2

# Class labels
CLASS_NAMES = ["ham", "spam"]
LABEL_MAP = {"ham": 0, "spam": 1}
LABEL_MAP_REVERSE = {0: "ham", 1: "spam"}

# Metrics
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
