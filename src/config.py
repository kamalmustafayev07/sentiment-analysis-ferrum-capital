import os

# =====================
# Labels
# =====================
LABEL_MAP = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# =====================
# Base paths
# =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# =====================
# Dataset paths
# =====================
TRAIN_PATH = os.path.join(DATA_DIR, "training.csv")
VAL_PATH = os.path.join(DATA_DIR, "validation.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

# =====================
# Model tf-idf artifacts
# =====================
MODEL_PATH = os.path.join(MODEL_DIR,"tf_idf", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR,"tf_idf", "tfidf_vectorizer.pkl")

# =====================
# BERT paths & params
# =====================
BERT_MODEL_NAME = "distilbert-base-uncased"

BERT_MODEL_DIR = os.path.join(MODEL_DIR, "bert", "distilbert")

BERT_MAX_LEN = 128
BERT_BATCH_SIZE = 16
BERT_EPOCHS = 3
BERT_LR = 2e-5
NUM_LABELS = 6