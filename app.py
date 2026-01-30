import streamlit as st
import joblib
import torch

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

from src.tf_idf.preprocessing import clean_text
from src.config import (
    LABEL_MAP,
    MODEL_PATH,
    VECTORIZER_PATH,
    BERT_MODEL_DIR,
    BERT_MAX_LEN
)

# =====================
# Page config
# =====================
st.set_page_config(
    page_title="Sentiment Analysis",
    layout="centered"
)

st.title("💬 Sentiment Analysis App")
st.write("Enter a sentence and choose a model to predict its emotion.")

# =====================
# Model selector
# =====================
model_type = st.selectbox(
    "Choose model:",
    ["TF-IDF (Logistic Regression)", "BERT (DistilBERT)"]
)

# =====================
# Load TF-IDF model
# =====================
@st.cache_resource
def load_tfidf():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


# =====================
# Load BERT model
# =====================
@st.cache_resource
def load_bert():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
    model.to(device)
    model.eval()
    return tokenizer, model, device


# =====================
# User input
# =====================
user_text = st.text_area("Input text:")

if st.button("Predict Emotion"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # -------- TF-IDF --------
        if model_type.startswith("TF-IDF"):
            model, vectorizer = load_tfidf()

            clean = clean_text(user_text)
            vector = vectorizer.transform([clean])
            prediction = model.predict(vector)[0]

            st.success(
                f"Predicted Emotion (TF-IDF): **{LABEL_MAP[prediction]}**"
            )

        # -------- BERT --------
        else:
            tokenizer, model, device = load_bert()

            encoding = tokenizer(
                user_text,
                truncation=True,
                padding="max_length",
                max_length=BERT_MAX_LEN,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = model(
                    input_ids=encoding["input_ids"].to(device),
                    attention_mask=encoding["attention_mask"].to(device)
                )

            label_id = torch.argmax(outputs.logits, dim=1).item()

            st.success(
                f"Predicted Emotion (BERT): **{LABEL_MAP[label_id]}**"
            )
