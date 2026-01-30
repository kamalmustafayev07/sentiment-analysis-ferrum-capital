import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from preprocessing import clean_text

from src.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    MODEL_PATH,
    VECTORIZER_PATH
)


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, val_df, test_df


def preprocess_datasets(train_df, val_df, test_df):
    train_df["clean_text"] = train_df["text"].apply(clean_text)
    val_df["clean_text"] = val_df["text"].apply(clean_text)
    test_df["clean_text"] = test_df["text"].apply(clean_text)
    return train_df, val_df, test_df


def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 3),
        min_df=1,          
        max_df=0.95,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train_tfidf, y_train)
    return model, vectorizer


def evaluate(model, vectorizer, X, y, dataset_name):
    X_tfidf = vectorizer.transform(X)
    y_pred = model.predict(X_tfidf)

    print(f"\n--- {dataset_name} Evaluation ---")
    print("Accuracy:", accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))


def main():
    train_df, val_df, test_df = load_data()
    train_df, val_df, test_df = preprocess_datasets(train_df, val_df, test_df)

    model, vectorizer = train_model(
        train_df["clean_text"],
        train_df["label"]
    )

    evaluate(model, vectorizer, val_df["clean_text"], val_df["label"], "Validation")
    evaluate(model, vectorizer, test_df["clean_text"], test_df["label"], "Test")

    # save artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("\nModel and vectorizer saved successfully.")


if __name__ == "__main__":
    main()
