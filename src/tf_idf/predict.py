import os
import joblib

from src.tf_idf.preprocessing import clean_text

from src.config import LABEL_MAP, MODEL_PATH, VECTORIZER_PATH

def predict_emotion(text: str) -> str:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    clean = clean_text(text)
    vector = vectorizer.transform([clean])
    label = model.predict(vector)[0]

    return LABEL_MAP[label]


if __name__ == "__main__":
    while True:
        text = input("\nEnter a sentence (or 'exit'): ")
        if text.lower() == "exit":
            break

        emotion = predict_emotion(text)
        print("Predicted emotion:", emotion)

