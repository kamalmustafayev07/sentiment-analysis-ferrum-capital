import os
import torch

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

from src.config import LABEL_MAP, BERT_MODEL_DIR


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load artifacts once
tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
model.to(DEVICE)
model.eval()


def predict_emotion(text: str) -> str:
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"].to(DEVICE),
            attention_mask=encoding["attention_mask"].to(DEVICE)
        )

    label_id = torch.argmax(outputs.logits, dim=1).item()
    return LABEL_MAP[label_id]


if __name__ == "__main__":
    while True:
        text = input("\nEnter a sentence (or 'exit'): ")
        if text.lower() == "exit":
            break

        emotion = predict_emotion(text)
        print("Predicted emotion:", emotion)
