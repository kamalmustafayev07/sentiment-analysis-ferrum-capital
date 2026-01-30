import os
import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW

from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from src.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    BERT_MODEL_NAME,
    BERT_MODEL_DIR,
    BERT_MAX_LEN,
    BERT_BATCH_SIZE,
    BERT_EPOCHS,
    BERT_LR,
    NUM_LABELS
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================
# Dataset
# =====================
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=BERT_MAX_LEN,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


# =====================
# Data loading
# =====================
def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, val_df, test_df


def build_dataloader(df, tokenizer, shuffle=False):
    dataset = EmotionDataset(
        df["text"].tolist(),
        df["label"].tolist(),
        tokenizer
    )
    return DataLoader(
        dataset,
        batch_size=BERT_BATCH_SIZE,
        shuffle=shuffle
    )


# =====================
# Training
# =====================
def train_model(train_loader, val_loader):
    tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=NUM_LABELS
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=BERT_LR)
    total_steps = len(train_loader) * BERT_EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    for epoch in range(BERT_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{BERT_EPOCHS}")
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()

            outputs = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                labels=batch["labels"].to(DEVICE)
            )

            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        print("Train loss:", total_loss / len(train_loader))
        evaluate(model, val_loader, "Validation")

    return model, tokenizer


# =====================
# Evaluation
# =====================
def evaluate(model, dataloader, dataset_name):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating ({dataset_name})"):
            outputs = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE)
            )

            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            labels.extend(batch["labels"].numpy())

    print(f"\n--- {dataset_name} Evaluation ---")
    print("Accuracy:", accuracy_score(labels, preds))
    print(classification_report(labels, preds))


# =====================
# Main
# =====================
def main():
    train_df, val_df, test_df = load_data()

    tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL_NAME)

    train_loader = build_dataloader(train_df, tokenizer, shuffle=True)
    val_loader = build_dataloader(val_df, tokenizer)
    test_loader = build_dataloader(test_df, tokenizer)

    model, tokenizer = train_model(train_loader, val_loader)

    evaluate(model, test_loader, "Test")

    os.makedirs(BERT_MODEL_DIR, exist_ok=True)
    model.save_pretrained(BERT_MODEL_DIR)
    tokenizer.save_pretrained(BERT_MODEL_DIR)

    print(f"\nBERT model saved to {BERT_MODEL_DIR}")


if __name__ == "__main__":
    main()
