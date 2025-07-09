import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class SocialMediaDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)
        }

try:
    df = pd.read_csv("training_data.csv")
except FileNotFoundError:
    print("Error: training_data.csv not found. Run preprocess_data.py first.")
    exit(1)

# Verify columns
label_columns = ["hate_speech", "cyberbullying", "incitement_violence", "threat_safety"]
if not all(col in df.columns for col in label_columns):
    print(f"Error: Missing columns in training_data.csv. Expected: {label_columns}, Found: {list(df.columns)}")
    exit(1)

# Clean data
df = df.dropna(subset=["text"])
df = df[df["text"].str.strip() != ""]
df[label_columns] = df[label_columns].fillna(0)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_texts, val_texts = train_df["text"].values, val_df["text"].values
train_labels, val_labels = train_df[label_columns].values, val_df[label_columns].values

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)

train_dataset = SocialMediaDataset(train_texts, train_labels, tokenizer)
val_dataset = SocialMediaDataset(val_texts, val_labels, tokenizer)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = (pred.predictions > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

training_args = TrainingArguments(
    output_dir="./cyber_shield_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
model.save_pretrained("./cyber_shield_model")
tokenizer.save_pretrained("./cyber_shield_model")
print("Model and tokenizer saved to ./cyber_shield_model")