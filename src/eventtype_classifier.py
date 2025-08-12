import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
import torch
from tqdm import tqdm
from .utils.data_utils import load_json_or_jsonl

class EventTypeClassifier:
    def __init__(self, device, checkpoint_dir, model_name, num_labels, label2id, id2label, lr=2e-5):
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        ).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def train(self, train_loader, val_loader, epochs=3):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} - Training"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")

            self.evaluate(val_loader)

            # Save checkpoint
            checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch+1}.pt")
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    def evaluate(self, data_loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].size(0)

        acc = correct / total
        print(f"Validation Accuracy: {acc:.4f}")