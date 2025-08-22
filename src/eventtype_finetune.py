import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

class EventRetrieverFineTune(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state.mean(dim=1)   # mean pooling
        embeddings = F.normalize(embeddings, p=2, dim=1)     # normalize để cosine
        return embeddings
    

class EventRetrieverTrainer:
    def __init__(self, model, tokenizer, train_loader, val_loader, event_types, device,
                 batch_size=16, lr=2e-5, epochs=3, max_length=128, checkpoint_dir="./checkpoints"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.event_types = event_types
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir

        self.optimizer = AdamW(self.model.parameters(), lr=lr)

        self.train_loader = train_loader
        self.val_loader   = val_loader

        os.makedirs(checkpoint_dir, exist_ok=True)

    def compute_loss(self, batch):
        sent_emb = self.model(batch["sent_input_ids"].to(self.device),
                              batch["sent_attention_mask"].to(self.device))
        pos_emb = self.model(batch["pos_input_ids"].to(self.device),
                             batch["pos_attention_mask"].to(self.device))
        neg_emb = self.model(batch["neg_input_ids"].to(self.device),
                             batch["neg_attention_mask"].to(self.device))

        pos_score = (sent_emb * pos_emb).sum(dim=1)
        neg_score = (sent_emb * neg_emb).sum(dim=1)

        loss = torch.clamp(1.0 - pos_score + neg_score, min=0).mean()
        return loss

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in loader:
                loss = self.compute_loss(batch)
                total_loss += loss.item()
        return total_loss / len(loader)

    def train(self):
        best_val_loss = 1e9
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0

            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)

            for batch in progress_bar:
                self.optimizer.zero_grad()
                loss = self.compute_loss(batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix({"batch_loss": loss.item()})

            avg_train_loss = total_loss / len(self.train_loader)
            avg_val_loss = self.evaluate(self.val_loader)

            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # save best model on val
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = os.path.join(self.checkpoint_dir, "retrieve_best_model")
                self.model.encoder.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
                print(f"Saved best model to {save_path} (val loss {avg_val_loss:.4f})")