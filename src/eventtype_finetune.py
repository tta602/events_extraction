import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from src.eventtriplet_dataset import EventTripletDataset

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
    def __init__(self, model, tokenizer, train_dataset, event_types, device,
                 batch_size=16, lr=2e-5, epochs=3, max_length=128, checkpoint_dir="./checkpoints"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.event_types = event_types
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir

        self.optimizer = AdamW(self.model.parameters(), lr=lr)

        # build DataLoader
        train_triplet = EventTripletDataset(train_dataset, event_types, tokenizer, max_length)
        self.train_loader = DataLoader(train_triplet, batch_size=batch_size, shuffle=True)

        os.makedirs(checkpoint_dir, exist_ok=True)

    def train(self):
        best_loss = 1e9
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0

            # dùng tqdm ở đây
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)

            for batch in progress_bar:
                self.optimizer.zero_grad()

                sent_emb = self.model(batch["sent_input_ids"].to(self.device),
                                    batch["sent_attention_mask"].to(self.device))
                pos_emb = self.model(batch["pos_input_ids"].to(self.device),
                                    batch["pos_attention_mask"].to(self.device))
                neg_emb = self.model(batch["neg_input_ids"].to(self.device),
                                    batch["neg_attention_mask"].to(self.device))

                pos_score = (sent_emb * pos_emb).sum(dim=1)
                neg_score = (sent_emb * neg_emb).sum(dim=1)

                loss = torch.clamp(1.0 - pos_score + neg_score, min=0).mean()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                # update tqdm bar
                progress_bar.set_postfix({"batch_loss": loss.item()})

            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

            # save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_path = os.path.join(self.checkpoint_dir, "retrieve_best_model")
                self.model.encoder.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
                print(f"Saved best model to {save_path}")