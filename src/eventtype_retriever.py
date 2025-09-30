from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch
import torch.nn.functional as F

class EventTypeRetriever:
    def __init__(self, model_name, tokenizer, device, event_types, max_length=256):
        self.device = device
        self.tokenizer = tokenizer
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.resize_token_embeddings(len(tokenizer))
        self.model.eval()

        self.max_length = max_length

        # Encode tất cả event types (ontology)
        self.event_types = event_types  # list[str]
        self.event_embeddings = self.encode_texts(event_types)

    def encode_texts(self, texts, batch_size=16):
        all_embeds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**enc)
                # Mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeds.append(embeddings.to(self.device)) 
        return torch.cat(all_embeds, dim=0)

    def retrieve(self, sentence, topk=3):
        # Encode sentence
        enc = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**enc)
            query_emb = outputs.last_hidden_state.mean(dim=1)
            query_emb = F.normalize(query_emb, p=2, dim=1)

        # Cosine similarity
        scores = torch.matmul(query_emb, self.event_embeddings.T)
        topk_scores, topk_idx = torch.topk(scores, k=topk, dim=1)

        results = [(self.event_types[i], topk_scores[0, j].item()) for j, i in enumerate(topk_idx[0])]
        return results
