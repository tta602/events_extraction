import json
import os
from torch.utils.data import Dataset
import torch
from .utils.data_utils import load_json_or_jsonl

class WikiEventsSentenceDataset(Dataset):
    def __init__(self, file_path, tokenizer, label2id, max_length=128, cache_path=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Nếu có cache_path và file tồn tại -> load
        if cache_path and os.path.exists(cache_path):
            print(f"Loading processed dataset from {cache_path}")
            self.samples = load_json_or_jsonl(cache_path)
        else:
            print(f"Processing raw data from {file_path}")
            data = load_json_or_jsonl(file_path)

            self.samples = []
            for doc in data:
                doc_id = doc["doc_id"] 
                sentences = doc["sentences"]
                events = doc.get("event_mentions", [])

                for event in events:
                    sent_idx = event["trigger"]["sent_idx"]
                    sentence_text = sentences[sent_idx][1]

                    trigger_text = event["trigger"]["text"]
                    event_type = event["event_type"]

                    sentence_with_trigger = sentence_text.replace(
                        trigger_text, f"<tgr> {trigger_text} </tgr>"
                    )

                    self.samples.append({
                        "doc_id": doc_id,
                        "sent_idx": sent_idx,
                        "text": sentence_with_trigger,
                        "label": label2id[event_type]
                    })

            # Nếu có cache_path thì lưu lại
            if cache_path:
                with open(cache_path, "w") as f:
                    json.dump(self.samples, f, ensure_ascii=False, indent=2)
                print(f"Saved processed dataset to {cache_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long)
        }
