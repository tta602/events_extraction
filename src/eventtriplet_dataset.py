import random
import torch

class EventTripletDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, all_event_types, tokenizer, max_length=128):
        self.base_dataset = base_dataset
        self.event_types = all_event_types
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        sent_text = sample["text"]             
        pos_type = sample["event_type"]        

        # chọn 1 negative event type
        neg_type = random.choice([et for et in self.event_types if et != pos_type])

        # tokenize sentence
        sent_enc = self.tokenizer(
            sent_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        pos_enc = self.tokenizer(
            pos_type,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        neg_enc = self.tokenizer(
            neg_type,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            # sentence
            "sent_input_ids": sent_enc["input_ids"].squeeze(0),
            "sent_attention_mask": sent_enc["attention_mask"].squeeze(0),

            # positive event type
            "pos_input_ids": pos_enc["input_ids"].squeeze(0),
            "pos_attention_mask": pos_enc["attention_mask"].squeeze(0),

            # negative event type
            "neg_input_ids": neg_enc["input_ids"].squeeze(0),
            "neg_attention_mask": neg_enc["attention_mask"].squeeze(0),
        }
