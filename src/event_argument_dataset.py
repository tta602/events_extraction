import json
import torch
from torch.utils.data import Dataset

class EventArgumentDataset(Dataset):
    """
    Dataset cho BART giai đoạn 2: extract arguments dựa vào top-k event types
    và ontology (câu hỏi cho mỗi role). Target là <Role> answer hoặc Role [ans] answer.
    """
    def __init__(self, samples, ontology_path, tokenizer,
                 max_length=128, output_max_length=64,
                 topk_event_types=None, retriever=None,
                 role_answer_format="<{}> {}"):  # hoặc "{} [ans] {}"
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.output_max_length = output_max_length
        self.topk_event_types = topk_event_types
        self.retriever = retriever
        self.role_answer_format = role_answer_format

        # Load ontology
        with open(ontology_path, "r", encoding="utf-8") as f:
            self.ontology = json.load(f)

        # Preprocess: expand samples -> sub-samples per role
        self.expanded_samples = []
        for item in self.samples:
            sentence = item["text"]
            if self.retriever and self.topk_event_types:
                top_events = [et for et,_ in self.retriever.retrieve(sentence, topk=self.topk_event_types)]
            else:
                top_events = [item["event_type"]]

            for event_type in top_events:
                if event_type not in self.ontology:
                    continue
                questions = self.ontology[event_type]["questions"]  # role -> question

                # Map roles in sample
                role2answer = {arg["role"]: arg["text"] for arg in item.get("arguments", [])}

                for role, question in questions.items():
                    answer = role2answer.get(role, "none")  # Nếu role không có -> none
                    input_text = f"sentence: {sentence} question: {question}"
                    target_text = self.role_answer_format.format(role, answer)
                    self.expanded_samples.append({
                        "input_text": input_text,
                        "target_text": target_text
                    })

    def __len__(self):
        return len(self.expanded_samples)

    def __getitem__(self, idx):
        sample = self.expanded_samples[idx]

        input_enc = self.tokenizer(
            sample["input_text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        target_enc = self.tokenizer(
            sample["target_text"],
            truncation=True,
            padding="max_length",
            max_length=self.output_max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "labels": target_enc["input_ids"].squeeze(0)
        }
