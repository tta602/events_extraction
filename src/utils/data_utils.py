from pathlib import Path
import json
import os

def load_json_or_jsonl(path):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        else:
            return json.load(f)

def build_label_maps(json_path, cache_path="label_maps.json"):
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded label maps from {cache_path}")
        return data["label2id"], data["id2label"]

    raw_data = load_json_or_jsonl(json_path)
    labels = {ev["event_type"] for doc in raw_data for ev in doc.get("event_mentions", [])}

    label2id = {label: i for i, label in enumerate(sorted(labels))}
    id2label = {i: label for label, i in label2id.items()}

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)
    print(f"Saved label maps to {cache_path}")
    return label2id, id2label