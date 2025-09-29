import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, AutoTokenizer
from torch.optim import AdamW
from tqdm import tqdm
from src.event_argument_dataset import EventArgumentDataset
from src.eventtype_retriever import EventTypeRetriever
from src.utils.data_utils import build_labels, load_json_or_jsonl
from src.utils.device_util import getDeviceInfo

# ---------------------- Config ----------------------
DEVICE = getDeviceInfo()
BART_MODEL = "facebook/bart-base"
MAX_LENGTH = 256
OUTPUT_MAX_LENGTH = 64
BATCH_SIZE = 16
EPOCHS = 1
LR = 3e-5
TOP_K = 3
CONTEXT_PATH = ""
CHECKPOINT_DIR = f"{CONTEXT_PATH}checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

TRAIN_JSON_PATH = f"{CONTEXT_PATH}processing_data/train.json"
VAL_JSON_PATH = f"{CONTEXT_PATH}processing_data/dev.json"
TEST_JSON_PATH = f"{CONTEXT_PATH}processing_data/test.json"
ONTOLOGY_PATH = f"{CONTEXT_PATH}ontoloy/event_role_WIKI_q.json"
LABEL_CACHE_PATH = f"{CONTEXT_PATH}processing_data/event_types.json"

event_types = build_labels(TRAIN_JSON_PATH, LABEL_CACHE_PATH)

# ---------------------- Load tokenizer & model ----------------------
tokenizer = AutoTokenizer.from_pretrained(BART_MODEL)
special_tokens = ["<tgr>"]
tokenizer.add_tokens(special_tokens)

model = BartForConditionalGeneration.from_pretrained(BART_MODEL).to(DEVICE)
model.resize_token_embeddings(len(tokenizer))

retriever = EventTypeRetriever(
    model_name=f"{CHECKPOINT_DIR}/retrieve_best_model",
    device=DEVICE,
    tokenizer=tokenizer,
    event_types=event_types
)

# ---------------------- Load samples ----------------------
train_samples = load_json_or_jsonl(TRAIN_JSON_PATH)
val_samples = load_json_or_jsonl(VAL_JSON_PATH)
test_samples = load_json_or_jsonl(TEST_JSON_PATH)

# ---------------------- Dataset & DataLoader ----------------------
train_dataset = EventArgumentDataset(
    samples=train_samples,
    ontology_path=ONTOLOGY_PATH,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH,
    output_max_length=OUTPUT_MAX_LENGTH,
    topk_event_types=TOP_K,
    retriever=retriever
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = EventArgumentDataset(
    samples=val_samples,
    ontology_path=ONTOLOGY_PATH,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH,
    output_max_length=OUTPUT_MAX_LENGTH,
    topk_event_types=TOP_K,
    retriever=retriever
)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataset = EventArgumentDataset(
    samples=test_samples,
    ontology_path=ONTOLOGY_PATH,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH,
    output_max_length=OUTPUT_MAX_LENGTH,
    topk_event_types=TOP_K,
    retriever=retriever
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------- Hàm evaluate ----------------------
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=OUTPUT_MAX_LENGTH)
            predictions = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
            targets = [tokenizer.decode(t, skip_special_tokens=True) for t in labels]

            all_predictions.extend(predictions)
            all_targets.extend(targets)

    avg_loss = total_loss / len(loader)
    return avg_loss, all_predictions, all_targets

# ---------------------- Training loop + checkpoint ----------------------
optimizer = AdamW(model.parameters(), lr=LR)
best_val_loss = float("inf")
best_epoch = -1

for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"BART Epoch {epoch}"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch} finished. Avg train loss: {total_loss/len(train_loader):.4f}")

    # Evaluate on validation set
    val_loss, val_preds, val_targets = evaluate(model, val_loader, DEVICE)
    print(f"Validation loss after epoch {epoch}: {val_loss:.4f}")

    # Lưu checkpoint tốt nhất
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"bart_best_model_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        # Lưu thông tin best epoch
        with open(os.path.join(CHECKPOINT_DIR, "best_checkpoint.txt"), "w") as f:
            f.write(str(best_epoch))
        print(f"Saved best model checkpoint to {ckpt_path}")

# ---------------------- Load best checkpoint để đánh giá test ----------------------
with open(os.path.join(CHECKPOINT_DIR, "best_checkpoint.txt"), "r") as f:
    best_epoch = int(f.read().strip())
best_ckpt_path = os.path.join(CHECKPOINT_DIR, f"bart_best_model_epoch{best_epoch}.pt")

model.load_state_dict(torch.load(best_ckpt_path))
model.to(DEVICE)
print(f"Loaded best model from epoch {best_epoch}")

# Evaluate on test set
test_loss, test_preds, test_targets = evaluate(model, test_loader, DEVICE)
print(f"Test loss: {test_loss:.4f}")
