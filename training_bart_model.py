import json
import torch
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, AutoTokenizer
from torch.optim import AdamW
from tqdm import tqdm
from src.event_argument_dataset import EventArgumentDataset
from src.utils.data_utils import load_json_or_jsonl
from src.utils.device_util import getDeviceInfo  # bạn giữ module này hoặc dùng torch.device

# ---------------------- Config ----------------------
DEVICE = getDeviceInfo()  # hoặc torch.device("cuda" if torch.cuda.is_available() else "cpu")
BART_MODEL = "facebook/bart-base"
MAX_LENGTH = 128
OUTPUT_MAX_LENGTH = 64
BATCH_SIZE = 16
EPOCHS = 1
LR = 3e-5

CONTEXT_PATH = ""
CHECKPOINT_DIR = f"{CONTEXT_PATH}checkpoints"

TRAIN_JSON_PATH = f"{CONTEXT_PATH}processing_data/train.json"
VAL_JSON_PATH = f"{CONTEXT_PATH}processing_data/dev.json"
TEST_JSON_PATH = f"{CONTEXT_PATH}processing_data/test.json"
ONTOLOGY_PATH = f"{CONTEXT_PATH}ontoloy/event_role_WIKI_q.json"

# ---------------------- Load tokenizer & model ----------------------
tokenizer = AutoTokenizer.from_pretrained(BART_MODEL)
special_tokens = ["<tgr>"]
tokenizer.add_tokens(special_tokens)

model = BartForConditionalGeneration.from_pretrained(BART_MODEL).to(DEVICE)
model.resize_token_embeddings(len(tokenizer))

# ---------------------- Load samples ----------------------
train_samples = load_json_or_jsonl(TRAIN_JSON_PATH)
val_samples = load_json_or_jsonl(VAL_JSON_PATH)
test_samples = load_json_or_jsonl(TEST_JSON_PATH)

# ---------------------- Dataset & DataLoader ----------------------
# Sample input_text: sentence: Roadside IED <tgr> kills </tgr> Russian major general in Syria question: Who died?
# Sample target_text: <Victim> general
train_dataset = EventArgumentDataset(
    samples=train_samples,
    ontology_path=ONTOLOGY_PATH,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH,
    output_max_length=OUTPUT_MAX_LENGTH,
    topk_event_types=3,   
    retriever=None     
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------------------- Dataset & DataLoader cho val/test ----------------------
val_dataset = EventArgumentDataset(
    samples=val_samples,
    ontology_path=ONTOLOGY_PATH,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH,
    output_max_length=OUTPUT_MAX_LENGTH,
    topk_event_types=3,
    retriever=None
)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataset = EventArgumentDataset(
    samples=test_samples,
    ontology_path=ONTOLOGY_PATH,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH,
    output_max_length=OUTPUT_MAX_LENGTH,
    topk_event_types=3,
    retriever=None
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

            # Inference
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=OUTPUT_MAX_LENGTH)
            predictions = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
            targets = [tokenizer.decode(t, skip_special_tokens=True) for t in labels]

            all_predictions.extend(predictions)
            all_targets.extend(targets)

    avg_loss = total_loss / len(loader)
    return avg_loss, all_predictions, all_targets

# ---------------------- Training loop + evaluation ----------------------
optimizer = AdamW(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"BART Epoch {epoch+1}"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1} finished. Avg train loss: {total_loss/len(train_loader):.4f}")

    # Evaluate on validation set
    val_loss, val_preds, val_targets = evaluate(model, val_loader, DEVICE)
    print(f"Validation loss after epoch {epoch+1}: {val_loss:.4f}")

# ---------------------- Evaluate on test set ----------------------
test_loss, test_preds, test_targets = evaluate(model, test_loader, DEVICE)
print(f"Test loss: {test_loss:.4f}")

# Inference ví dụ trên câu mới
sentence = "Roadside IED <tgr> kills </tgr> Russian major general in Syria"
top_events = ["Life.Die.Unspecified"]  # hoặc từ retriever
input_text = f"event_types: {' | '.join(top_events)} sentence: {sentence}"
inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)
generated_ids = model.generate(**inputs, max_length=OUTPUT_MAX_LENGTH)
predicted = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("Predicted arguments:", predicted)