import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import os
from src.eventtriplet_dataset import EventTripletDataset
from src.eventtype_finetune import EventRetrieverFineTune, EventRetrieverTrainer
from src.utils.device_util import getDeviceInfo
from src.utils.data_utils import build_label_maps, build_labels
from src.eventtype_retriever import EventTypeRetriever
from src.wikievents_dataset import WikiEventsSentenceDataset

device = getDeviceInfo()
print(f"Device info::: {device}")

MODEL_NAME = "roberta-base"

MAX_LENGTH = 128
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
EPOCHS = 1

CONTEXT_PATH = ""
CHECKPOINT_DIR = f"{CONTEXT_PATH}checkpoints"

TRAIN_JSON_PATH = f"{CONTEXT_PATH}data/train.jsonl"
VAL_JSON_PATH = f"{CONTEXT_PATH}data/dev.jsonl"
TEST_JSON_PATH = f"{CONTEXT_PATH}data/test.jsonl"

LABEL_CACHE_PATH = f"{CONTEXT_PATH}processing_data/event_types.json"
TRAIN_CACHE_PATH = f"{CONTEXT_PATH}processing_data/train.json"
VAL_CACHE_PATH = f"{CONTEXT_PATH}processing_data/dev.json"
TEST_CACHE_PATH = f"{CONTEXT_PATH}processing_data/test.json"

#get event type
event_types = build_labels(TRAIN_JSON_PATH, LABEL_CACHE_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
special_tokens = ["<tgr>"]
tokenizer.add_tokens(special_tokens)

# Dataset & Dataloader
#file_path, tokenizer, label2id, max_length=128, cache_path=None
train_dataset = WikiEventsSentenceDataset(TRAIN_JSON_PATH, tokenizer, MAX_LENGTH, TRAIN_CACHE_PATH)
val_dataset = WikiEventsSentenceDataset(VAL_JSON_PATH, tokenizer, MAX_LENGTH, VAL_CACHE_PATH)
test_dataset = WikiEventsSentenceDataset(TEST_JSON_PATH, tokenizer, MAX_LENGTH, TEST_CACHE_PATH)

train_triplet_dataset = EventTripletDataset(train_dataset, event_types, tokenizer, MAX_LENGTH)
val_triplet_dataset = EventTripletDataset(val_dataset, event_types, tokenizer, MAX_LENGTH)
test_triplet_dataset = EventTripletDataset(test_dataset, event_types, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_triplet_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_triplet_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_triplet_dataset, batch_size=BATCH_SIZE, shuffle=False)

# model_name, device, event_types
sentence = "Roadside IED <tgr> kills </tgr> Russian major general in Syria"
top_k = 3

retriever = EventTypeRetriever(model_name=MODEL_NAME, device=device, event_types=event_types, tokenizer=tokenizer, max_length=MAX_LENGTH)
print(retriever.retrieve(sentence, topk=top_k))

model = EventRetrieverFineTune(MODEL_NAME)
model = EventTypeRetriever(
    model_name=f"{CHECKPOINT_DIR}/retrieve_best_model",
    device=device,
    tokenizer=AutoTokenizer.from_pretrained(f"{CHECKPOINT_DIR}/retrieve_best_model"),
    event_types=event_types,
    max_length=MAX_LENGTH
)
model.encoder.resize_token_embeddings(len(tokenizer)) 

trainer = EventRetrieverTrainer(
    model = model,
    tokenizer = tokenizer,
    train_loader=train_loader,  
    val_loader = val_loader,
    event_types = event_types,
    device = device,
    batch_size = BATCH_SIZE,
    lr = LEARNING_RATE,
    epochs = EPOCHS,
    checkpoint_dir = CHECKPOINT_DIR
)

trainer.train()

avg_test_loss = trainer.test(test_loader)

retriever = EventTypeRetriever(
    model_name=f"{CHECKPOINT_DIR}/retrieve_best_model",
    device=device,
    tokenizer=tokenizer,
    event_types=event_types,
    max_length=MAX_LENGTH
)

print(retriever.retrieve(sentence, topk=top_k))
