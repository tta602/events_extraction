import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os
from src.utils.device_util import getDeviceInfo
from src.utils.data_utils import build_label_maps
from draft_code.eventtype_classifier import EventTypeClassifier
from src.wikievents_dataset import WikiEventsSentenceDataset

device = getDeviceInfo()
print(f"Device info::: {device}")

MODEL_NAME = "roberta-base"
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 1

CONTEXT_PATH = ""
CHECKPOINT_DIR = f"{CONTEXT_PATH}checkpoints"

TRAIN_JSON_PATH = f"{CONTEXT_PATH}data/train.jsonl"
VAL_JSON_PATH = f"{CONTEXT_PATH}data/dev.jsonl"
TEST_JSON_PATH = f"{CONTEXT_PATH}data/test.jsonl"

LABEL_CACHE_PATH = f"{CONTEXT_PATH}processing_data/label_maps.json"
TRAIN_CACHE_PATH = f"{CONTEXT_PATH}processing_data/train.json"
VAL_CACHE_PATH = f"{CONTEXT_PATH}processing_data/dev.json"
TEST_CACHE_PATH = f"{CONTEXT_PATH}processing_data/test.json"

label2id, id2label = build_label_maps(TRAIN_JSON_PATH, LABEL_CACHE_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
special_tokens = ["<tgr>"]
tokenizer.add_tokens(special_tokens)


# Dataset & Dataloader
#file_path, tokenizer, label2id, max_length=128, cache_path=None
train_dataset = WikiEventsSentenceDataset(TRAIN_JSON_PATH, tokenizer, label2id, MAX_LENGTH, TRAIN_CACHE_PATH)
val_dataset = WikiEventsSentenceDataset(VAL_JSON_PATH, tokenizer, label2id, MAX_LENGTH, VAL_CACHE_PATH)
test_dataset = WikiEventsSentenceDataset(TEST_JSON_PATH, tokenizer, label2id, MAX_LENGTH, TEST_CACHE_PATH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Model device, checkpoint_dir, model_name, num_labels, label2id, id2label, lr=2e-5
classifier = EventTypeClassifier(device, CHECKPOINT_DIR, MODEL_NAME, len(label2id), label2id, id2label, lr=LEARNING_RATE)
classifier.model.resize_token_embeddings(len(tokenizer))  # resize embeddings cho token mới

classifier.train(train_loader=train_loader, val_loader=val_loader, epochs=EPOCHS)

classifier.evaluate(test_loader)
