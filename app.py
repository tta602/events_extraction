# app.py
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BartForConditionalGeneration, AutoTokenizer
import nltk
import re
from fastapi.middleware.cors import CORSMiddleware

# ---- utils from your repo ----
from src.eventtype_retriever import EventTypeRetriever
from src.utils.data_utils import build_labels, load_json_or_jsonl

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Where your saved checkpoints / tokenizers are
CHECKPOINT_DIR = Path("checkpoints")        # <-- sửa nếu khác
RETRIEVER_CKPT_DIR = CHECKPOINT_DIR / "retrieve_best_model"
# BART best checkpoint file is expected like: bart_best_model_epoch{n}.pt
BEST_CHECKPOINT_TXT = CHECKPOINT_DIR / "best_checkpoint.txt"

# Base model names (fallback)
BART_BASE = "facebook/bart-base"
RETRIEVER_BASE = "roberta-base"

# Tokenizer/model params (phải khớp lúc train)
MAX_LENGTH = 256
OUTPUT_MAX_LENGTH = 64
TOP_K = 1

# Files
TRAIN_LABELS_PATH = "processing_data/train.json"
EVENT_TYPES_CACHE = "processing_data/event_types.json"
ONTOLOGY_PATH = "ontoloy/event_role_WIKI_q.json"  # sửa đúng path của bạn

# ---- ensure nltk punkt available ----
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

from nltk.tokenize import sent_tokenize


# ---------------- Helpers ----------------
def find_best_bart_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Return path to best bart checkpoint saved as bart_best_model_epoch{n}.pt using best_checkpoint.txt."""
    if not checkpoint_dir.exists():
        return None
    txt = checkpoint_dir / "best_checkpoint.txt"
    if not txt.exists():
        # fallback: try to find any file matching bart_best_model_epoch*.pt
        for f in checkpoint_dir.glob("bart_best_model_epoch*.pt"):
            return f
        return None
    try:
        epoch = int(txt.read_text().strip())
    except Exception:
        return None
    candidate = checkpoint_dir / f"bart_best_model_epoch{epoch}.pt"
    return candidate if candidate.exists() else None


def load_bart_and_tokenizer(checkpoint_dir: Path, base_model_name: str):
    """
    Try to load tokenizer from checkpoint_dir (if saved), otherwise from base_model_name.
    Then load bart base model, resize embeddings, and load saved state_dict if found.
    Returns: tokenizer, model
    """
    # load tokenizer: prefer saved tokenizer in checkpoint_dir
    if checkpoint_dir.exists() and (checkpoint_dir / "tokenizer_config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
        print(f"Loaded tokenizer from {checkpoint_dir}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        print(f"Loaded tokenizer from base {base_model_name}")

    # ensure special tokens used in training exist (e.g., <tgr>, [sep_arg], etc.)
    # add tokens if necessary (match whatever you used in training)
    special_tokens = ["<tgr>", "[sep_arg]"]
    added = tokenizer.add_tokens([t for t in special_tokens if t not in tokenizer.get_vocab()])
    if added:
        print(f"Added {added} special tokens to tokenizer")

    # load model
    model = BartForConditionalGeneration.from_pretrained(base_model_name)
    # resize embeddings to match tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # try to load best checkpoint weights (state_dict)
    best_ckpt = find_best_bart_checkpoint(checkpoint_dir)
    if best_ckpt:
        state = torch.load(str(best_ckpt), map_location="cpu")
        # state may be state_dict only or full model -> assume state_dict
        try:
            model.load_state_dict(state)
            print(f"Loaded BART weights from {best_ckpt}")
        except Exception as e:
            print("Warning: failed to load state_dict directly:", e)
    else:
        print("No bart checkpoint found in", checkpoint_dir)

    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def load_retriever_and_tokenizer(retriever_ckpt_dir: Path, base_model_name: str, event_types: List[str], max_length=128):
    """
    Load retriever model & tokenizer. Prefer checkpoint dir if exists (it may contain tokenizer).
    """
    if retriever_ckpt_dir.exists() and (retriever_ckpt_dir / "tokenizer_config.json").exists():
        retr_tokenizer = AutoTokenizer.from_pretrained(str(retriever_ckpt_dir))
        print(f"Loaded retriever tokenizer from {retriever_ckpt_dir}")
    else:
        retr_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        print(f"Loaded retriever tokenizer from base {base_model_name}")

    # instantiate retriever (your class handles encoding event types)
    retriever = EventTypeRetriever(
        model_name=str(retriever_ckpt_dir) if retriever_ckpt_dir.exists() else base_model_name,
        tokenizer=retr_tokenizer,
        device=DEVICE,
        event_types=event_types,
        max_length=max_length
    )
    return retriever, retr_tokenizer


# ---------------- Load resources at startup ----------------
print("Loading event types...")
event_types = build_labels(TRAIN_LABELS_PATH, EVENT_TYPES_CACHE)  # list[str]

print("Loading ontology...")
if not Path(ONTOLOGY_PATH).exists():
    raise FileNotFoundError(f"Ontology file not found: {ONTOLOGY_PATH}")
with open(ONTOLOGY_PATH, "r", encoding="utf-8") as f:
    ontology = json.load(f)

print("Loading BART + tokenizer...")
bart_tokenizer, bart_model = load_bart_and_tokenizer(CHECKPOINT_DIR, BART_BASE)

print("Loading Retriever + tokenizer...")
retriever, retr_tokenizer = load_retriever_and_tokenizer(RETRIEVER_CKPT_DIR, RETRIEVER_BASE, event_types, max_length=MAX_LENGTH)


# ---------------- FastAPI ----------------
app = FastAPI(title="Event Extraction API (2-stage R-GQA style)")
app.add_middleware(
  CORSMiddleware,
  allow_origins=["http://localhost:5173", "http://localhost:3000"],  # adjust port Vite uses
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)
class ExtractRequest(BaseModel):
    text: str
    top_k: int = TOP_K

class RoleAnswer(BaseModel):
    event_type: str
    role: str
    question: str
    answer: str
    # answer_str: str  # formatted like "<Role> answer"

class SentenceResult(BaseModel):
    input: str
    index_input: int
    role_answers: List[RoleAnswer]

class EventSummary(BaseModel):
    event_type: str
    frequency: int
    sentences: List[str]  # Các câu chứa sự kiện này
    total_roles: int  # Tổng số role được tìm thấy cho sự kiện này

class SummaryResponse(BaseModel):
    top_events: List[EventSummary]
    total_sentences: int
    total_events: int

def simple_sent_tokenize(text: str):
    # cắt theo dấu . ! ? + xuống dòng
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

@app.post("/extract", response_model=List[SentenceResult])
def extract(req: ExtractRequest):
    text = req.text
    top_k = req.top_k

    # 1) split sentences
    sentences = simple_sent_tokenize(text)
    results: List[Dict[str, Any]] = []

    for idx, sent in enumerate(sentences):
        sent_clean = sent.strip()
        if not sent_clean:
            continue

        # 2) retrieve top-k event types for this sentence
        try:
            top_events = [et for et, _ in retriever.retrieve(sent_clean, topk=top_k)]
        except Exception as e:
            # fallback: use empty list or use retriever base event type
            print("Retriever error:", e)
            top_events = []

        sentence_role_answers: List[Dict[str, Any]] = []

        # 3) for each top event type, iterate its questions (roles)
        for ev in top_events:
            if ev not in ontology:
                continue
            questions = ontology[ev].get("questions", {})
            # For each role/question, prepare input and call BART
            for role, question in questions.items():
                # Input format the same you used to train:
                input_text = f"sentence: {sent_clean} question: {question}"

                enc = bart_tokenizer(
                    input_text,
                    truncation=True,
                    padding="max_length",
                    max_length=MAX_LENGTH,
                    return_tensors="pt"
                )
                input_ids = enc["input_ids"].to(DEVICE)
                attention_mask = enc["attention_mask"].to(DEVICE)

                with torch.no_grad():
                    generated_ids = bart_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=OUTPUT_MAX_LENGTH,
                        num_beams=4,
                        do_sample=False,
                        early_stopping=True
                    )
                answer = bart_tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
                if answer == "":
                    answer = "none"

                # answer_str = f"<{role}> {answer}"
                sentence_role_answers.append({
                    "event_type": ev,
                    "role": role,
                    "question": question,
                    "answer": answer,
                })

        results.append({
            "input": sent_clean,
            "index_input": idx,
            "role_answers": sentence_role_answers
        })

    return results


@app.post("/extract-summary", response_model=SummaryResponse)
def extract_summary(req: ExtractRequest):
    """
    Tổng hợp tất cả sự kiện từ các câu và tìm top 3 sự kiện quan trọng nhất
    """
    text = req.text
    top_k = req.top_k

    # 1) Lấy kết quả extract từ endpoint cũ
    sentences = simple_sent_tokenize(text)
    all_events = []  # List để lưu tất cả sự kiện
    event_sentences = {}  # Dict để lưu câu chứa mỗi sự kiện
    event_roles = {}  # Dict để đếm số role của mỗi sự kiện

    for idx, sent in enumerate(sentences):
        sent_clean = sent.strip()
        if not sent_clean:
            continue

        # 2) retrieve top-k event types for this sentence with confidence filtering
        try:
            retrieved_events = retriever.retrieve(sent_clean, topk=top_k * 3)
            top_events = [et for et, score in retrieved_events if score > 0.5][:top_k]
        except Exception as e:
            print("Retriever error:", e)
            top_events = []

        # 3) Xử lý từng sự kiện trong câu này
        for ev in top_events:
            if ev not in ontology:
                continue
            
            # Thêm sự kiện vào danh sách
            all_events.append(ev)
            
            # Lưu câu chứa sự kiện này
            if ev not in event_sentences:
                event_sentences[ev] = []
            if sent_clean not in event_sentences[ev]:
                event_sentences[ev].append(sent_clean)
            
            # Đếm số role của sự kiện này
            if ev not in event_roles:
                event_roles[ev] = 0
            questions = ontology[ev].get("questions", {})
            event_roles[ev] += len(questions)

    # 4) Đếm tần suất và tìm top 3
    event_counter = Counter(all_events)
    top_3_events = event_counter.most_common(3)

    # 5) Tạo response
    top_events_summary = []
    for event_type, frequency in top_3_events:
        top_events_summary.append(EventSummary(
            event_type=event_type,
            frequency=frequency,
            sentences=event_sentences.get(event_type, []),
            total_roles=event_roles.get(event_type, 0)
        ))

    return SummaryResponse(
        top_events=top_events_summary,
        total_sentences=len(sentences),
        total_events=len(event_counter)
    )


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}