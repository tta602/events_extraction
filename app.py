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
from src.llm_event_detector import LLMEventDetector

# ---- Import Configuration ----
import config

# ---------------- CONFIG (from config.py) ----------------
# You can now edit settings in config.py instead of here!
# Or override via environment variables

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model paths
CHECKPOINT_DIR = config.CHECKPOINT_DIR
RETRIEVER_CKPT_DIR = config.RETRIEVER_CKPT_DIR
BEST_CHECKPOINT_TXT = config.BEST_CHECKPOINT_TXT

# Base model names
BART_BASE = config.BART_BASE
RETRIEVER_BASE = config.RETRIEVER_BASE

# Tokenizer/model params
MAX_LENGTH = config.MAX_LENGTH
OUTPUT_MAX_LENGTH = config.OUTPUT_MAX_LENGTH
TOP_K = config.TOP_K

# LLM Fallback Config
USE_LLM_FALLBACK = config.USE_LLM_FALLBACK
LLM_CONFIDENCE_THRESHOLD = config.LLM_CONFIDENCE_THRESHOLD
OPENAI_API_KEY = config.OPENAI_API_KEY

# Files
TRAIN_LABELS_PATH = config.TRAIN_LABELS_PATH
EVENT_TYPES_CACHE = config.EVENT_TYPES_CACHE
ONTOLOGY_PATH = config.ONTOLOGY_PATH

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
# Print configuration summary
config.print_config_summary()
config.validate_config()

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

# Optional: Load LLM detector for zero-shot event type detection
llm_detector = None
if USE_LLM_FALLBACK and OPENAI_API_KEY:
    try:
        print("Loading LLM Event Detector...")
        llm_detector = LLMEventDetector(
            api_key=OPENAI_API_KEY,
            known_event_types=event_types,
            model="gpt-4o-mini"  # Fast and cheap
        )
        print(f"✅ LLM fallback enabled (threshold: {LLM_CONFIDENCE_THRESHOLD})")
    except Exception as e:
        print(f"⚠️  Failed to load LLM detector: {e}")
        llm_detector = None
else:
    print("ℹ️  LLM fallback disabled. Set USE_LLM_FALLBACK=true and OPENAI_API_KEY to enable.")


# ---------------- FastAPI ----------------
app = FastAPI(title="Event Extraction API (2-stage R-GQA style)")
app.add_middleware(
  CORSMiddleware,
  allow_origins=[
      "http://localhost:5173", 
      "http://localhost:3000",
      "http://127.0.0.1:5173",
      "http://127.0.0.1:3000"
    ],  # adjust port Vite uses
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

class RoleInfo(BaseModel):
    role: str
    answer: str
    sentence: str

class EventSummary(BaseModel):
    event_type: str
    frequency: int
    sentences: List[str]  # Các câu chứa sự kiện này
    total_roles: int  # Tổng số role được tìm thấy cho sự kiện này
    roles: List[RoleInfo]  # Thông tin chi tiết về roles được phát hiện

class SummaryResponse(BaseModel):
    top_events: List[EventSummary]
    total_sentences: int
    total_events: int

def simple_sent_tokenize(text: str):
    # cắt theo dấu . ! ? + xuống dòng
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

def get_questions_for_event_type(event_type: str, ontology: dict) -> dict:
    """
    Get questions for an event type, with fallback to parent categories.
    
    Examples:
        - "Life.Die.Unspecified" → Use exact match in ontology
        - "Life.Die.Assassination" (new) → Fallback to "Life.Die.Unspecified"
        - "Disaster.AircraftMalfunction.TechnicalFailure" (new) → Try parent categories
    
    Fallback hierarchy:
        1. Exact match: "Category.Subcategory.Detail"
        2. Subcategory: "Category.Subcategory.Unspecified"
        3. Category: "Category.Unspecified.Unspecified" or similar
        4. Generic disaster/movement/life questions
    """
    # 1. Try exact match
    if event_type in ontology:
        return ontology[event_type].get("questions", {})
    
    # 2. Try parent with .Unspecified
    parts = event_type.split(".")
    
    if len(parts) >= 3:
        # Try: Category.Subcategory.Unspecified
        parent_1 = f"{parts[0]}.{parts[1]}.Unspecified"
        if parent_1 in ontology:
            print(f"[FALLBACK] Using {parent_1} questions for {event_type}")
            return ontology[parent_1].get("questions", {})
    
    if len(parts) >= 2:
        # Try: Category.*.Unspecified (find any subcategory under Category)
        category = parts[0]
        subcategory_prefix = f"{category}."
        for key in ontology:
            if key.startswith(subcategory_prefix) and key.endswith(".Unspecified"):
                print(f"[FALLBACK] Using {key} questions for {event_type}")
                return ontology[key].get("questions", {})
    
    # 3. Last resort: generic questions based on category
    if len(parts) >= 1:
        category = parts[0]
        generic_questions = {
            "Disaster": {
                "Victim": "who are the victims?",
                "Place": "where did this happen?",
                "Instrument": "what caused this disaster?"
            },
            "Movement": {
                "Transporter": "who or what is moving?",
                "Origin": "where did the movement start?",
                "Destination": "where is the destination?",
                "PassengerArtifact": "who or what is being transported?"
            },
            "Life": {
                "Victim": "who is affected?",
                "Agent": "who is responsible?",
                "Place": "where did this occur?"
            },
            "Transaction": {
                "Buyer": "who is the buyer?",
                "Seller": "who is the seller?",
                "Artifact": "what is being transacted?",
                "Price": "what is the price?"
            },
            "Justice": {
                "Investigator": "who is investigating?",
                "Defendant": "who is being investigated?",
                "Place": "where is this happening?"
            }
        }
        
        if category in generic_questions:
            print(f"[FALLBACK] Using generic {category} questions for {event_type}")
            return generic_questions[category]
    
    # 4. No questions found
    print(f"[WARNING] No questions found for {event_type}, skipping role extraction")
    return {}

def retrieve_event_types_with_fallback(sentence: str, top_k: int) -> List[str]:
    """
    Retrieve event types using retriever, with optional LLM fallback for low-confidence cases.
    
    Returns:
        List of event type strings
    """
    try:
        # Step 1: Try retriever first
        retrieved_events = retriever.retrieve(sentence, topk=top_k * 3)
        high_conf_events = [(et, score) for et, score in retrieved_events if score > 0.5]
        
        # If we have enough high-confidence results, use them
        if len(high_conf_events) >= top_k:
            return [et for et, _ in high_conf_events[:top_k]]
        
        # Step 2: Check if max score is below LLM threshold
        max_score = max([score for _, score in retrieved_events], default=0.0)
        
        if llm_detector and max_score < LLM_CONFIDENCE_THRESHOLD:
            print(f"[LLM FALLBACK] Retriever confidence too low ({max_score:.3f}), using LLM for: {sentence[:50]}...")
            
            # Use LLM to detect event types
            llm_events = llm_detector.detect_event_types(sentence, top_k=top_k, confidence_threshold=0.6)
            
            if llm_events:
                result_events = []
                for event_type, conf, is_new in llm_events:
                    result_events.append(event_type)
                    if is_new:
                        print(f"  ⚠️  NEW EVENT TYPE DETECTED: {event_type} (confidence: {conf:.2f})")
                    else:
                        print(f"  ✅ Matched known type: {event_type} (confidence: {conf:.2f})")
                return result_events[:top_k]
        
        # Fallback: use retriever results even if low confidence
        return [et for et, _ in retrieved_events[:top_k]]
        
    except Exception as e:
        print(f"[RETRIEVAL ERROR] {e}")
        return []

@app.post("/extract", response_model=List[SentenceResult])
def extract(req: ExtractRequest):
    """
    Extract events and roles for each sentence.
    Uses BATCH inference: All roles per sentence are processed in a single BART call.
    Much faster than individual inference!
    """
    text = req.text
    top_k = req.top_k

    # 1) split sentences
    sentences = simple_sent_tokenize(text)
    results: List[Dict[str, Any]] = []
    
    total_bart_calls = 0  # Track performance

    for idx, sent in enumerate(sentences):
        sent_clean = sent.strip()
        if not sent_clean:
            continue

        # 2) retrieve top-k event types (with optional LLM fallback)
        top_events = retrieve_event_types_with_fallback(sent_clean, top_k)

        sentence_role_answers: List[Dict[str, Any]] = []

        # 3) for each top event type, collect all questions for batch inference
        batch_inputs = []
        batch_metadata = []  # Store (event_type, role, question) for each input
        
        for ev in top_events:
            # Get questions with fallback to parent categories or generic questions
            questions = get_questions_for_event_type(ev, ontology)
            
            if not questions:
                continue  # Skip if no questions found at all
            
            for role, question in questions.items():
                input_text = f"sentence: {sent_clean} question: {question}"
                batch_inputs.append(input_text)
                batch_metadata.append((ev, role, question))
        
        # Batch inference: process all questions at once
        if batch_inputs:
            total_bart_calls += 1  # Count as 1 batch call instead of N individual calls
            
            enc = bart_tokenizer(
                batch_inputs,
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
            
            # Decode all outputs
            for idx, (ev, role, question) in enumerate(batch_metadata):
                raw_answer = bart_tokenizer.decode(generated_ids[idx], skip_special_tokens=True).strip()
                
                # Parse answer từ format "<Role> answer" -> lấy phần sau ">"
                if raw_answer.startswith("<") and ">" in raw_answer:
                    answer = raw_answer.split(">", 1)[1].strip()
                else:
                    answer = raw_answer
                
                if answer == "":
                    answer = "none"

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

    print(f"[PERFORMANCE] /extract: {total_bart_calls} BART inference calls for {len(sentences)} sentences")
    return results


@app.post("/extract-summary", response_model=SummaryResponse)
def extract_summary(req: ExtractRequest):
    """
    Tổng hợp tất cả sự kiện từ các câu và tìm top 3 sự kiện quan trọng nhất.
    Uses BATCH inference: All roles per sentence are processed in a single BART call.
    """
    text = req.text
    top_k = req.top_k

    # 1) Lấy kết quả extract từ endpoint cũ
    sentences = simple_sent_tokenize(text)
    all_events = []  # List để lưu tất cả sự kiện
    event_sentences = {}  # Dict để lưu câu chứa mỗi sự kiện
    event_role_details = {}  # Dict để lưu chi tiết roles (role, answer, sentence)
    total_bart_calls = 0  # Track performance

    for idx, sent in enumerate(sentences):
        sent_clean = sent.strip()
        if not sent_clean:
            continue

        # 2) retrieve top-k event types (with optional LLM fallback)
        top_events = retrieve_event_types_with_fallback(sent_clean, top_k)

        # 3) Xử lý từng sự kiện và chuẩn bị batch inference
        batch_inputs = []
        batch_metadata = []  # (event_type, role, question)
        
        for ev in top_events:
            # Get questions with fallback to parent categories or generic questions
            questions = get_questions_for_event_type(ev, ontology)
            
            if not questions:
                continue  # Skip if no questions found
            
            # Thêm sự kiện vào danh sách
            all_events.append(ev)
            
            # Lưu câu chứa sự kiện này
            if ev not in event_sentences:
                event_sentences[ev] = []
            if sent_clean not in event_sentences[ev]:
                event_sentences[ev].append(sent_clean)
            
            # Khởi tạo nếu chưa có
            if ev not in event_role_details:
                event_role_details[ev] = []
            
            # Collect all questions for batch processing
            for role, question in questions.items():
                input_text = f"sentence: {sent_clean} question: {question}"
                batch_inputs.append(input_text)
                batch_metadata.append((ev, role, question))
        
        # Process batch
        if batch_inputs:
            total_bart_calls += 1  # Count as 1 batch call
            
            enc = bart_tokenizer(
                batch_inputs,
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
            
            # Process all outputs
            for idx, (ev, role, question) in enumerate(batch_metadata):
                raw_answer = bart_tokenizer.decode(generated_ids[idx], skip_special_tokens=True).strip()
                
                # Parse answer từ format "<Role> answer" -> lấy phần sau ">"
                if raw_answer.startswith("<") and ">" in raw_answer:
                    answer = raw_answer.split(">", 1)[1].strip()
                else:
                    answer = raw_answer
                
                if answer == "":
                    answer = "none"
                
                # Chỉ lưu roles có answer hợp lệ (không phải "none")
                if answer.lower() != "none" and len(answer.strip()) > 0:
                    # Kiểm tra xem role+answer này đã tồn tại chưa (để tránh duplicate)
                    exists = any(
                        r.role == role and r.answer.lower() == answer.lower()
                        for r in event_role_details[ev]
                    )
                    if not exists:
                        event_role_details[ev].append(RoleInfo(
                            role=role,
                            answer=answer,
                            sentence=sent_clean
                        ))

    # 4) Đếm tần suất và tìm top 3
    event_counter = Counter(all_events)
    top_3_events = event_counter.most_common(3)

    # 5) Tạo response
    top_events_summary = []
    for event_type, frequency in top_3_events:
        # Đếm số roles thực sự được phát hiện (không phải tổng questions trong ontology)
        detected_roles = event_role_details.get(event_type, [])
        top_events_summary.append(EventSummary(
            event_type=event_type,
            frequency=frequency,
            sentences=event_sentences.get(event_type, []),
            total_roles=len(detected_roles),  # Số roles thực sự phát hiện được
            roles=detected_roles
        ))

    print(f"[PERFORMANCE] /extract-summary: {total_bart_calls} BART inference calls for {len(sentences)} sentences")
    return SummaryResponse(
        top_events=top_events_summary,
        total_sentences=len(sentences),
        total_events=len(event_counter)
    )


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}