# 🎯 Event Extraction System

Hệ thống AI tự động trích xuất sự kiện và vai trò từ văn bản tiếng Anh, sử dụng kiến trúc 2 giai đoạn với RoBERTa (Event Detection) và BART (Role Extraction).

---

## ✨ Tính năng chính

- 🤖 **Event Detection**: Nhận diện 49+ loại sự kiện (có thể mở rộng vô hạn với LLM)
- 📝 **Role Extraction**: Trích xuất 5-10 vai trò cho mỗi sự kiện (who, what, where, when, etc.)
- 🚀 **Zero-shot Learning**: Hỗ trợ domain mới không cần retrain (LLM fallback)
- ⚡ **Batch Processing**: Xử lý nhanh với batch inference
- 🎨 **Beautiful UI**: Giao diện hiện đại với visualization và highlighting
- 🔌 **RESTful API**: Dễ dàng tích hợp vào hệ thống hiện có
- 🐳 **Docker Ready**: Containerized deployment

---

## 📋 Yêu cầu hệ thống

- **Python**: 3.8+
- **Node.js**: 16+ (cho frontend)
- **RAM**: 8GB minimum (16GB recommended)
- **CPU**: 4 cores minimum
- **Disk**: 10GB cho models và dependencies
- **GPU**: Optional (tăng tốc 5-10x)

---

## 🚀 Quick Start

### Option 1: Docker (Khuyến nghị)

```bash
# Build và run cả backend + frontend
docker-compose up --build

# Truy cập:
# - Frontend: http://localhost:5173
# - Backend API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

### Option 2: Manual Setup (Development)

#### 1. Backend Setup

```bash
# Clone repo (nếu chưa có)
git clone <repo-url>
cd events_extraction

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Cài dependencies
pip install -r requirements.txt

# Download NLTK data (nếu cần)
python -c "import nltk; nltk.download('punkt')"

# Chạy backend
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

#### 2. Frontend Setup

```bash
# Mở terminal mới
cd event-extract-ui

# Cài dependencies
npm install

# Chạy frontend
npm run dev

# Frontend sẽ chạy tại: http://localhost:5173
```

#### 3. Verify

```bash
# Check backend health
curl http://localhost:8000/health

# Test API
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "A bomb exploded in Baghdad, killing 15 people.", "top_k": 2}'
```

---

## 📁 Cấu trúc project

```
events_extraction/
├── app.py                 # FastAPI backend chính
├── config.py              # File cấu hình tập trung
├── requirements.txt       # Python dependencies
├── docker-compose.yml     # Docker setup
├── Dockerfile            # Backend Docker image
│
├── checkpoints/          # Model checkpoints
│   ├── retrieve_best_model/      # RoBERTa retriever
│   └── bart_best_model_*.pt      # BART model
│
├── ontoloy/             # Ontology (event types + questions)
│   └── event_role_WIKI_q.json
│
├── src/                 # Source code
│   ├── eventtype_retriever.py    # Event detection
│   ├── llm_event_detector.py     # LLM fallback (optional)
│   └── utils/                     # Utilities
│
├── processing_data/     # Training data
│   ├── train.json
│   ├── dev.json
│   └── event_types.json
│
├── event-extract-ui/    # React frontend
│   ├── src/
│   │   ├── App.tsx
│   │   └── components/
│   ├── package.json
│   └── Dockerfile
│
└── scripts/            # Training/evaluation scripts
```

---

## ⚙️ Cấu hình

### Cách 1: Sửa trực tiếp `config.py` (Đơn giản)

```python
# config.py
USE_LLM_FALLBACK = True  # Bật LLM fallback
OPENAI_API_KEY = "sk-proj-YOUR-KEY-HERE"
LLM_CONFIDENCE_THRESHOLD = 0.5
MAX_LENGTH = 256
TOP_K = 1
```

### Cách 2: Environment Variables (Production)

```bash
# Bật LLM fallback
export USE_LLM_FALLBACK=true
export OPENAI_API_KEY=sk-proj-YOUR-KEY-HERE
export LLM_CONFIDENCE_THRESHOLD=0.5

# Chạy backend
python -m uvicorn app:app --reload
```

### Các tham số quan trọng:

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `USE_LLM_FALLBACK` | `false` | Bật/tắt LLM fallback cho event types mới |
| `OPENAI_API_KEY` | `None` | API key OpenAI (nếu dùng LLM) |
| `LLM_CONFIDENCE_THRESHOLD` | `0.5` | Ngưỡng confidence để gọi LLM |
| `MAX_LENGTH` | `256` | Độ dài input tối đa |
| `TOP_K` | `1` | Số event types retrieve mỗi câu |
| `DEVICE` | auto | Device cho inference (`cpu`, `cuda`, `mps`) |

**Chi tiết đầy đủ:** Xem file `config.py`

---

## 🔌 API Endpoints

### 1. Extract Events (Detailed)

```bash
POST /extract
Content-Type: application/json

{
  "text": "A bomb exploded in Baghdad, killing 15 people.",
  "top_k": 2
}
```

**Response:**
```json
{
  "events": [
    {
      "sentence": "A bomb exploded in Baghdad, killing 15 people.",
      "event_type": "Conflict.Attack.DetonateExplode",
      "arguments": [
        {"role": "Attacker", "answer": "bomber"},
        {"role": "Place", "answer": "Baghdad"},
        {"role": "Victim", "answer": "15 people"}
      ]
    }
  ]
}
```

### 2. Extract Summary (Aggregated)

```bash
POST /extract-summary

{
  "text": "Multiple sentences text...",
  "top_k": 2
}
```

**Response:**
```json
{
  "top_events": [
    {
      "event_type": "Conflict.Attack.DetonateExplode",
      "frequency": 2,
      "sentences": ["...", "..."],
      "total_roles": 5,
      "roles": [
        {"role": "Attacker", "answer": "bomber", "sentence": "..."},
        {"role": "Victim", "answer": "15 people", "sentence": "..."}
      ]
    }
  ],
  "total_sentences": 3,
  "total_events": 2
}
```

### 3. Health Check

```bash
GET /health
```

**API Docs:** http://localhost:8000/docs

---

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────┐
│   Input Text        │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Stage 1: RoBERTa   │  ← Event Type Detection
│  Retriever Model    │     (49 types trained)
└──────────┬──────────┘
           │
           ├─ High confidence (>0.5) ──────┐
           │                                │
           └─ Low confidence (<0.5) ────────┤
                        │                   │
              ┌─────────▼────────┐          │
              │  LLM Fallback    │          │
              │  (GPT-4o-mini)   │          │
              │  Zero-shot       │          │
              └─────────┬────────┘          │
                        │                   │
           ┌────────────┴───────────────────┘
           │
┌──────────▼──────────┐
│  Stage 2: BART      │  ← Role Extraction
│  Question Answering │     (5-10 roles per event)
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Output: Events +   │
│  Roles + Answers    │
└─────────────────────┘
```

### Stage 1: Event Type Detection
- **Model**: RoBERTa-base (fine-tuned)
- **Method**: Cosine similarity với event embeddings
- **Output**: Top-k event types với confidence scores
- **Fallback**: LLM nếu confidence < threshold

### Stage 2: Role Extraction
- **Model**: BART-base (fine-tuned)
- **Method**: Question-Answering (QA) approach
- **Input**: Sentence + Event Type + Questions từ ontology
- **Output**: Answers cho mỗi role (who, what, where, when, etc.)

---

## 🧪 Testing

### Test với curl:

```bash
# Simple test
curl -X POST http://localhost:8000/extract-summary \
  -H "Content-Type: application/json" \
  -d '{
    "text": "A suicide bomber killed 15 people in Baghdad yesterday.",
    "top_k": 2
  }' | python -m json.tool
```

### Test với Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/extract-summary",
    json={
        "text": "A suicide bomber killed 15 people in Baghdad yesterday.",
        "top_k": 2
    }
)
print(response.json())
```

### Test cases mẫu:

Xem file `DEMO_GUIDE.md` để có danh sách test cases đầy đủ.

---

## 🎨 Frontend Features

- **Tab "Chi tiết"**: 
  - Hiển thị từng câu và events được phát hiện
  - Roles được highlight trong text gốc
  - Color coding theo event types

- **Tab "Tổng hợp"**:
  - Statistics tổng quan (số câu, số events)
  - Top 3 events phổ biến nhất
  - Danh sách tất cả roles detected

- **Tab "Biểu đồ"**:
  - Pie chart: Phân bố event types
  - Bar chart: Số lượng roles per event
  - Heatmap: Mối quan hệ Event-Role

---

## 🔧 Troubleshooting

### Backend không start được

```bash
# Check port 8000 có bị chiếm không
lsof -i :8000
# Nếu có, kill process:
kill -9 <PID>

# Check Python version
python --version  # Cần >= 3.8

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Frontend không load

```bash
# Check port 5173
lsof -i :5173

# Clear npm cache
npm cache clean --force
rm -rf node_modules package-lock.json
npm install

# Check backend connection
curl http://localhost:8000/health
```

### Model không tìm thấy

```bash
# Check checkpoints folder
ls -la checkpoints/
ls -la checkpoints/retrieve_best_model/

# Nếu thiếu, cần download hoặc train lại
```

### API response chậm

- **CPU inference**: ~2 seconds per document (normal)
- **Cải thiện**:
  - Dùng GPU: 5-10x faster
  - Giảm `top_k` xuống 1
  - Giảm `MAX_LENGTH` xuống 128
  - Deploy trên server mạnh hơn

---

## 📊 Performance

| Metric | CPU | GPU (CUDA) |
|--------|-----|------------|
| Inference time per document | ~2s | ~0.3s |
| Throughput | 30 docs/min | 200 docs/min |
| Memory usage | 4GB | 6GB |
| Recommended setup | 8GB RAM, 4 cores | 16GB RAM, NVIDIA GPU |

---

## 🔐 Bảo mật

- **Self-hosted**: Dữ liệu không rời khỏi server
- **LLM optional**: Có thể disable hoàn toàn (set `USE_LLM_FALLBACK=false`)
- **No logging**: Không log input text (configurable)
- **Docker isolated**: Chạy trong container riêng biệt
- **API auth**: Có thể thêm token authentication

---

## 💰 Chi phí

### Infrastructure:
- **Self-hosted**: FREE (chỉ phí server)
- **Cloud (AWS t3.xlarge)**: ~$150/month
- **Cloud với GPU (g4dn.xlarge)**: ~$500/month

### LLM API (Optional):
- **GPT-4o-mini**: $0.0002 per call
- **Typical usage**: 100-1000 calls/month = $0.02-$0.20/month
- **Very affordable!**

---

## 📚 Documentation

- **Hướng dẫn demo**: Xem `DEMO_GUIDE.md`
- **API Documentation**: http://localhost:8000/docs (khi backend đang chạy)
- **Configuration**: Xem `config.py`
- **Training scripts**: Xem thư mục `scripts/`

---

## 🤝 Support & Contact

Nếu gặp vấn đề:
1. Check logs trong terminal
2. Check API health: `curl http://localhost:8000/health`
3. Xem phần Troubleshooting ở trên
4. Liên hệ team

---

## 📝 License

[Add your license here]

---

## 🎉 Quick Start Checklist

- [ ] Clone repo
- [ ] Install Python dependencies: `pip install -r requirements.txt`
- [ ] Install Node dependencies: `cd event-extract-ui && npm install`
- [ ] Start backend: `python -m uvicorn app:app --reload`
- [ ] Start frontend: `cd event-extract-ui && npm run dev`
- [ ] Open browser: http://localhost:5173
- [ ] Test với sample text
- [ ] Read `DEMO_GUIDE.md` để biết cách demo

**Done! Hệ thống sẵn sàng sử dụng! 🚀**