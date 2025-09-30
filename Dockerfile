# backend Dockerfile
FROM python:3.11-slim

WORKDIR /code

# Cài dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code
COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
