# backend Dockerfile
FROM python:3.11-slim

WORKDIR /code

# Cài dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt


# Copy toàn bộ code
COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
