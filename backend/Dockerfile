# backend/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
# Try app.main:app first (package layout), then main:app (flat layout)
CMD ["/bin/sh", "-lc", "python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 || python -m uvicorn main:app --host 0.0.0.0 --port 8000"]







