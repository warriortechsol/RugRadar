# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Copy dependency manifests first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]








