# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install dependencies first (cache-friendly)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY backend/. .

# Expose API port
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]









