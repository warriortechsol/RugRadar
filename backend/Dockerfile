FROM python:3.11-slim

WORKDIR /app

# Install system deps needed for Web3, cryptography, psycopg, etc.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps separately for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add source code
COPY . .

EXPOSE 8000

# Use exec form for proper signal handling in Docker
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]







