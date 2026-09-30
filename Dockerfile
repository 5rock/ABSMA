FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Force CPU execution
ENV CUDA_VISIBLE_DEVICES=""

# HuggingFace configuration
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV TOKENIZERS_PARALLELISM=false

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

RUN pip install --upgrade pip

# CPU-only PyTorch
RUN pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision torchaudio

# Remaining dependencies
RUN pip install -r requirements.txt

# SpaCy model
RUN python -m spacy download en_core_web_sm

# Copy application
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

RUN mkdir -p /app/.cache/huggingface && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 5000

CMD ["python", "app/app.py"]
