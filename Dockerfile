FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV SENTENCE_TRANSFORMER_CACHE_DIR=/opt/models/sentence-transformers

ARG SENTENCE_TRANSFORMER_MODEL_REPO_ID=intfloat/multilingual-e5-small
ARG SENTENCE_TRANSFORMER_MODEL_REVISION=main

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.3,<3.0" \
    && pip install --no-cache-dir -r requirements.txt

# Preload the 384-dimensional retrieval encoder into a cached image layer.
# A fine-tuned public model can replace the build ARG; a private model can be
# selected at runtime with EMBEDDING_MODEL + HF_TOKEN.
COPY app/shared/nlp/embeddings/download_sentence_transformer_model.py /tmp/download_sentence_transformer_model.py
RUN HF_HOME=/tmp/hf-cache python /tmp/download_sentence_transformer_model.py \
    --repo-id ${SENTENCE_TRANSFORMER_MODEL_REPO_ID} \
    --revision ${SENTENCE_TRANSFORMER_MODEL_REVISION} \
    --cache-dir ${SENTENCE_TRANSFORMER_CACHE_DIR} \
    && rm -rf /tmp/hf-cache /tmp/download_sentence_transformer_model.py

COPY app ./app
COPY sql ./sql
COPY README.md .

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
