FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV FASTTEXT_MODEL_PATH=/opt/models/fasttext-es/model.bin
ARG DOWNLOAD_FASTTEXT_MODEL=true

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Keep the rollback FastText model in a cached layer unless a BERT-only image
# is requested with --build-arg DOWNLOAD_FASTTEXT_MODEL=false.
COPY app/shared/nlp/embeddings/download_fasttext_model.py /tmp/download_fasttext_model.py
RUN if [ "${DOWNLOAD_FASTTEXT_MODEL}" = "true" ]; then \
      HF_HOME=/tmp/hf-cache python /tmp/download_fasttext_model.py \
        --repo-id facebook/fasttext-es-vectors \
        --filename model.bin \
        --destination ${FASTTEXT_MODEL_PATH}; \
    fi \
    && rm -rf /tmp/hf-cache /tmp/download_fasttext_model.py

COPY app ./app
COPY sql ./sql
COPY README.md .

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
