FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV FASTTEXT_MODEL_PATH=/opt/models/fasttext-es/model.bin

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Keep the 300-dimensional Spanish FastText model in a cached image layer.
# Normal source-code changes do not download the multi-GB model again.
COPY app/shared/nlp/embeddings/download_fasttext_model.py /tmp/download_fasttext_model.py
RUN HF_HOME=/tmp/hf-cache python /tmp/download_fasttext_model.py \
    --repo-id facebook/fasttext-es-vectors \
    --filename model.bin \
    --destination ${FASTTEXT_MODEL_PATH} \
    && rm -rf /tmp/hf-cache /tmp/download_fasttext_model.py

COPY app ./app
COPY sql ./sql
COPY README.md .

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
