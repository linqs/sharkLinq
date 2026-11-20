# ML inference service that loads trained models, generates embeddings, and serves a REST API on port 8000.

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# System libraries required by opencv-python.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies before copying source for layer caching.
WORKDIR /app

COPY requirements/ requirements/
RUN pip install --no-cache-dir -r requirements/requirements.txt

# Copy application source code.
COPY . .

# Runtime data volume for model checkpoints, embeddings, annotation CSVs, and images.
VOLUME /sm-data

EXPOSE 8000

# Preloaded model names list (override via environment variables).
ENV SHARK_MODELS=""

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
