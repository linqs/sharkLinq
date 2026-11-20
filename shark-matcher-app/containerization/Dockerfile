# Shark Matcher App — Web Application Service
#
# Serves the web UI, REST API, and database.
# All ML inference is delegated to the shark-match-model inference service via HTTP.
#
# Build from the shark-matcher-app/ root:
#   docker build -f containerization/Dockerfile -t shark-matcher-app .
#
# Runtime volumes expected (see docker-compose.yml):
#   /sm-data  — database, images, logs, model artifacts, and annotation CSVs

FROM python:3.12-slim

# ---------------------------------------------------------------------------
# System libraries
#   libgomp1  — required by some scikit-learn / numpy builds
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies
# Install before copying source so this layer is cached across code changes.
# ---------------------------------------------------------------------------
WORKDIR /app

COPY requirements/ requirements/
RUN pip install --no-cache-dir -r requirements/requirements.txt

# ---------------------------------------------------------------------------
# Application source
# ---------------------------------------------------------------------------
COPY . .

# Install the package so all internal modules resolve correctly.
RUN pip install --no-cache-dir -e .

# ---------------------------------------------------------------------------
# Runtime configuration
#
# /sm-data is the shared volume that holds the SQLite database, shark images,
# application logs, model artifacts, and annotation CSVs.
# It is mounted at runtime — never baked in.
#
# core/app_constants.py resolves paths relative to the repo root:
#   THIS_DIR      = /app
#   BASE_DIR      = /          (parent of /app)
#   sm-data       = /sm-data
#
# Subdirectories created here ensure the container starts cleanly even if
# the volume is not mounted (e.g. standalone docker run without -v).
# ---------------------------------------------------------------------------
RUN mkdir -p \
    /sm-data/db \
    /sm-data/frontend-images \
    /sm-data/data-out/logs

EXPOSE 12345

# Environment variables (override via docker-compose environment: or -e flags).
#   INFERENCE_SERVICE_URL  — URL of the shark-match-model inference service
#   SHARK_MODELS           — comma-separated model names to preload
ENV INFERENCE_SERVICE_URL=http://shark-match-model:8000

CMD ["python3", "-m", "frontend.cli.serve", ".", "--no-browser"]
