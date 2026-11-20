# Inference API

The `src/api` package holds the FastAPI inference service that `shark-matcher-app` calls for matches, readiness, suggestions, and embedding updates.

## Modules

| File | Responsibility |
| --- | --- |
| `server.py` | The FastAPI `app`, request and response models, Bearer authentication, model-list parsing, and embedding-update job handling |

### Routes

| Route | Auth | Purpose |
| --- | --- | --- |
| `GET /health` | Public | Process liveness only |
| `GET /ready` | Bearer | Readiness of every required model; returns 503 when one is not ready |
| `POST /match` | Bearer | Ranked neighbors of one image for a retrieval method and model |
| `GET /stats` | Bearer | Loaded models, embedding metadata, covered filenames, and the active generation identity |
| `POST /suggestions` | Bearer | Cached unmatch or merge review suggestions for a model |
| `POST /embeddings/generate` | Bearer | Regenerates a model's embeddings from an app-published snapshot identifier and reloads the retriever; with `"background": true` it returns `{"status": "started", "job_id": ...}` at once |
| `GET /embeddings/status` | Bearer | State of the current or most recent generation job, for polling a background run |

The interactive API documentation routes are disabled.

## Entry Points

The server takes no command-line arguments, so it has no `--help`. It refuses to start without a valid `SHARK_INFERENCE_API_KEY`.

```bash
python3 -m src.api.server
```

The root `README.md` section "Running the Inference Server" owns the full start-up recipe, bind address, and `SHARK_MODELS` format.

## Boundaries and Dependencies

- `SHARK_MODELS` selects the served models. `SHARK_RETRIEVAL_MODE` and `SHARK_RERANK` fill in entries that omit those fields.
- `Retriever` in `src/retrieval/retrieval.py` owns model loading; the server only maps requests onto it.
- Wire models come from `src/contracts`, and snapshots are checked by `load_inference_snapshot` in `src/retrieval/inference_snapshot.py`.
- Generation runs one job at a time. By default it starts `python3 -m src.retrieval.embeddings` as a child process with `--update-embeddings` and the snapshot identifier.
- Live generation supports only the `baseline` and `multimodal` modes.
- The request carries a snapshot identifier, never a path.
