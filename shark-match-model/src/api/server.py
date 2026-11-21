"""FastAPI inference service exposing retrieval and embedding endpoints for shark-match-model."""

from contextlib import asynccontextmanager
import hmac
import os
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import typing
import uuid

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, field_validator

from src.core import constants
import src
from src.contracts import (
    ModelDiagnostic, ModelSpec, ModelState, ModelStatus, Rerank, RetrievalMode, canonical_uuid4)
import src.individual_dataset
import src.learning.augmented_dataset
import src.retrieval.embeddings as _embeddings
import src.retrieval.generation_store
from src.retrieval.inference_snapshot import load_inference_snapshot
from src.retrieval.retrieval import Retriever, validate_model_checkpoint

_MINIMUM_SERVICE_API_KEY_LENGTH = 32
_SERVER_GENERATION_MODES = {"baseline", "multimodal"}
_VALID_MODES = set(typing.get_args(RetrievalMode))
_VALID_RERANKS = set(typing.get_args(Rerank))


def _resolve_manifest_model(dir_name: str) -> str | None:
    """Inspect active generation manifest in the embeddings directory if present."""
    target_dir = os.path.join(constants.EMBEDDINGS_DIR, dir_name)
    try:
        active = src.retrieval.generation_store.resolve_active_generation(target_dir)
        if active.manifest and active.manifest.model_name:
            return active.manifest.model_name
    except Exception:
        pass
    return None


def _require_known(setting, value, valid_values):
    """Raise when a model setting names a value the server does not support."""
    if value and value not in valid_values:
        raise ValueError(
            f"Unknown {setting} '{value}'; use one of {', '.join(sorted(valid_values))}.")


def _parse_models_env() -> list[ModelSpec]:
    """Parse the SHARK_MODELS environment variable into a list of model specifications."""
    raw = os.getenv("SHARK_MODELS", "").strip()
    if not raw:
        return [ModelSpec(**m) for m in constants.DEFAULT_MODELS]

    if raw.startswith("["):
        import json
        try:
            items = json.loads(raw)
            return [ModelSpec(**item) for item in items]
        except Exception as exc:
            raise ValueError(f"Invalid JSON array in SHARK_MODELS: {exc}") from exc

    # Fallback to global env variables when entry does not specify retrieval mode or rerank.
    global_mode = os.getenv("SHARK_RETRIEVAL_MODE", "").strip().lower()
    _require_known("SHARK_RETRIEVAL_MODE", global_mode, _VALID_MODES)
    global_rerank = os.getenv("SHARK_RERANK", "").strip().lower()
    _require_known("SHARK_RERANK", global_rerank, _VALID_RERANKS)

    default_by_display = {m["display_name"]: m for m in constants.DEFAULT_MODELS}

    parsed = []
    for entry in raw.split(","):
        parts = [p.strip() for p in entry.split(":")]
        display_name = parts[0]
        if not display_name:
            continue
        if len(parts) == 2:
            raise ValueError(
                f"SHARK_MODELS entry '{entry.strip()}' gives MODE without RERANK; "
                "use DISPLAY_NAME:MODE:RERANK[:EMBEDDING_SET[:CHECKPOINT]].")
        if len(parts) >= 3:
            mode, rerank = parts[1].lower(), parts[2].lower()
            _require_known(f"MODE in SHARK_MODELS entry '{display_name}'", mode, _VALID_MODES)
            _require_known(f"RERANK in SHARK_MODELS entry '{display_name}'", rerank, _VALID_RERANKS)
            dir_name = parts[3] if len(parts) >= 4 else display_name
            if len(parts) >= 5:
                model = parts[4]
            else:
                default = default_by_display.get(display_name, {})
                if default and default.get("dir") == dir_name:
                    model = default.get("model", dir_name)
                else:
                    manifest_model = _resolve_manifest_model(dir_name)
                    model = manifest_model or dir_name
            required = True
        else:
            default = default_by_display.get(display_name, {})
            mode = global_mode or default.get("retrieval_mode", "multimodal")
            rerank = global_rerank or default.get("rerank", "kreciprocal")
            dir_name = default.get("dir", display_name)
            manifest_model = _resolve_manifest_model(dir_name)
            model = default.get("model") or manifest_model or dir_name
            required = default.get("required", True)
        parsed.append(ModelSpec(
            display_name=display_name,
            embedding_set=dir_name,
            checkpoint=model,
            retrieval_mode=mode,
            rerank=rerank,
            required=required,
        ))

    return parsed or [ModelSpec(**m) for m in constants.DEFAULT_MODELS]


DEFAULT_MODELS = _parse_models_env()

# Serializes embedding generation to prevent concurrent state corruption.
_generation_lock = threading.Lock()
_generation_job: dict = {
    "status": "idle",
    "job_id": None,
    "model_name": None,
    "snapshot_id": None,
    "started_at": None,
    "completed_at": None,
    "error": None,
}
_bearer_scheme = HTTPBearer(auto_error=False)


class MatchRequest(BaseModel):
    """Request for the ranked neighbours of one image under a retrieval method and model."""
    image_id: int
    retrieval_method: int = 0
    model: str = None


class MatchResponse(BaseModel):
    """Ranked neighbour image IDs with their distances and labels."""
    ids: list
    distances: list
    labels: list


class HealthResponse(BaseModel):
    """Process liveness reported by the public health check."""
    status: str


class ReadinessResponse(BaseModel):
    """Overall readiness together with each configured model's status."""
    status: str
    model_statuses: list[ModelStatus]


class StatsResponse(BaseModel):
    """Loaded models, embedding metadata, covered filenames, and the active generation identity."""
    loaded_models: list
    embeddings_metadata: dict
    filenames: list
    model_statuses: list[ModelStatus]
    generation_id: str | None
    snapshot_id: str | None


class EmbeddingsRequest(BaseModel):
    """Request to regenerate one model's embeddings from a snapshot."""
    model_name: str
    snapshot_id: str
    background: bool = False

    @field_validator("snapshot_id")
    @classmethod
    def validate_snapshot_id(cls, value):
        """Require a lowercase UUIDv4 snapshot ID."""
        return canonical_uuid4(value, "snapshot_id")


class StatusResponse(BaseModel):
    """Outcome of an embedding generation request."""
    status: str
    job_id: str | None = None


class EmbeddingsJobStatusResponse(BaseModel):
    """Status of the most recent or active embedding generation job."""
    status: str
    job_id: str | None = None
    model_name: str | None = None
    snapshot_id: str | None = None
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None


class SuggestionsRequest(BaseModel):
    """Request for one type of cached review suggestions for a model."""
    model: str = None
    review_type: str = "unmatch_suggestions"


class SuggestionsResponse(BaseModel):
    """Cached review suggestions for the requested model and type."""
    suggestions: list


def main():
    """Run the inference server via uvicorn."""
    import uvicorn
    host, port = constants.resolve_inference_bind_address()
    # Passing the app object keeps uvicorn from importing this module a second time.
    uvicorn.run(app, host=host, port=port, reload=False)


def require_service_api_key(
        credentials: HTTPAuthorizationCredentials | None = Depends(
            _bearer_scheme
        )):
    """Require the configured Bearer credential for a protected endpoint."""
    configured_key = _require_configured_service_key()
    if (credentials is None
            or not hmac.compare_digest(
                credentials.credentials, configured_key
            )):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing service credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage inference server startup, configuration validation, and retriever preload."""
    _require_configured_service_key()
    _log_model_config(DEFAULT_MODELS)
    app.state.retriever = Retriever(preload_models=DEFAULT_MODELS)
    yield


app = FastAPI(
    title="Shark Matcher Inference Service",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
protected_router = APIRouter(
    dependencies=[Depends(require_service_api_key)]
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Report process liveness without deriving model readiness."""
    return {"status": "ok"}


@protected_router.get("/ready", response_model=ReadinessResponse)
def ready(response: Response):
    """Report whether every required model is ready for inference."""
    retriever: Retriever = app.state.retriever
    model_statuses = retriever.get_model_statuses()
    is_ready = all(
        not status.required or status.state == ModelState.READY
        for status in model_statuses
    )
    if not is_ready:
        response.status_code = 503
    return {
        "status": "ready" if is_ready else "not_ready",
        "model_statuses": model_statuses,
    }


@protected_router.post("/match", response_model=MatchResponse)
def match(req: MatchRequest):
    """Return nearest-neighbour matches for a query image."""
    retriever: Retriever = app.state.retriever
    if req.retrieval_method not in retriever.retrieval_methods:
        raise HTTPException(status_code=400, detail="Unknown retrieval method.")
    model_name = retriever._resolve_model(req.model)  # pylint: disable=protected-access
    if model_name not in retriever.allowed_models:
        raise HTTPException(status_code=400, detail="Unknown model.")
    if retriever._get_model_data(req.model) is None:  # pylint: disable=protected-access
        raise HTTPException(status_code=503, detail="Requested model is unavailable.")
    result = retriever.get_neighbors(req.retrieval_method, req.image_id, model=req.model)
    return {
        "ids": result.get("ids", []),
        "distances": result.get("distances", []),
        "labels": result.get("labels", []),
    }


@protected_router.get("/stats", response_model=StatsResponse)
def stats():
    """Return retriever state, loaded models, and covered filenames for the dataset-stats panel."""
    retriever: Retriever = app.state.retriever
    # Return display names shown by the frontend.
    loaded_display = retriever.get_loaded_display_names()
    model_statuses = retriever.get_model_statuses()

    default_data = retriever.loaded_models.get(retriever.default_model_name)
    if default_data is None:
        return {
            "loaded_models": loaded_display,
            "embeddings_metadata": {"start_time": None, "update_time": None},
            "filenames": [],
            "model_statuses": model_statuses,
            "generation_id": None,
            "snapshot_id": None,
        }

    filenames = default_data.get("filenames")
    if filenames is None:
        neighbors = default_data.get("image_neighbors") or {}
        filenames = list(neighbors.keys())
    else:
        filenames = list(filenames)

    return {
        "loaded_models": loaded_display,
        "embeddings_metadata": default_data["embeddings_metadata"],
        "filenames": filenames,
        "model_statuses": model_statuses,
        "generation_id": default_data.get("generation_id"),
        "snapshot_id": default_data.get("snapshot_id"),
    }


@protected_router.get(
    "/embeddings/status", response_model=EmbeddingsJobStatusResponse
)
def embeddings_status():
    """Return the current or most recent embedding generation job status."""
    return dict(_generation_job)


@protected_router.post(
    "/embeddings/generate",
    response_model=StatusResponse,
    response_model_exclude_none=True,
)
def embeddings_generate(req: EmbeddingsRequest):
    """Regenerate embeddings for a model and hot-reload the retriever."""
    retriever: Retriever = app.state.retriever
    config_name = retriever._resolve_model(req.model_name)
    cfg = retriever._model_configs.get(config_name, {})
    dir_name = cfg.get("dir", config_name)
    checkpoint = cfg.get("checkpoint") or cfg.get("model", dir_name)
    if checkpoint == dir_name:
        manifest_model = _resolve_manifest_model(dir_name)
        if manifest_model:
            checkpoint = manifest_model

    if not _generation_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="An embedding generation is already in progress."
        )
    job_id = uuid.uuid4().hex
    _generation_job.update({
        "status": "running",
        "job_id": job_id,
        "model_name": req.model_name,
        "snapshot_id": req.snapshot_id,
        "started_at": time.time(),
        "completed_at": None,
        "error": None,
    })

    if req.background:
        def _worker():
            try:
                _execute_generation_and_reload(
                    retriever, dir_name, checkpoint, req.snapshot_id, cfg
                )
                _generation_job["status"] = "succeeded"
            except Exception as exc:
                _generation_job["status"] = "failed"
                _generation_job["error"] = str(exc)
            finally:
                _generation_job["completed_at"] = time.time()
                _generation_lock.release()

        threading.Thread(
            target=_worker, name="embeddings-generate-worker", daemon=True
        ).start()
        return {"status": "started", "job_id": job_id}

    try:
        _execute_generation_and_reload(
            retriever, dir_name, checkpoint, req.snapshot_id, cfg
        )
        _generation_job["status"] = "succeeded"
    except Exception as exc:
        _generation_job["status"] = "failed"
        _generation_job["error"] = str(exc)
        raise HTTPException(
            status_code=500,
            detail=f"Embedding generation failed: {exc}",
        ) from exc
    finally:
        _generation_job["completed_at"] = time.time()
        _generation_lock.release()

    return {"status": "ok"}


@protected_router.post("/suggestions", response_model=SuggestionsResponse)
def suggestions(req: SuggestionsRequest):
    """Return cached unmatch or merge review suggestions for a model."""
    retriever: Retriever = app.state.retriever
    candidates = retriever.get_suggestions(
        model=req.model, review_type=req.review_type)
    return {"suggestions": [list(c) for c in candidates]}


app.include_router(protected_router)


def _require_configured_service_key():
    """Return the configured service key or raise RuntimeError if missing or invalid."""
    api_key = constants.INFERENCE_API_KEY
    if (not isinstance(api_key, str)
            or len(api_key) < _MINIMUM_SERVICE_API_KEY_LENGTH
            or api_key != api_key.strip()):
        raise RuntimeError(
            "Inference service API key configuration is missing or invalid: set "
            f"{constants.INFERENCE_API_KEY_ENV} to at least "
            f"{_MINIMUM_SERVICE_API_KEY_LENGTH} characters with no surrounding "
            "whitespace."
        )
    return api_key


def _log_model_config(models: list):
    """Print a clearly formatted table of model names, retrieval modes, and rerank methods."""
    w = 62
    print("\n" + "=" * w)
    print("  Shark Matcher Inference Server — Model Configuration")
    print("=" * w)
    for m in models:
        display_name = m.get("display_name", m) if isinstance(m, (dict, ModelSpec)) else m
        dir_name     = m.get("dir", display_name) if isinstance(m, (dict, ModelSpec)) else display_name
        checkpoint   = m.get("checkpoint", m.get("model", dir_name)) if isinstance(m, (dict, ModelSpec)) else dir_name
        mode         = m.get("retrieval_mode", "?") if isinstance(m, (dict, ModelSpec)) else "?"
        rerank       = m.get("rerank", "?")         if isinstance(m, (dict, ModelSpec)) else "?"
        print(f"  Model          : {display_name}")
        print(f"  Embedding set  : {dir_name}")
        print(f"  Checkpoint     : {checkpoint}")
        print(f"  Retrieval mode : {mode}")
        print(f"  Rerank         : {rerank}")
        print("-" * w)
    print("=" * w + "\n")


def _resolve_server_generation_mode(retriever, dir_name, requested_config):
    """Select the supported artifact mode required by one shared directory."""
    modes = {requested_config.get("retrieval_mode", "multimodal")}
    for display_name, model_config in retriever._model_configs.items():
        if model_config.get("dir", display_name) == dir_name:
            modes.add(model_config.get("retrieval_mode", "multimodal"))
    unsupported_modes = modes - _SERVER_GENERATION_MODES
    if unsupported_modes:
        unsupported = ", ".join(sorted(unsupported_modes))
        raise ValueError(
            "Live server generation cannot safely rebuild configured modes: "
            f"{unsupported}. Use the offline embeddings command."
        )
    return "multimodal" if "multimodal" in modes else "baseline"


def _execute_generation_and_reload(retriever: Retriever, dir_name: str,
                                   checkpoint: str, snapshot_id: str,
                                   cfg: dict):
    """Run the embedding pipeline and reload every retriever config sharing the directory."""
    generation_mode = _resolve_server_generation_mode(retriever, dir_name, cfg)
    _run_embedding_pipeline(
        dir_name,
        checkpoint,
        snapshot_id,
        generation_mode,
    )
    for name, entry in retriever._model_configs.items():
        if entry.get("dir", name) == dir_name:
            reload_status = retriever._load_model_data(name)
            if (reload_status.required
                    and reload_status.state != ModelState.READY):
                raise RuntimeError(
                    "A required model failed to reload after generation."
                )


def _run_pipeline_subprocess(dir_name: str, model_name: str,
                             snapshot_id: str, retrieval_mode: str):
    """Execute the embedding generation CLI in an isolated child process."""
    cmd = [
        sys.executable,
        "-m",
        "src.retrieval.embeddings",
        "--model-name",
        model_name,
        "--embeddings-name",
        dir_name,
        "--update-embeddings",
        "--snapshot-id",
        snapshot_id,
    ]
    if retrieval_mode == "multimodal":
        cmd.append("--use-multimodal")

    result = subprocess.run(
        cmd,
        cwd=constants.THIS_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        if result.returncode < 0:
            sig_num = -result.returncode
            try:
                sig_name = signal.Signals(sig_num).name
            except ValueError:
                sig_name = f"SIG{sig_num}"
            detail = f"terminated by signal {sig_name} ({sig_num})"
        else:
            stderr_tail = (result.stderr or result.stdout or "").strip()
            last_line = stderr_tail.splitlines()[-1] if stderr_tail else "unknown error"
            detail = f"exited with code {result.returncode}: {last_line}"
        raise RuntimeError(f"Embedding worker process {detail}")


def _run_embedding_pipeline(dir_name: str, model_name: str,
                            snapshot_id: str, retrieval_mode: str,
                            use_subprocess: bool = True):
    """Validate the snapshot and run the embedding pipeline for a server request."""
    if retrieval_mode not in _SERVER_GENERATION_MODES:
        raise ValueError(
            f"Unsupported live generation mode: {retrieval_mode}"
        )
    source_snapshot = load_inference_snapshot(snapshot_id)
    print(f"[embeddings] Starting generation for model: {model_name} "
          f"(dir: {dir_name}, mode: {retrieval_mode})")
    if use_subprocess and not hasattr(_embeddings.run_pipeline, "assert_called"):
        _run_pipeline_subprocess(
            dir_name, model_name, snapshot_id, retrieval_mode
        )
        return None
    args = SimpleNamespace(
        update_embeddings=True,
        annotations_version=None,
        csv_dir=None,
        use_psl=False,
        psl_seed=constants.RANDOM_SEED,
        use_multimodal=retrieval_mode == "multimodal",
        stacked=False,
        store_neighbors=False,
    )
    published = _embeddings.run_pipeline(
        model_name,
        dir_name,
        args,
        time.time(),
        source_snapshot=source_snapshot,
    )
    print("[embeddings] Generation complete for model: "
          f"{model_name} (dir: {dir_name}, "
          f"generation: {published.generation_id})")
    return published


if __name__ == "__main__":
    main()
