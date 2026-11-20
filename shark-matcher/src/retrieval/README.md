# Retrieval

The `src/retrieval` package builds embeddings from a trained model, publishes them as versioned generations, and serves neighbor lookups, evaluation, and review suggestions from them.

## Modules

| File | Responsibility |
| --- | --- |
| `embeddings.py` | The embedding-generation command and `run_pipeline`, which builds and publishes one generation; also incremental update, dataset-only, and suggestions-only maintenance |
| `generation_store.py` | Storage layout of an embedding set: publish, resolve the active generation, validate, retain, and roll back |
| `inference_snapshot.py` | `load_inference_snapshot`: validates an app-published snapshot (manifest, hashes, row counts, CSV shape, file boundary) |
| `neighbors.py` | Computes and updates image and centroid nearest neighbors, and applies PSL reranking to cached neighbors |
| `retrieval.py` | `Retriever`: loads a published generation per model and answers neighbor, statistics, and suggestion lookups; `validate_model_checkpoint` |
| `rerank.py` | k-reciprocal reranking: `KReciprocalReranker` and `kreciprocal_rerank` |
| `suggestions.py` | Merge and unmatch review suggestions, cached as `suggestions.pt` in the embedding set |
| `evaluation.py` | Hits@K, CMC, mAP, MRR, and average-rank evaluation of stored embeddings |
| `fine_evaluation.py` | Per-query comparison and cross-model failure analysis; `evaluation.py --fine-eval` runs it, and `evaluation.py` also uses its CSV-pinning helper |
| `train_evaluation.py` | Prints Hits@K and loss tables from a trained model's training history |

## Entry Points

Run from the repository root; `--help` lists every option.

```bash
python3 -m src.retrieval.embeddings --help
python3 -m src.retrieval.evaluation --help
python3 -m src.retrieval.suggestions --help
python3 -m src.retrieval.train_evaluation --help
```

- `embeddings` builds or updates an embedding set; flags select the PSL, multimodal, and stacked variants, maintenance publications, `--dry-run`, and `--rollback`.
- `evaluation` scores a stored embedding set and can apply the PSL, multimodal, stacked, and k-reciprocal views.
- `suggestions` prints merge and unmatch suggestions; `--save-suggestions` refreshes the cache.
- `train_evaluation` reads training history written by the trainer.
- `neighbors.py`, `retrieval.py`, `generation_store.py`, `inference_snapshot.py`, and `rerank.py` are libraries.
- `fine_evaluation.py` runs a fixed debugging comparison when started directly; use `evaluation --fine-eval` instead.

## Boundaries and Dependencies

- The root `README.md` sections "Generate Embeddings" and "Evaluate Embeddings" describe the generation workflow and its flags.
- `generation_store.py` owns every artifact filename per retrieval mode (`baseline`, `psl`, `multimodal`, `stacked`) and the generation layout; do not rebuild either elsewhere.
- Each embedding set holds a `generations/` folder and a `current.json` pointer. Readers switch only after validation, and publication keeps the active generation plus one prior generation.
- Tools that read a set by name use `resolve_active_directory` in `generation_store.py`, never a path built from the set name.
- `MODEL_ALIASES` maps request model names to embedding sets as comma-separated `ALIAS:NAME` pairs; unset, it is `ViT:vit_triplet,Miew:miew`.
- Wire models come from `src/contracts`; metadata lookups go through `src/csv_store`.
- `src/psl` and `src/multimodal` are imported lazily, only when their flags are used.
- `embeddings.py` imports `src/cli/convert_data.py`.
- `src/api/server.py` is the main consumer, through `Retriever`, `run_pipeline`, `load_inference_snapshot`, and `validate_model_checkpoint`.
- Tests live in `tests/unit/retrieval` and `tests/contracts`.
