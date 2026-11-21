"""Handles interaction with the stored embeddings in order to retrieve relevant data relating to model predictions."""

import os

import numpy as np
import torch

from src.core import constants
from src.contracts import ModelDiagnostic, ModelSpec, ModelState, ModelStatus
import src.individual_dataset
import src.learning.augmented_dataset
import src.retrieval.generation_store
import src.retrieval.rerank
from src.core import utils


def main():
    """Run ad-hoc debugging for the Retriever module."""
    pass


def _parse_model_aliases() -> dict:
    """Build the model alias mapping from the MODEL_ALIASES environment variable or fall back to defaults."""
    raw = os.getenv("MODEL_ALIASES", "").strip()
    if not raw:
        return {"ViT": "vit_triplet", "Miew": "miew"}
    mapping = {}
    for pair in raw.split(","):
        if ":" in pair:
            alias, name = pair.split(":", 1)
            mapping[alias.strip()] = name.strip()
    return mapping


def validate_model_checkpoint(spec: ModelSpec, trained_models_dir: str | None = None) -> None:
    """Validate that a model specification's checkpoint exists, failing fast on invalid configurations."""
    checkpoint = spec.checkpoint.strip().lower()
    if checkpoint in constants.PRECANNED_MODELS:
        return
    models_dir = trained_models_dir or constants.TRAINED_MODELS_DIR
    config_path = os.path.join(models_dir, spec.checkpoint, "config.json")
    if not os.path.exists(config_path):
        precanned = ", ".join(sorted(constants.PRECANNED_MODELS))
        raise RuntimeError(
            f"Configured model '{spec.display_name}' specifies checkpoint '{spec.checkpoint}', "
            f"but no config file was found at '{config_path}' and it is not a known pre-canned model ({precanned})."
        )


# pylint: disable=too-many-instance-attributes
class Retriever:
    """Retriever object storing parameters to be used in retrieval."""

    MODEL_MAPPING = _parse_model_aliases()

    def __init__(self, preload_models, use_psl=False, use_multimodal=False,
                 use_stacked=False, use_kreciprocal=False,
                 embeddings_root=None, csv_dir_override=None,
                 validate_checkpoints=False):
        """Initialize the Retriever and configure retrieval parameters and models to preload."""
        self.device = utils.get_torch_device()
        self.embeddings_root = embeddings_root or constants.EMBEDDINGS_DIR
        self.csv_dir_override = (
            utils.resolve_directory_path(csv_dir_override, constants.DB_DATA_DIR)
            if csv_dir_override else None
        )
        # Global flags are the defaults for bare-string specs in preload_models.
        self.use_psl = use_psl
        self.use_multimodal = use_multimodal
        self.use_stacked = use_stacked
        self.use_kreciprocal = use_kreciprocal

        if not preload_models:
            raise ValueError(
                "preload_models cannot be empty. At least one model must be specified."
            )

        # Per-model config keyed by display_name, normalized into ModelSpec contracts.
        self._model_configs = {}
        for item in preload_models:
            if isinstance(item, ModelSpec):
                spec = item
            elif isinstance(item, dict):
                spec = ModelSpec(**item)
            else:
                mode = src.retrieval.generation_store.resolve_retrieval_mode(
                    use_psl=self.use_psl,
                    use_multimodal=self.use_multimodal,
                    use_stacked=self.use_stacked,
                )
                rerank = "kreciprocal" if self.use_kreciprocal else "none"
                spec = ModelSpec(
                    display_name=str(item),
                    embedding_set=str(item),
                    checkpoint=str(item),
                    retrieval_mode=mode,
                    rerank=rerank,
                    required=True,
                )
            self._model_configs[spec.display_name] = spec

        if validate_checkpoints:
            for spec in self._model_configs.values():
                if spec.required:
                    validate_model_checkpoint(spec)

        # Map dir -> first display_name so MODEL_MAPPING aliases (which resolve
        # to a dir name) can be looked up in _model_configs.
        self._dir_to_first_display = {}
        for dn, cfg in self._model_configs.items():
            d = cfg.get("dir", dn)
            if d not in self._dir_to_first_display:
                self._dir_to_first_display[d] = dn

        # The first model is the default (home-page stats, unspecified requests).
        first = preload_models[0]
        self.default_model_name = (
            first.get("display_name") or first.get("name", "")
            if isinstance(first, (dict, ModelSpec)) else first
        )

        self.loaded_models = {}
        self.allowed_models = set(self._model_configs)
        self._model_statuses = {
            display_name: ModelStatus(
                display_name=display_name,
                embedding_set=config["dir"],
                required=config["required"],
                state=(
                    ModelState.NOT_ATTEMPTED
                    if config["required"] else ModelState.DISABLED
                ),
            )
            for display_name, config in self._model_configs.items()
        }

        # Pre-load all specified models.
        for display_name, config in self._model_configs.items():
            if config["required"] and display_name not in self.loaded_models:
                self._load_model_data(display_name)

        # Reference table for retrieval methods.
        self.retrieval_methods = {
            0: ("image_image", self.get_image_neighbors),
            1: ("image_image_average", self.get_image_neighbors_average),
            2: ("loo_centroid", self.get_loo_centroid_neighbors),
        }

    def get_neighbors(self, retrieval_method, query_image_id, model=None):
        """Get neighbors for a query image using the specified retrieval method."""
        retrieval_function = self.retrieval_methods[retrieval_method]
        return retrieval_function[1](query_image_id, model=model)

    def get_image_neighbors(self, query_image_id, model=None):
        """Get nearest-image neighbors for a query image via on-the-fly NumPy L2 or PSL neighbor table."""
        data = self._get_model_data(model)
        if not data:
            return {"ids": [], "distances": [], "labels": []}

        image_neighbors = data.get("image_neighbors")
        if image_neighbors is not None:
            query_image_file = data["dataset"].get_filename_by_id(query_image_id)
            if query_image_file is None:
                print(f"WARNING: Query image with id {query_image_id} not found in the dataset.")
                return {"ids": [], "distances": [], "labels": []}

            neighbor_result = image_neighbors.get(query_image_file)
            if neighbor_result is None:
                print(f"WARNING: Neighbors for {query_image_file} "
                      "not found in neighbor reference dictionary.")
                return {"ids": [], "distances": [], "labels": []}

            return {
                "ids": [int(id_val) for id_val in neighbor_result["matches"]],
                "distances": [float(d) for d in neighbor_result["match_distances"]],
                "labels": [int(lbl) for lbl in neighbor_result["match_labels"]],
            }

        id_to_index = data.get("id_to_index")
        if id_to_index is None:
            id_to_index = {int(id_val): idx for idx, id_val in enumerate(data["ids"])}
            data["id_to_index"] = id_to_index

        query_idx = id_to_index.get(int(query_image_id))
        if query_idx is None:
            print(f"WARNING: Query image with id {query_image_id} not found in the dataset.")
            return {"ids": [], "distances": [], "labels": []}

        embeddings = data["embeddings"]
        # pylint: disable=protected-access
        dists = src.retrieval.rerank._pairwise_l2(
            embeddings[query_idx:query_idx + 1], embeddings
        )[0]
        dists[query_idx] = 0.0
        order = np.argsort(dists, kind="stable")
        ids_arr = data["ids"]
        labels_arr = data["labels"]
        return {
            "ids": [int(ids_arr[i]) for i in order],
            "distances": [float(dists[i]) for i in order],
            "labels": [int(labels_arr[i]) for i in order],
        }

    def get_image_neighbors_average(self, query_image_id, model=None):
        """Retrieve neighbor images ranked by the average distance of their individual's images to the query."""
        # First get the image neighbors.
        image_neighbors = self.get_image_neighbors(query_image_id, model=model)

        # Store the image ids and distances by individual for each individual.
        individual_dict = {}
        for index, id_val in enumerate(image_neighbors["ids"]):
            distance = image_neighbors["distances"][index]
            label = image_neighbors["labels"][index]

            # Add the distance to the running total distance metric for each individual.
            if individual_dict.get(label) is None:
                individual_dict[label] = [(id_val, distance)]
            else:
                individual_dict[label].append((id_val, distance))

        # Compute average individual distances for all individuals labeled the same as the query,
        # but excluding the query (this is not a result).
        individual_average_distances = {}
        for key, value in individual_dict.items():
            individual_distances = []
            for item in value:
                # Don't include query, where the distance is zero, in the neighbor
                # average computation.
                if item[0] != query_image_id:
                    individual_distances.append(item[1])

            # Compute the mean of the distances for this individual.
            individual_average = np.mean(np.array(individual_distances))

            individual_average_distances[key] = individual_average

        # Sort individuals by their average distances.
        individual_average_distances = dict(sorted(individual_average_distances.items(),
                                              key=lambda item: item[1]))

        # Build a return dict ordered by individual distances.
        return_dict = {"ids": [],
                       "distances": [],
                       "labels": []}

        # For each individual, add all of the images associated with it to the results.
        for label, _ in individual_average_distances.items():
            for neighbor in individual_dict[label]:
                # individual_dict values are already sorted by distance from construction.
                return_dict["ids"].append(neighbor[0])
                return_dict["distances"].append(neighbor[1])
                return_dict["labels"].append(label)

        return return_dict

    def get_loo_centroid_neighbors(self, query_image_id, model=None):
        """Retrieve neighbors using leave-one-out (LOO) centroid retrieval, recomputing the query's individual centroid without the query."""
        data = self._get_model_data(model)
        if not data:
            return {"ids": [], "distances": [], "labels": []}

        reranker = data.get('kreciprocal_reranker')
        if reranker is not None:
            return reranker.get_neighbors(query_image_id)

        # Lazily build per-individual sum/count/centroid state.
        if 'loo_state' not in data:
            self._build_loo_state(data)
        state = data['loo_state']

        q_idx = state['id_to_emb_idx'].get(query_image_id)
        if q_idx is None:
            print(f"WARNING: Query image id {query_image_id} not in dataset.")
            return {"ids": [], "distances": [], "labels": []}

        q_emb = data['embeddings'][q_idx]
        q_label = int(data['labels'][q_idx])

        # Distances from query to every standard renormalized centroid.
        dists = np.linalg.norm(state['centroid_matrix'] - q_emb, axis=1)

        # Override the query's own individual's distance with the LOO distance.
        n_L = state['count_by_label'][q_label]
        if n_L >= 2:
            loo_sum = state['sum_by_label'][q_label] - q_emb
            loo_norm = np.linalg.norm(loo_sum)
            if loo_norm > 0:
                loo_centroid = loo_sum / loo_norm
                dists[state['label_to_idx'][q_label]] = float(
                    np.linalg.norm(loo_centroid - q_emb))
        # If n_L == 1, LOO is undefined; the standard centroid distance stays
        # (which is 0 since the centroid equals the single image embedding).
        # min_instances >= 2 at the eval layer rules this out for queries.

        # Sort centroids by distance and expand each to its member images.
        sorted_centroid_indices = np.argsort(dists)
        label_order = state['label_order']
        images_by_label = state['images_by_label']
        ids_arr = data['ids']
        labels_arr = data['labels']

        ids_out = []
        distances_out = []
        labels_out = []
        for c_idx in sorted_centroid_indices:
            label = label_order[c_idx]
            d = float(dists[c_idx])
            for img_idx in images_by_label[label]:
                ids_out.append(int(ids_arr[img_idx]))
                distances_out.append(d)
                labels_out.append(int(labels_arr[img_idx]))

        return {
            "ids": ids_out,
            "distances": distances_out,
            "labels": labels_out,
        }

    def get_suggestions(self, model=None, review_type='unmatch_suggestions'):
        """Return cached unmatch or merge suggestions for the specified model."""
        if review_type not in ('unmatch_suggestions', 'merge_suggestions'):
            return []
        data = self._get_model_data(model)
        if not data:
            return []
        return data.get(review_type, [])

    def get_dataset(self, model=None):
        """Externally provide this retrieval object's dataset."""
        data = self._get_model_data(model)
        if data:
            return data['dataset']
        return None

    def get_model_statuses(self):
        """Return model statuses in configured order."""
        return list(self._model_statuses.values())

    def get_loaded_display_names(self) -> list:
        """Return the display names of all loaded models, in sorted order."""
        return sorted(self.loaded_models.keys())

    def to_dict(self, compact=False, model=None):
        """Return the retriever state as a plain dictionary."""
        native_name = self._resolve_model(model)
        data = self._get_model_data(model)

        if not data:
            return {"model_name": native_name,
                    "device": self.device,
                    "image_neighbors": {},
                    "embeddings_metadata": {"start_time": None, "update_time": None}
            }

        item_dict = {"model_name": native_name,
                "device": self.device,
                "image_neighbors": data['image_neighbors'],
                "embeddings_metadata": data['embeddings_metadata']
        }

        return item_dict

    def print(self):
        """Print object as dict."""
        print(self.to_dict())

    def _resolve_model(self, model_alias):
        """Resolve a model alias, directory name, or configured display name."""
        if model_alias in self.MODEL_MAPPING:
            # MODEL_MAPPING resolves to a dir name; map back to the display_name for that dir.
            dir_name = self.MODEL_MAPPING[model_alias]
            return self._dir_to_first_display.get(dir_name, dir_name)
        if model_alias is None:
            return self.default_model_name
        if model_alias in self._model_configs:
            return model_alias
        # Accept a raw embedding-set directory name as a fallback.
        if model_alias in self._dir_to_first_display:
            return self._dir_to_first_display[model_alias]
        return model_alias

    def _get_model_data(self, model):
        """Return loaded data for a resolved and allowed model."""
        native_name = self._resolve_model(model)

        # Check if the model is allowed to be loaded by the user.
        if native_name not in self.allowed_models:
            print(f"WARNING: Model '{native_name}' was requested but not "
                  "specified at startup. Rejecting request.")
            return None

        status = self._model_statuses[native_name]
        if status.state in {
                ModelState.FAILED,
                ModelState.DISABLED,
                ModelState.LOADING,
        }:
            return None

        # Lazy loading: If the model was not attempted, try to load it now.
        if status.state == ModelState.NOT_ATTEMPTED:
            self._load_model_data(native_name)

        if self._model_statuses[native_name].state != ModelState.READY:
            return None
        return self.loaded_models[native_name]

    def _set_model_status(self, model_name, state, error=None):
        """Update the load status for a configured model."""
        config = self._model_configs[model_name]
        status = ModelStatus(
            display_name=model_name,
            embedding_set=config["dir"],
            required=config["required"],
            state=state,
            error=error,
        )
        self._model_statuses[model_name] = status
        return status

    @staticmethod
    def _diagnose_model_load_failure(error):
        """Convert a model-load exception to a stable path-free diagnostic."""
        if isinstance(error, FileNotFoundError):
            return ModelDiagnostic(
                code="artifact_missing",
                message="Required model data is missing.",
            )
        if isinstance(
                error,
                src.retrieval.generation_store.GenerationValidationError,
        ):
            return ModelDiagnostic(
                code="generation_invalid",
                message="The active model generation is invalid.",
            )
        if isinstance(error, (EOFError, AttributeError, RuntimeError, TypeError)):
            return ModelDiagnostic(
                code="artifact_incompatible",
                message="Required model data is incompatible.",
            )
        return ModelDiagnostic(
            code="load_failed",
            message="Required model data could not be loaded.",
        )

    def _load_model_data(self, model_name):
        """Load one model from a single resolved embedding generation."""
        print(f"Loading data for model: {model_name}")
        config = self._model_configs.get(
            model_name, {"retrieval_mode": "multimodal", "rerank": "kreciprocal"})
        # dir may differ from model_name when multiple configs share one embeddings directory.
        dir_name = config.get("dir", model_name)
        if not config.get("required", True):
            return self._set_model_status(model_name, ModelState.DISABLED)
        self.loaded_models.pop(model_name, None)
        self._set_model_status(model_name, ModelState.LOADING)
        try:
            artifact_layout = (
                src.retrieval.generation_store.get_generation_artifact_layout(
                    config["retrieval_mode"])
            )
        except src.retrieval.generation_store.GenerationValidationError:
            print(f"WARNING: Unknown retrieval_mode '{config['retrieval_mode']}'. "
                  "Falling back to 'multimodal'.")
            artifact_layout = (
                src.retrieval.generation_store.get_generation_artifact_layout(
                    "multimodal")
            )
        use_kreciprocal = config["rerank"] == "kreciprocal"

        model_data = {}

        try:
            embedding_set_root = os.path.join(self.embeddings_root, dir_name)
            resolved_generation = (
                src.retrieval.generation_store.resolve_active_generation(
                    embedding_set_root)
            )
            model_dir = resolved_generation.directory
            model_data['generation_id'] = resolved_generation.generation_id
            model_data['snapshot_id'] = (
                resolved_generation.manifest.snapshot_id
                if resolved_generation.manifest is not None else None
            )

            # Load the cached dataset on CPU to avoid device-specific deserialization.
            model_data['dataset'] = torch.load(
                os.path.join(model_dir, "stored_dataset.pt"),
                map_location="cpu", weights_only=False)
            ds = model_data['dataset']
            if self.csv_dir_override:
                ds.csv_dir = self.csv_dir_override
            elif getattr(ds, 'csv_dir', None):
                ds.csv_dir = utils.resolve_relative_csv_dir(ds.csv_dir)
            model_data['ids'] = model_data['dataset'].get_ids()
            model_data['id_to_index'] = {
                int(id_val): idx for idx, id_val in enumerate(model_data['ids'])
            }
            model_data['filenames'] = model_data['dataset'].get_filenames()
            model_data['labels'] = model_data['dataset'].get_labels().cpu().detach().numpy()

            # Load image embeddings; load neighbor dicts only for PSL modes that lack a metric space.
            model_data['embeddings'] = torch.load(
                os.path.join(model_dir, artifact_layout.image_embeddings),
                map_location="cpu", weights_only=False).cpu().detach().numpy()

            if artifact_layout.retrieval_mode in {"psl", "stacked"}:
                model_data['image_neighbors'] = torch.load(
                    os.path.join(model_dir, artifact_layout.image_neighbors),
                    map_location="cpu", weights_only=False)
            else:
                model_data['image_neighbors'] = None

            # Load centroid embeddings and labels when available.
            try:
                model_data['centroid_embeddings'] = torch.load(
                    os.path.join(model_dir, artifact_layout.centroid_embeddings),
                    map_location="cpu", weights_only=False).cpu().detach().numpy()
                model_data['centroid_neighbors'] = None
                model_data['centroid_labels'] = torch.load(
                    os.path.join(model_dir, artifact_layout.centroid_labels),
                    map_location="cpu", weights_only=False)
            except FileNotFoundError:
                print(f"WARNING: Centroid embeddings not found for model {model_name} "
                      f"(dir: {dir_name}). Centroid retrieval will not work.")
                model_data['centroid_embeddings'] = None
                model_data['centroid_neighbors'] = None
                model_data['centroid_labels'] = None

            # Load the embeddings metadata file, if available.
            metadata_path = os.path.join(model_dir, "embeddings_metadata.json")
            model_data['embeddings_metadata'] = (
                utils.load_json_file(metadata_path) if os.path.exists(metadata_path) else None)

            if model_data['embeddings_metadata'] is None:
                # A generation being built has no metadata until the publisher finishes it.
                if not os.path.basename(model_dir).startswith(
                        src.retrieval.generation_store.STAGING_PREFIX):
                    print(f"WARNING: Embeddings metadata not found for {model_name} "
                          f"(dir: {dir_name}).")
                model_data['embeddings_metadata'] = {"start_time": None, "update_time": None}

            # Lazily load cached unmatch and merge suggestions, defaulting to empty lists on failure.
            try:
                import src.retrieval.suggestions as _suggestions  # pylint: disable=import-outside-toplevel
                payload = _suggestions.load_suggestions(model_dir)
                model_data['unmatch_suggestions'] = payload['unmatch_suggestions']
                model_data['merge_suggestions'] = payload['merge_suggestions']
            except Exception as suggestions_err:  # pylint: disable=broad-exception-caught
                print(f"WARNING: Failed to load suggestions.pt for {model_name}: "
                      f"{suggestions_err}. Review tab will see no suggestions.")
                model_data['unmatch_suggestions'] = []
                model_data['merge_suggestions'] = []

            self.loaded_models[model_name] = model_data
            status = self._set_model_status(model_name, ModelState.READY)

            if use_kreciprocal:
                # Build after marking READY so _get_model_data can access loaded state.
                # Only method 2 (LOO centroid) — k-reciprocal hurts method 0
                # (image-image) in this domain per evaluation results.
                try:
                    import src.retrieval.rerank as _rerank  # pylint: disable=import-outside-toplevel
                    model_data['kreciprocal_reranker'] = _rerank.KReciprocalReranker(
                        self, base_method=2, model=model_name)
                except Exception as rerank_err:  # pylint: disable=broad-exception-caught
                    print(f"WARNING: Failed to build KReciprocalReranker for "
                          f"{model_name}: {rerank_err}. Method 2 will fall back "
                          f"to plain LOO centroid retrieval.")
                    model_data['kreciprocal_reranker'] = None

            return status
        except Exception as error:  # pylint: disable=broad-exception-caught
            diagnostic = self._diagnose_model_load_failure(error)
            self.loaded_models.pop(model_name, None)
            status = self._set_model_status(
                model_name, ModelState.FAILED, diagnostic
            )
            print("\n" + "="*80)
            print(f"[ERROR] EMBEDDING GENERATION UNAVAILABLE FOR MODEL: "
                  f"'{model_name}' (dir: '{dir_name}')")
            print(f"{diagnostic.code}: {diagnostic.message}")
            mode_flags = src.retrieval.generation_store.retrieval_mode_cli_flags(
                config["retrieval_mode"])
            generate_command = " ".join([
                "python3 -m src.retrieval.embeddings",
                "--model-name", config.get("checkpoint", dir_name),
                "--embeddings-name", dir_name,
                "--annotations-version <annotations-version>",
                *mode_flags,
            ])
            print("To fix this, point SHARK_MODELS at an existing embedding set, "
                  "or generate this one by running:")
            print(f"    {generate_command}")
            print("The server will continue to start to allow browsing the "
                  "dataset, but matching features will not work.")
            print("="*80 + "\n")
            return status

    def _build_loo_state(self, data):
        """Build the per-individual sum/count/centroid state used by get_loo_centroid_neighbors."""
        embeddings = data['embeddings']
        ids = data['ids']
        labels = data['labels']
        _, d = embeddings.shape

        sum_by_label = {}
        count_by_label = {}
        images_by_label = {}
        id_to_emb_idx = {}
        for i, (img_id, lbl) in enumerate(zip(ids, labels)):
            lbl_int = int(lbl)
            id_to_emb_idx[img_id] = i
            if lbl_int not in sum_by_label:
                sum_by_label[lbl_int] = np.zeros(d, dtype=np.float64)
                count_by_label[lbl_int] = 0
                images_by_label[lbl_int] = []
            sum_by_label[lbl_int] += embeddings[i]
            count_by_label[lbl_int] += 1
            images_by_label[lbl_int].append(i)

        # Standard renormalized centroid per individual, sorted by label for
        # deterministic row order.
        label_order = sorted(sum_by_label.keys())
        label_to_idx = {lbl: i for i, lbl in enumerate(label_order)}
        centroid_matrix = np.zeros((len(label_order), d), dtype=np.float64)
        for i, lbl in enumerate(label_order):
            c = sum_by_label[lbl] / count_by_label[lbl]
            norm = np.linalg.norm(c)
            if norm > 0:
                c = c / norm
            centroid_matrix[i] = c

        data['loo_state'] = {
            'sum_by_label': sum_by_label,
            'count_by_label': count_by_label,
            'images_by_label': images_by_label,
            'label_order': label_order,
            'label_to_idx': label_to_idx,
            'centroid_matrix': centroid_matrix,
            'id_to_emb_idx': id_to_emb_idx,
        }
        print(f"[LOO] Built per-individual centroid state for "
              f"{len(label_order)} individuals.")


if __name__ == "__main__":
    main()
