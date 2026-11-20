# Models

The `src/models` package holds the neural-network building blocks and image augmentations that the trainer and the embedding pipeline assemble.

## Modules

| File | Responsibility |
| --- | --- |
| `vit_backbone.py` | `ViTBackbone`: a Hugging Face Vision Transformer that returns CLS-token embeddings |
| `miewid_backbone.py` | `MiewIDBackbone`: a wrapper that loads the Hugging Face MiewID model as a feature extractor |
| `mlp.py` | `MLP`: a feed-forward projection or classification head with configurable depth and dropout |
| `arcface_heads.py` | ArcFace margin heads: `ArcMarginProduct`, `ArcMarginProduct_subcenter`, `ArcFaceLossAdaptiveMargin`, `ArcFaceSubCenterDynamic` |
| `augmentations.py` | `AUGMENTATION_CONSTRUCTORS`, which maps augmentation names to torchvision transforms, plus the custom `CLAHE` transform |

## Entry Points

None. This package is imported, never run.

## Boundaries and Dependencies

- `src/config/config.py` registers these classes in its constructor tables, so configurations refer to them by name.
- `src/core/utils.py` builds backbones for loading checkpoints and embedding models.
- `vit_backbone.py` imports `src/config/config_schema.py`; no other module here imports matcher code.
- Both backbones download pretrained weights from Hugging Face on first use.
- `MiewIDBackbone` loads its model with `trust_remote_code`.
- Unit tests for this package live in `tests/unit/models` and `tests/unit/learning`.
