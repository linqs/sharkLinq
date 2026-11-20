"""Pydantic validation schemas for training, model, and dataset configurations."""

from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class DataConfig(BaseModel):
    """Dataset input dimensions, normalization parameters, and sequence limits."""
    image_size: Tuple[int, int]
    normalization: Optional[Dict[str, Any]] = None
    channels: int = 3
    # For VLM
    max_len: int = 128

    model_config = ConfigDict(extra='ignore')


class LoRAConfig(BaseModel):
    """Low-Rank Adaptation constructor and module targeting settings."""
    constructor: Any = None
    options: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra='ignore')


class OptimizerConfig(BaseModel):
    """Optimizer constructor, learning rate, weight decay, and keyword options."""
    name: str = "AdamW"
    constructor: Any = None
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    options: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra='ignore')


class SchedulerConfig(BaseModel):
    """Learning rate scheduler constructor and keyword options."""
    name: str = "CosineAnnealingLR"
    constructor: Any = None
    options: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra='ignore')


class LossConfig(BaseModel):
    """Loss function name, constructor, and keyword options."""
    name: str
    constructor: Any = None
    options: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra='ignore')


class EarlyStoppingConfig(BaseModel):
    """Early stopping patience, delta thresholds, and optimization direction."""
    enabled: bool = True
    patience: int = 10
    min_delta: float = 0.0
    mode: str = "max"  # "max" for hits@k/acc, "min" for loss

    model_config = ConfigDict(extra='ignore')


class TrainingConfig(BaseModel):
    """Training loop execution, batch sizing, optimization, and evaluation settings."""
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    grad_checkpoint: bool = False
    epochs: int = 100

    early_stopping: Optional[EarlyStoppingConfig] = None

    optimizer: OptimizerConfig
    scheduler: Optional[SchedulerConfig] = None
    loss: LossConfig

    # Evaluator is currently a function object in config/config.py
    evaluator: Any = None

    freeze_backbone: bool = False
    num_augmentations: int = 1
    rebalance_augmentations: bool = False
    # P×K batch construction for metric learning.
    identities_per_batch: Optional[int] = None  # P: number of identities per batch.
    samples_per_identity: Optional[int] = None  # K: images per identity per batch.
    # Fusion-pipeline P×K aliases kept distinct so existing config keys are not dropped by extra='ignore'.
    p_value: Optional[int] = None
    k_value: Optional[int] = None
    # Linear LR warmup epochs (used by the multimodal trainer).
    warmup_epochs: int = 0
    freeze_bn_stats: bool = False
    include_original: bool = True
    num_workers: Optional[int] = None  # DataLoader worker processes. None -> resolved from src.config.config.NUM_WORKERS.
    # Max L2 norm for gradient clipping, applied to all model parameters.
    grad_clip_norm: float = 1.0
    # Evaluator keys driving checkpoint selection and early stopping (fall back to constants defaults when None).
    checkpoint_metric: Optional[str] = None
    early_stopping_metric: Optional[str] = None

    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True)


class ComponentConfig(BaseModel):
    """Named model component with constructor and initialization arguments."""
    name: str
    constructor: Any = None
    options: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True)


class ModelConfig(BaseModel):
    """Neural network backbone, projection head, and multimodal fusion topology."""
    # Optional for fusion configs building an end-to-end model rather than a (backbone, projection_head) pair.
    backbone: Optional[ComponentConfig] = None
    projection_head: Optional[ComponentConfig] = None

    # Fusion-specific architecture knobs set when type='fusion'.
    fusion_method: Optional[str] = None
    shared_dim: Optional[int] = None
    final_dim: Optional[int] = None
    num_heads: int = 4
    expert_dropout: float = 0.1
    projection_dropout: float = 0.1
    expert_depth: int = 1

    model_config = ConfigDict(extra='ignore')


class FullConfig(BaseModel):
    """Complete validated experiment configuration across training, model, and data."""
    training: TrainingConfig
    lora: Optional[LoRAConfig] = None
    model: ModelConfig
    data: Optional[DataConfig] = None

    # Optional type field for special configs like 'fusion'.
    type: Optional[str] = None

    # Fusion-pipeline blocks defined as raw dicts to prevent extra='ignore' from dropping them on round-trip.
    metadata_features: Optional[Dict[str, Any]] = None
    data_sources: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra='ignore')