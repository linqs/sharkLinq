"""Configuration registry for the multimodal late-fusion pipeline."""

# Default metadata feature specification used by integrated and post-hoc late-fusion paths.
DEFAULT_METADATA_FEATURES = {
    "use_soft_encoding": False,
    "include_sex": True,
    "include_type": True,
    "include_quality": True,
    "include_location": True,
    "include_length": True,
    "include_time": True,
}
