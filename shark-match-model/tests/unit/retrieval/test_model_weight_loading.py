"""Tests that loading a fine-tuned model never silently falls back to random weights."""

import inspect
import json

import pytest
import torch

from src.core import constants
from src.core import utils


class _StubBackbone(torch.nn.Module):
    def __init__(self, **_options):
        super().__init__()
        self.layer = torch.nn.Linear(2, 2)

    def get_output_dim(self):
        return 2


@pytest.fixture(name="model_dir")
def model_dir_fixture(tmp_path, monkeypatch):
    """A trained-models root holding one model directory with a config and no weights."""
    directory = tmp_path / "some-model"
    directory.mkdir()
    config = {"model": {"backbone": {"constructor": "Stub", "options": {}},
                        "projection_head": {"name": "MLP", "options": {}}}}
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(constants, "TRAINED_MODELS_DIR", str(tmp_path))
    monkeypatch.setitem(utils.ALL_BACKBONE_CONSTRUCTORS, "Stub", _StubBackbone)
    return directory


def test_embedding_loader_defaults_to_the_shipped_best_checkpoint():
    parameter = inspect.signature(utils.load_embedding_model).parameters["backbone_prefix"]
    assert parameter.default == "best"


def test_missing_backbone_weights_raise_naming_the_files(model_dir):
    with pytest.raises(FileNotFoundError, match="best-backbone-state.pt"):
        utils.load_model("some-model")


def test_missing_weights_for_another_prefix_name_that_prefix(model_dir):
    (model_dir / "best-backbone-state.pt").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="last-backbone-state.pt"):
        utils.load_model("some-model", backbone_prefix="last")
