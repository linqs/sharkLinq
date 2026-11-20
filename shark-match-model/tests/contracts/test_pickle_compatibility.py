"""Tests that stored pickles naming earlier module paths still resolve to the current classes."""

import importlib
import pickle

import pytest

import src.individual_dataset
import src.learning.augmented_dataset
import src.multimodal.fusion_dataset
from src.core import utils


@pytest.mark.parametrize("old_name, new_name", [
    (old, new) for old, new in utils.PICKLE_MODULE_ALIASES.items() if new is not None
])
def test_old_module_name_imports_the_replacement_module(old_name, new_name):
    """Each aliased name imports the replacement module itself and leaves its spec intact."""
    replacement = importlib.import_module(new_name)

    assert importlib.import_module(old_name) is replacement
    assert replacement.__spec__.name == new_name


@pytest.mark.parametrize("module_name, class_name, expected", [
    ("src.sharkfin_dataset", "SharkfinDataset", src.individual_dataset.IndividualDataset),
    ("backend.individual_dataset", "IndividualDataset", src.individual_dataset.IndividualDataset),
    ("src.learning.contrastive_learning_sharkfin_dataset", "ContrastiveLearningSharkfinDataset",
     src.learning.augmented_dataset.AugmentedDataset),
    ("backend.learning.augmented_dataset", "ContrastiveLearningIndividualDataset",
     src.learning.augmented_dataset.AugmentedDataset),
    ("src.experiments.multimodal.fusion_dataset", "FusionDataset",
     src.multimodal.fusion_dataset.FusionDataset),
])
def test_pickles_naming_old_paths_load_current_classes(module_name, class_name, expected):
    """A pickle that references a class by an earlier module path unpickles to the current class."""
    payload = f"c{module_name}\n{class_name}\n.".encode("ascii")

    assert pickle.loads(payload) is expected
