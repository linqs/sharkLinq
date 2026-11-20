"""Unreadable images fail loudly, and can be found up front."""

import pytest
import torch
from PIL import Image

import src.individual_dataset


def _dataset(tmp_path, corrupt):
    """Build a three-image dataset where the names in `corrupt` are not valid image files."""
    images = tmp_path / "images"
    images.mkdir()
    names = ["a.jpg", "b.jpg", "c.jpg"]
    for name in names:
        if name in corrupt:
            (images / name).write_bytes(b"not an image")
        else:
            Image.new("RGB", (16, 16)).save(images / name)
    annotations = tmp_path / "annotations.tsv"
    annotations.write_text("".join(f"{i}\t{name}\t{i % 2}\n" for i, name in enumerate(names)))
    return src.individual_dataset.IndividualDataset(
        str(images), str(annotations), train_annotations_path=str(annotations),
        test_annotations_path=str(annotations), output_image_size=(16, 16),
        normalization=None, device=torch.device("cpu"))


def test_check_passes_when_every_image_is_readable(tmp_path):
    _dataset(tmp_path, corrupt=[]).check_images_readable()


def test_check_lists_every_unreadable_image(tmp_path):
    dataset = _dataset(tmp_path, corrupt=["a.jpg", "c.jpg"])
    with pytest.raises(RuntimeError, match=r"2 unreadable image\(s\).*a\.jpg.*c\.jpg"):
        dataset.check_images_readable()


def test_reading_an_unreadable_image_raises_instead_of_returning_a_placeholder(tmp_path):
    dataset = _dataset(tmp_path, corrupt=["b.jpg"])
    assert dataset[0][0].shape[0] == 3
    with pytest.raises(RuntimeError, match=r"Cannot read image .*b\.jpg"):
        dataset[1]
