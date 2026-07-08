import torch
from PIL import Image

from preprocess import load_image, transform


def test_transform_shape():
    img = Image.new("RGB", (80, 50), (10, 20, 30))
    t = transform(img)
    assert t.shape == (3, 64, 64)
    assert t.min() >= 0
    assert t.max() <= 1


def test_load_image_from_bytes(tmp_path):
    img = Image.new("RGB", (64, 64), (200, 10, 10))
    path = tmp_path / "x.png"
    img.save(path)
    t = load_image(path.read_bytes())
    assert t.shape == (3, 64, 64)
    assert torch.isfinite(t).all()
