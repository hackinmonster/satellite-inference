from PIL import Image

from detect import pad_to_tile, split_tiles


def test_pad_to_next_multiple():
    img = Image.new("RGB", (70, 70), (1, 2, 3))
    padded = pad_to_tile(img, tile=64)
    assert padded.size == (128, 128)


def test_already_aligned_not_padded():
    img = Image.new("RGB", (128, 64), (1, 2, 3))
    padded = pad_to_tile(img, tile=64)
    assert padded.size == (128, 64)


def test_split_covers_padded_area():
    img = Image.new("RGB", (70, 70), (1, 2, 3))
    tiles, coords, w, h = split_tiles(img, tile=64)
    assert (w, h) == (128, 128)
    assert len(tiles) == 4
    assert tiles[0].shape == (3, 64, 64)
