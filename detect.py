from PIL import Image

from engine import Engine
from model import CLASSES
from preprocess import transform

TILE = 64


def pad_to_tile(img, tile=TILE):
    w, h = img.size
    nw = ((w + tile - 1) // tile) * tile
    nh = ((h + tile - 1) // tile) * tile
    if (nw, nh) == (w, h):
        return img
    canvas = Image.new("RGB", (nw, nh))
    canvas.paste(img, (0, 0))
    return canvas


def split_tiles(img, tile=TILE):
    img = pad_to_tile(img.convert("RGB"), tile)
    w, h = img.size
    tiles = []
    coords = []
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            crop = img.crop((x, y, x + tile, y + tile))
            tiles.append(transform(crop))
            coords.append((x, y))
    return tiles, coords, w, h


def classify_grid(engine: Engine, img, tile=TILE):
    orig_w, orig_h = img.size
    tensors, coords, w, h = split_tiles(img, tile)
    if not tensors:
        raise ValueError("image smaller than tile size")
    results, latency_ms = engine.predict_many(tensors)
    rows = h // tile
    cols = w // tile
    grid = []
    for r in range(rows):
        row = []
        for c in range(cols):
            row.append(results[r * cols + c]["label"])
        grid.append(row)
    return {
        "grid": grid,
        "rows": rows,
        "cols": cols,
        "tile": tile,
        "image_size": [orig_w, orig_h],
        "batch_latency_ms": round(latency_ms, 2),
        "classes": CLASSES,
    }


def paint_map(grid, tile=TILE):
    colors = {
        "water": (30, 90, 180),
        "forest": (30, 110, 40),
        "urban": (90, 90, 90),
        "agriculture": (180, 190, 60),
        "barren": (180, 140, 80),
        "cloud": (230, 230, 235),
    }
    rows = len(grid)
    cols = len(grid[0])
    img = Image.new("RGB", (cols * tile, rows * tile))
    for r, row in enumerate(grid):
        for c, label in enumerate(row):
            patch = Image.new("RGB", (tile, tile), colors.get(label, (0, 0, 0)))
            img.paste(patch, (c * tile, r * tile))
    return img
