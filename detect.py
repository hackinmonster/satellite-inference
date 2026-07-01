from PIL import Image

from engine import Engine
from model import CLASSES
from preprocess import transform

TILE = 64

# leftover edges smaller than TILE are dropped for now
def split_tiles(img, tile=TILE):
    img = img.convert("RGB")
    w, h = img.size
    tiles = []
    coords = []
    for y in range(0, h - tile + 1, tile):
        for x in range(0, w - tile + 1, tile):
            crop = img.crop((x, y, x + tile, y + tile))
            tiles.append(transform(crop))
            coords.append((x, y))
    return tiles, coords, w, h


def classify_grid(engine: Engine, img, tile=TILE):
    tensors, coords, w, h = split_tiles(img, tile)
    if not tensors:
        raise ValueError("image smaller than tile size")
    results, latency_ms = engine.predict_many(tensors)
    rows = (h - tile) // tile + 1
    cols = (w - tile) // tile + 1
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
        "image_size": [w, h],
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
