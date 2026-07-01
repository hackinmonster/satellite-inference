from PIL import Image

from model import CLASSES
from train import make_tile

# 3x3 mosaic so we can test the sliding-window path
layout = [
    ["water", "forest", "urban"],
    ["agriculture", "barren", "cloud"],
    ["forest", "water", "urban"],
]

tile = 64
canvas = Image.new("RGB", (tile * 3, tile * 3))
for r, row in enumerate(layout):
    for c, name in enumerate(row):
        canvas.paste(make_tile(name, tile), (c * tile, r * tile))
canvas.save("samples/mosaic.png")
print("wrote samples/mosaic.png", canvas.size)
