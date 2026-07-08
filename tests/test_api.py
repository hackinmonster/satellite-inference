from fastapi.testclient import TestClient

from server import app


def test_health_and_predict():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["ok"] is True

        with open("samples/urban.png", "rb") as f:
            r = client.post("/predict", files={"file": ("urban.png", f, "image/png")})
        assert r.status_code == 200
        body = r.json()
        assert body["label"] == "urban"
        assert body["confidence"] > 0.9


def test_grid_on_mosaic():
    with TestClient(app) as client:
        with open("samples/mosaic.png", "rb") as f:
            r = client.post("/predict/grid", files={"file": ("mosaic.png", f, "image/png")})
        assert r.status_code == 200
        body = r.json()
        assert body["grid"][0][0] == "water"
        assert body["grid"][0][2] == "urban"
        assert body["rows"] == 3


def test_bad_image_returns_400():
    with TestClient(app) as client:
        r = client.post("/predict", files={"file": ("x.png", b"not-an-image", "image/png")})
        assert r.status_code == 400
