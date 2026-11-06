import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.main import app  # noqa: E402
from data import CLASSES, load_split  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_classes(client):
    assert client.get("/classes").json()["classes"] == CLASSES


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Satellite Land-Use Classifier" in r.text


def test_predict_returns_valid_top3(client):
    x, _ = load_split("test")
    image = Image.fromarray(x[0].permute(1, 2, 0).numpy())
    r = client.post("/predict", files={"file": ("tile.png", png_bytes(image), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in CLASSES
    assert len(body["top3"]) == 3
    probs = [p["probability"] for p in body["top3"]]
    assert probs == sorted(probs, reverse=True)
    assert 0 < sum(probs) <= 1.0001


def test_predict_matches_dataset_labels(client):
    """The API should agree with the test labels on most real test images."""
    x, y = load_split("test")
    correct = 0
    for i in range(50):
        image = Image.fromarray(x[i].permute(1, 2, 0).numpy())
        r = client.post("/predict", files={"file": ("tile.png", png_bytes(image), "image/png")})
        correct += r.json()["prediction"] == CLASSES[int(y[i])]
    assert correct >= 45


def test_resizes_other_image_sizes(client):
    image = Image.new("RGB", (300, 200), (40, 120, 60))
    r = client.post("/predict", files={"file": ("big.png", png_bytes(image), "image/png")})
    assert r.status_code == 200


def test_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_rejects_large_file(client):
    r = client.post("/predict", files={"file": ("big.bin", b"0" * (6 * 1024 * 1024), "image/png")})
    assert r.status_code == 413
