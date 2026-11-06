"""FastAPI service that classifies satellite images into 10 land-use classes.

Run:
    uvicorn app.main:app --reload
Then open http://127.0.0.1:8000 for the upload page, or http://127.0.0.1:8000/docs for the API docs.
"""

import io
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.request import urlretrieve

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data import CLASSES, normalize  # noqa: E402
from models import load_trained  # noqa: E402

MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "models" / "resnet18_pretrained.pt"))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = ROOT / MODEL_PATH
# Trained weights are attached to the GitHub release under the same file name.
RELEASE_URL = "https://github.com/eslammohibrahim/satellite-land-use-classifier/releases/download/v1.0/"
IMAGE_SIZE = 64  # EuroSAT images are 64x64; other sizes are resized to match
MAX_BYTES = 5 * 1024 * 1024

state = {}


def get_model():
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Model not found, downloading {MODEL_PATH.name} from the v1.0 release ...")
        urlretrieve(RELEASE_URL + MODEL_PATH.name, MODEL_PATH)
    return load_trained("resnet18_pretrained", MODEL_PATH)


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["model"] = get_model()
    yield
    state.clear()


app = FastAPI(
    title="Satellite Land-Use Classifier",
    description="Classifies a satellite image into one of 10 land-use classes (EuroSAT).",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


class Prediction(BaseModel):
    label: str
    probability: float


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    top3: list[Prediction]


def preprocess(raw: bytes) -> torch.Tensor:
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The file is not a readable image.")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    x = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return x.view(IMAGE_SIZE, IMAGE_SIZE, 3).permute(2, 0, 1).unsqueeze(0)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/classes")
def classes():
    return {"classes": CLASSES}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 5 MB.")
    x = preprocess(raw)
    with torch.no_grad():
        probs = torch.softmax(state["model"](normalize(x)), dim=1)[0]
    top = torch.topk(probs, 3)
    top3 = [Prediction(label=CLASSES[i], probability=round(p, 4))
            for p, i in zip(top.values.tolist(), top.indices.tolist())]
    return PredictResponse(prediction=top3[0].label, confidence=top3[0].probability, top3=top3)
