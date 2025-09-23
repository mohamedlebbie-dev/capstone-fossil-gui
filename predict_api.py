# predict_api.py — minimal, GUI-safe mock that Ray can replace

from pathlib import Path
import hashlib
import random
from typing import Dict

# Model names must match GUI buttons
SUPPORTED_MODELS = {"VGG16", "ResNet18", "CustomCNN"}

# Placeholder class labels (replace with your final taxonomy)
FOSSIL_LABELS = [
    "Otodus obliquus",
    "Carcharhinus spp.",
    "Squalicorax kaupi",
    "Scapanorhynchus texanus",
    "Hemipristis serra",
    "Ophiomorpha nodosa",
    "Belemnitella americana",
]

def _mock_predict(image_path: Path, model_name: str) -> dict:
    """Deterministic per-filename so demos are stable."""
    seed = int(hashlib.sha256(image_path.name.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    label = rng.choice(FOSSIL_LABELS)
    conf = round(rng.uniform(0.70, 0.95), 3)
    return {"model": model_name, "top_class": label, "confidence": float(conf)}

def predict(image_path: str, model_name: str) -> dict:
    """
    Required interface:
        predict(image_path, model_name) -> {"model": str, "top_class": str, "confidence": float}
    """
    p = Path(image_path)
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}. Supported: {sorted(SUPPORTED_MODELS)}")
    # Swap this routing to call Ray's real models later.
    return _mock_predict(p, model_name)
