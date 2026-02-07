"""
SigLIP (google/siglip-base-patch16-384) image and text embeddings, 768-dim.
"""
import io
import requests
from typing import List
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel
from config import SIGLIP_MODEL_ID, EMBEDDING_DIM


_model = None
_processor = None
_device = None


def _get_device():
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def _load_model():
    global _model, _processor
    if _model is None:
        _processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_ID)
        _model = AutoModel.from_pretrained(SIGLIP_MODEL_ID)
        _model.to(_get_device())
        _model.eval()
    return _model, _processor


def image_from_url(url: str) -> Image.Image:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def get_image_embedding(image: Image.Image | str) -> List[float]:
    """Return 768-dim embedding for one image (PIL or URL)."""
    model, processor = _load_model()
    device = _get_device()
    if isinstance(image, str):
        image = image_from_url(image)
    inputs = processor(images=image, text=[""], padding="max_length", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        if hasattr(model, "get_image_features"):
            image_embeds = model.get_image_features(pixel_values=inputs["pixel_values"])
        else:
            # Fallback: pass dummy text and take image_embeds from full forward
            inputs["input_ids"] = processor.tokenizer([""], padding="max_length", return_tensors="pt")["input_ids"].to(device)
            inputs["attention_mask"] = (inputs["input_ids"] != processor.tokenizer.pad_token_id).long().to(device)
            out = model(**inputs)
            image_embeds = out.image_embeds
    vec = image_embeds[0].float().cpu().numpy()
    assert len(vec) == EMBEDDING_DIM, f"expected {EMBEDDING_DIM}, got {len(vec)}"
    return vec.tolist()


def get_text_embedding(text: str) -> List[float]:
    """Return 768-dim embedding for text (same model text encoder)."""
    if not (text or text.strip()):
        text = " "
    model, processor = _load_model()
    device = _get_device()
    inputs = processor(text=[text], padding="max_length", return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        if hasattr(model, "get_text_features"):
            text_embeds = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
        else:
            out = model(**{k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask", "position_ids")})
            text_embeds = out.text_embeds
    vec = text_embeds[0].float().cpu().numpy()
    assert len(vec) == EMBEDDING_DIM, f"expected {EMBEDDING_DIM}, got {len(vec)}"
    return vec.tolist()
