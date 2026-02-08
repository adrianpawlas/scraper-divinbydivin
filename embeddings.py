"""
SigLIP (google/siglip-base-patch16-384) image and text embeddings, 768-dim.
Uses explicit SiglipImageProcessor and SiglipTokenizer to avoid AutoProcessor
tokenizer-mapping bugs in some transformers versions (e.g. on GitHub Actions).
"""
import io
import requests
from typing import List
from PIL import Image
import torch
from transformers import (
    AutoModel,
    SiglipImageProcessor,
    SiglipTokenizer,
)
from config import SIGLIP_MODEL_ID, EMBEDDING_DIM


_model = None
_image_processor = None
_tokenizer = None
_device = None


def _get_device():
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def _load_model():
    global _model, _image_processor, _tokenizer
    if _model is None:
        _image_processor = SiglipImageProcessor.from_pretrained(SIGLIP_MODEL_ID)
        _tokenizer = SiglipTokenizer.from_pretrained(SIGLIP_MODEL_ID)
        _model = AutoModel.from_pretrained(SIGLIP_MODEL_ID)
        _model.to(_get_device())
        _model.eval()
    return _model, _image_processor, _tokenizer


def image_from_url(url: str) -> Image.Image:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def get_image_embedding(image: Image.Image | str) -> List[float]:
    """Return 768-dim embedding for one image (PIL or URL)."""
    model, image_processor, tokenizer = _load_model()
    device = _get_device()
    if isinstance(image, str):
        image = image_from_url(image)
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        # Use vision_model directly so we always get pooler_output (batch, 768).
        # get_image_features() can return ModelOutput where [0] is last_hidden_state.
        vision_out = model.vision_model(pixel_values)
        image_embeds = vision_out.pooler_output  # (batch, 768)
    vec = image_embeds[0].float().cpu().numpy()
    if vec.ndim > 1:
        vec = vec.flatten()
    assert len(vec) == EMBEDDING_DIM, f"expected {EMBEDDING_DIM}, got {len(vec)}"
    return vec.tolist()


def get_text_embedding(text: str) -> List[float]:
    """Return 768-dim embedding for text (same model text encoder)."""
    if not (text or text.strip()):
        text = " "
    model, _image_processor, tokenizer = _load_model()
    device = _get_device()
    encoded = tokenizer(
        [text],
        padding="max_length",
        return_tensors="pt",
        truncation=True,
        return_attention_mask=True,
    )
    input_ids = encoded.input_ids.to(device)
    attention_mask = (
        encoded.attention_mask.to(device)
        if "attention_mask" in encoded
        else (input_ids != tokenizer.pad_token_id).long().to(device)
    )
    with torch.no_grad():
        text_embeds = model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        if hasattr(text_embeds, "pooler_output") and text_embeds.pooler_output is not None:
            text_embeds = text_embeds.pooler_output
    vec = text_embeds[0].float().cpu().numpy()
    if vec.ndim > 1:
        vec = vec.flatten()
    assert len(vec) == EMBEDDING_DIM, f"expected {EMBEDDING_DIM}, got {len(vec)}"
    return vec.tolist()
