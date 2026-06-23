"""
Utility functions for the RCEdit-500K reference completion pipeline.

This module provides:
- OpenAI API integration (GPT-4o for VLM analysis, GPT-Image-1.5 for style/alter editing)
- Flux-Klein-9B model loading/inference for generation, personalization, and editing
- Grounded-SAM-2 segmentation (SAM2 + GroundingDINO)
- Image processing and type-specific reference construction
"""

import json
import os
import re
import io
import time
import base64
import gc
import random
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from typing import Optional, Dict, List, Any
import torch
import cv2
import pycocotools.mask as mask_util
from openai import OpenAI
from diffusers import Flux2KleinPipeline
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torchvision.ops import box_convert
from grounding_dino.groundingdino.util.inference import load_model, predict
import grounding_dino.groundingdino.datasets.transforms as T


# ============================================================================
# Configuration Constants
# ============================================================================

# Segmentation model configuration
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# Flux-Klein-9B model configuration
FLUX_KLEIN_MODEL_NAME = "black-forest-labs/FLUX.2-klein-9B"

# GPT Image model configuration
GPT_IMAGE_MODEL = "gpt-image-1.5"

# Global state
_openai_client = None
_gpt_image_client = None

# Global settings for model behavior
_model_dtype = None       # None = bfloat16 (default), "float16" for V100
_unload_model = True      # True = unload after use, False = keep cached
_model_seed = 42          # int or "random"
_gpu_id = None            # None = default GPU

# Model cache (used when _unload_model = False)
_cached_models = {
    "flux_klein_pipeline": None,
    "sam2_predictor": None,
    "grounding_model": None,
}


# ============================================================================
# OpenAI API Functions
# ============================================================================

def get_openai_client() -> OpenAI:
    """Get or create OpenAI client using OPENAI_API_KEY environment variable."""
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        _openai_client = OpenAI(api_key=api_key, max_retries=5)
    return _openai_client


def get_gpt_image_client() -> OpenAI:
    """Get or create OpenAI client for GPT Image API (same client, separate reference)."""
    global _gpt_image_client
    if _gpt_image_client is None:
        _gpt_image_client = get_openai_client()
    return _gpt_image_client


def call_gpt_with_backoff(client, messages, model_name="gpt-4o",
                          max_attempts=10, initial_sleep=2.0, temperature=0.3):
    """Call OpenAI API with exponential backoff on rate-limit errors."""
    attempt = 0
    sleep_time = initial_sleep
    last_exception = None

    while attempt < max_attempts:
        attempt += 1
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=4000,
                top_p=0.95,
                timeout=180,
            )
            return resp
        except Exception as e:
            last_exception = e
            s = str(e).lower()
            if ("rate" in s and "limit" in s) or "ratelimit" in s or "too many" in s or "quota" in s:
                if attempt < max_attempts:
                    print(f"Rate limit hit, retry {attempt}/{max_attempts} after {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                    sleep_time *= 2
                    continue
            raise

    if last_exception:
        raise last_exception
    raise RuntimeError("Unknown call_gpt error")


def encode_image_from_pil(img: Image.Image, size_max=1024) -> str:
    """Encode PIL Image to base64 JPEG string, resizing if needed."""
    img = img.convert("RGB")
    w, h = img.size
    if w > size_max or h > size_max:
        scale = size_max / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ============================================================================
# GPT Image API Functions
# ============================================================================

def pil_to_file_like(img: Image.Image, format: str = "PNG") -> io.BytesIO:
    """Convert PIL Image to file-like object for GPT Image API."""
    img = img.convert("RGBA") if format.upper() == "PNG" else img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    buf.name = f"image.{format.lower()}"
    return buf


def generate_image_with_gpt_api(
    images: list,
    prompt: str,
    model_name: str = GPT_IMAGE_MODEL,
    max_attempts: int = 5,
    initial_sleep: float = 2.0,
) -> Optional[Image.Image]:
    """Generate/edit image using GPT Image API with retry logic.

    Args:
        images: List of PIL Images to use as input
        prompt: Text prompt for generation/editing
        model_name: GPT image model name (default: gpt-image-1.5)
        max_attempts: Maximum retry attempts
        initial_sleep: Initial sleep time for exponential backoff

    Returns:
        Generated PIL Image or None on failure
    """
    client = get_gpt_image_client()

    image_files = []
    for img in images:
        if isinstance(img, Image.Image):
            image_files.append(pil_to_file_like(img, "PNG"))
        elif isinstance(img, str):
            image_files.append(pil_to_file_like(Image.open(img), "PNG"))
        else:
            image_files.append(img)

    attempt = 0
    sleep_time = initial_sleep
    last_exception = None

    while attempt < max_attempts:
        attempt += 1
        try:
            print(f"  Calling GPT Image API ({model_name})...")
            start_time = time.time()

            result = client.images.edit(
                model=model_name,
                image=image_files,
                prompt=prompt,
                input_fidelity="high",
                quality="high",
            )

            elapsed = time.time() - start_time
            print(f"  GPT Image API completed in {elapsed:.1f}s")

            image_base64 = result.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)
            return Image.open(io.BytesIO(image_bytes))

        except Exception as e:
            last_exception = e
            s = str(e).lower()
            if attempt < max_attempts:
                print(f"  GPT Image API error (attempt {attempt}/{max_attempts}): {e}")
                time.sleep(sleep_time)
                sleep_time *= 2
                for f in image_files:
                    if hasattr(f, 'seek'):
                        f.seek(0)
                continue
            break

    print(f"  GPT Image API failed after {max_attempts} attempts: {last_exception}")
    return None


# ============================================================================
# JSON Parsing
# ============================================================================

def extract_first_json(text: str) -> Optional[Dict]:
    """Extract first JSON object from text."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# ============================================================================
# Global Settings
# ============================================================================

def set_model_dtype(dtype_str: Optional[str]):
    """Set model dtype: None/'bfloat16' for default, 'float16' for V100."""
    global _model_dtype
    if dtype_str == "bfloat16":
        dtype_str = None
    _model_dtype = dtype_str
    print(f"  Model dtype: {dtype_str if dtype_str else 'bfloat16 (default)'}")


def get_model_dtype() -> torch.dtype:
    """Get current model dtype as torch.dtype."""
    return torch.float16 if _model_dtype == "float16" else torch.bfloat16


def get_model_dtype_str() -> Optional[str]:
    """Get current model dtype setting as string."""
    return _model_dtype


def set_unload_model(unload: bool):
    """Set whether to unload models after each use to free VRAM."""
    global _unload_model
    _unload_model = unload
    print(f"  Unload model: {unload}")


def get_unload_model() -> bool:
    return _unload_model


def set_model_seed(seed):
    """Set random seed for model inference. Use int or 'random'."""
    global _model_seed
    _model_seed = seed
    print(f"  Model seed: {seed}")


def get_model_seed():
    return _model_seed


def get_model_seed_value() -> int:
    """Get actual seed value (generates random if configured as 'random')."""
    if _model_seed == "random":
        return random.randint(0, 2**32 - 1)
    return _model_seed


def create_torch_generator(device: str = None) -> torch.Generator:
    """Create a torch Generator with the configured seed."""
    if device is None:
        device = get_device()
    seed = get_model_seed_value()
    return torch.Generator(device=device).manual_seed(seed)


def set_gpu_id(gpu_id: int = None):
    """Set GPU ID for model loading."""
    global _gpu_id
    _gpu_id = gpu_id


def get_gpu_id() -> int:
    return _gpu_id


def get_device() -> str:
    """Get device string (e.g., 'cuda:0')."""
    if _gpu_id is not None:
        return f"cuda:{_gpu_id}"
    return "cuda"


def get_device_for_cpu_offload() -> int:
    """Get GPU ID for enable_model_cpu_offload."""
    return _gpu_id if _gpu_id is not None else 0


def clear_model_cache():
    """Clear all cached models and free GPU memory."""
    global _cached_models
    for key in list(_cached_models.keys()):
        if _cached_models[key] is not None:
            del _cached_models[key]
            _cached_models[key] = None
    cleanup_gpu_memory()


# ============================================================================
# GPU Memory Management
# ============================================================================

def cleanup_gpu_memory():
    """Clean up GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ============================================================================
# Model Loading Functions
# ============================================================================

def load_sam2_model():
    """Load SAM2 model (with optional caching)."""
    global _cached_models
    if not get_unload_model() and _cached_models["sam2_predictor"] is not None:
        return _cached_models["sam2_predictor"]

    device = get_device() if torch.cuda.is_available() else "cpu"
    print(f"  Loading SAM2 model (device={device})...")
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    print("  SAM2 model loaded")

    if not get_unload_model():
        _cached_models["sam2_predictor"] = sam2_predictor
    return sam2_predictor


def load_grounding_dino_model():
    """Load Grounding DINO model (with optional caching)."""
    global _cached_models
    if not get_unload_model() and _cached_models["grounding_model"] is not None:
        return _cached_models["grounding_model"]

    device = get_device() if torch.cuda.is_available() else "cpu"
    print(f"  Loading Grounding DINO model (device={device})...")
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=device
    )
    print("  Grounding DINO model loaded")

    if not get_unload_model():
        _cached_models["grounding_model"] = grounding_model
    return grounding_model


def unload_segmentation_models(sam2_predictor, grounding_model):
    """Unload segmentation models (respects cache setting)."""
    if not get_unload_model():
        return
    if sam2_predictor is not None:
        del sam2_predictor
    if grounding_model is not None:
        del grounding_model
    cleanup_gpu_memory()


def load_flux_klein_pipeline() -> Flux2KleinPipeline:
    """Load FLUX.2-klein-9B pipeline (with optional caching)."""
    global _cached_models
    if not get_unload_model() and _cached_models["flux_klein_pipeline"] is not None:
        return _cached_models["flux_klein_pipeline"]

    dtype = get_model_dtype()
    gpu_id = get_device_for_cpu_offload()
    print(f"  Loading FLUX.2-klein-9B pipeline (dtype={dtype}, gpu_id={gpu_id})...")

    pipeline = Flux2KleinPipeline.from_pretrained(
        FLUX_KLEIN_MODEL_NAME,
        torch_dtype=dtype
    )
    pipeline.enable_model_cpu_offload(gpu_id=gpu_id)
    print("  FLUX.2-klein-9B pipeline loaded")

    if not get_unload_model():
        _cached_models["flux_klein_pipeline"] = pipeline
    return pipeline


def unload_flux_klein_pipeline(pipeline):
    """Unload FLUX.2-klein pipeline (respects cache setting)."""
    if not get_unload_model():
        return
    if pipeline is not None:
        del pipeline
    cleanup_gpu_memory()


# ============================================================================
# Image Processing Utilities
# ============================================================================

def single_mask_to_rle(mask):
    """Convert mask to RLE format."""
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def load_image_for_grounding(image_input: Image.Image):
    """Load and transform image for Grounding DINO."""
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_source = np.asarray(image_input)
    image_transformed, _ = transform(image_input, None)
    return image_source, image_transformed


def preprocess_image_pair(input_image: Image.Image, output_image: Image.Image,
                          target_size: int = 1024, verbose: bool = False):
    """Resize image pair to ~target_size while maintaining aspect ratio (divisible by 16)."""
    w, h = input_image.size
    if w > h:
        new_w = target_size
        new_h = int(h * target_size / w)
    else:
        new_h = target_size
        new_w = int(w * target_size / h)

    new_w = max(16, (new_w // 16) * 16)
    new_h = max(16, (new_h // 16) * 16)

    resized_input = input_image.resize((new_w, new_h), Image.LANCZOS)
    resized_output = output_image.resize((new_w, new_h), Image.LANCZOS)
    return resized_input, resized_output


def save_image(image: Image.Image, output_path: str):
    """Save PIL Image to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)


# ============================================================================
# GPT Model Interface (Single-Call with gpt_prompt_forward.txt)
# ============================================================================

def _extract_gpt_response_text(response) -> str:
    """Extract text content from GPT response."""
    raw = response.choices[0].message.content
    if isinstance(raw, list):
        return "\n".join(item.text for item in raw if hasattr(item, 'text'))
    return raw


def load_forward_prompt(prompt_file: str = "prompts/gpt_prompt_forward.txt") -> str:
    """Load the unified forward prompt template."""
    script_dir = Path(__file__).parent.parent
    prompt_path = script_dir / prompt_file
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def call_gpt_model(input_image: Image.Image, output_image: Image.Image,
                   prompt: str, model_name: str = "gpt-4o",
                   **kwargs) -> Dict[str, Any]:
    """Call GPT-4o with the unified single-call prompt (gpt_prompt_forward.txt).

    This performs all three steps in one call:
    1. Type & Assertion
    2. Reference Construction
    3. Final Instruction (new_instruction, new_instruction_weak)

    Args:
        input_image: The input/source image (ORIGINAL_IMAGE)
        output_image: The output/target image (TARGET_IMAGE)
        prompt: The raw editing prompt (RAW_PROMPT)
        model_name: GPT model name (default: gpt-4o)

    Returns:
        Dictionary with: assertion, type, and type-specific reference keys
    """
    print(f"  [GPT] Single-call analysis with {model_name}...")

    client = get_openai_client()

    # Load and format prompt
    template = load_forward_prompt()
    system_prompt = template.replace("{prompt}", prompt)

    # Encode images
    input_b64 = encode_image_from_pil(input_image)
    output_b64 = encode_image_from_pil(output_image)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{input_b64}", "detail": "high"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{output_b64}", "detail": "high"}},
            ],
        }
    ]

    response = call_gpt_with_backoff(client, messages, model_name=model_name)
    raw_text = _extract_gpt_response_text(response)

    parsed = extract_first_json(raw_text)
    if parsed is None:
        raise ValueError(f"Failed to parse JSON from GPT response: {raw_text[:500]}")

    print(f"  [GPT] Type: {parsed.get('type')}, Assertion: {parsed.get('assertion')}")
    return parsed


# ============================================================================
# Diffusion Model Interfaces (Flux-Klein-9B only)
# ============================================================================

def generation_model(prompt: str, width: int = 512, height: int = 512,
                    debug: bool = False, save_path: str = None) -> Image.Image:
    """Generate a new image from text prompt using FLUX.2-klein-9B."""
    print(f"  [Generation] Prompt: {prompt[:80]}...")

    pipeline = load_flux_klein_pipeline()
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.inference_mode():
            result_image = pipeline(
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=1.0,
                num_inference_steps=4,
                generator=create_torch_generator(device)
            ).images[0]

        if debug and save_path:
            save_image(result_image, save_path)
        return result_image
    finally:
        unload_flux_klein_pipeline(pipeline)


def personalization_model(prompt: str, image: Image.Image,
                         width: int = 512, height: int = 512,
                         debug: bool = False, save_path: str = None) -> Image.Image:
    """Personalization: maintain object identity in a different scene using FLUX.2-klein-9B."""
    print(f"  [Personalization] Prompt: {prompt[:80]}...")

    pipeline = load_flux_klein_pipeline()
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.inference_mode():
            result_image = pipeline(
                image=[image],
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=1.0,
                num_inference_steps=4,
                generator=create_torch_generator(device)
            ).images[0]

        if debug and save_path:
            save_image(result_image, save_path)
        return result_image
    finally:
        unload_flux_klein_pipeline(pipeline)


def edit_model(prompt: str, image: Image.Image, condition_images: Optional[List[Image.Image]] = None,
              width: int = 512, height: int = 512,
              debug: bool = False, save_path: str = None) -> Image.Image:
    """Edit an image based on prompt using FLUX.2-klein-9B.

    For single-image editing (no condition_images), passes [image].
    For multi-image editing (with condition_images), passes [image] + condition_images.
    """
    print(f"  [Edit] Prompt: {prompt[:80]}...")

    pipeline = load_flux_klein_pipeline()
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image_list = [image]
        if condition_images:
            image_list.extend(condition_images)

        with torch.inference_mode():
            result_image = pipeline(
                image=image_list,
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=1.0,
                num_inference_steps=4,
                generator=create_torch_generator(device)
            ).images[0]

        if debug and save_path:
            save_image(result_image, save_path)
        return result_image
    finally:
        unload_flux_klein_pipeline(pipeline)


def segmentation_model(keyword: str, image: Image.Image,
                      debug: bool = False, save_path: str = None) -> Dict[str, Any]:
    """Segment objects in image using Grounded-SAM-2 (GroundingDINO + SAM2).

    Returns:
        Dictionary with 'keyword', 'annotations' list, and 'box_format'
    """
    print(f"  [Segmentation] Keyword: {keyword}")

    sam2_predictor = load_sam2_model()
    grounding_model = load_grounding_dino_model()

    try:
        image_source, image_transformed = load_image_for_grounding(image)

        boxes, logits, phrases = predict(
            model=grounding_model,
            image=image_transformed,
            caption=keyword,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )

        h, w, _ = image_source.shape
        boxes_xyxy = box_convert(boxes=boxes * torch.Tensor([w, h, w, h]),
                                 in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()

        sam2_predictor.set_image(image_source)

        annotations = []
        for box_xyxy, logit, phrase in zip(boxes_xyxy, logits, phrases):
            masks, scores, _ = sam2_predictor.predict(
                point_coords=None, point_labels=None,
                box=box_xyxy, multimask_output=False,
            )
            mask = masks[0]
            mask_rle = single_mask_to_rle(mask)
            annotations.append({
                "class_name": phrase,
                "bbox": box_xyxy.tolist(),
                "segmentation": mask_rle,
                "score": float(logit)
            })

        annotations.sort(key=lambda x: x['score'], reverse=True)
        print(f"  [Segmentation] Found {len(annotations)} object(s)")

        return {"keyword": keyword, "annotations": annotations, "box_format": "xyxy"}
    finally:
        unload_segmentation_models(sam2_predictor, grounding_model)


# ============================================================================
# BBox Augmentation Utilities
# ============================================================================

def augment_bbox_for_draw(bbox: List[float], image_size: tuple) -> List[int]:
    """Augment bbox for drawing (marked reference) with random perturbations."""
    img_w, img_h = image_size
    x1, y1, x2, y2 = bbox
    orig_w, orig_h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    new_w, new_h, new_cx, new_cy = orig_w, orig_h, cx, cy

    if random.random() < 0.1:  # Scale augmentation
        scale = random.uniform(0.8, 1.2)
        new_w *= scale
        new_h *= scale
    if random.random() < 0.1:  # Shift augmentation
        new_cx += random.uniform(-orig_w * 0.25, orig_w * 0.25)
        new_cy += random.uniform(-orig_h * 0.25, orig_h * 0.25)
    if random.random() < 0.1:  # Crop augmentation
        crop = random.uniform(0.5, 1.0)
        new_w *= crop
        new_h *= crop

    return [
        int(max(0, new_cx - new_w / 2)),
        int(max(0, new_cy - new_h / 2)),
        int(min(img_w, new_cx + new_w / 2)),
        int(min(img_h, new_cy + new_h / 2)),
    ]


def augment_bbox_for_crop(bbox: List[float], image_size: tuple) -> List[int]:
    """Augment bbox for cropping (cropped reference). Ensures min 256x256."""
    img_w, img_h = image_size
    x1, y1, x2, y2 = bbox
    orig_w, orig_h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    # Ensure minimum 256x256
    if orig_w < 256 or orig_h < 256:
        scale = max(256 / orig_w, 256 / orig_h)
        new_w, new_h = orig_w * scale, orig_h * scale
    else:
        new_w, new_h = orig_w, orig_h

    new_cx, new_cy = cx, cy

    if random.random() < 0.1:
        s = random.uniform(0.8, 1.2)
        new_w *= s
        new_h *= s
    if random.random() < 0.1:
        new_cx += random.uniform(-orig_w * 0.25, orig_w * 0.25)
        new_cy += random.uniform(-orig_h * 0.25, orig_h * 0.25)
    if random.random() < 0.1:
        c = random.uniform(0.5, 1.0)
        new_w *= c
        new_h *= c

    return [
        int(max(0, new_cx - new_w / 2)),
        int(max(0, new_cy - new_h / 2)),
        int(min(img_w, new_cx + new_w / 2)),
        int(min(img_h, new_cy + new_h / 2)),
    ]


# ============================================================================
# Object Extraction Utilities
# ============================================================================

def extract_segmented_object(keyword: str, image: Image.Image,
                            debug: bool = False) -> Image.Image:
    """Extract object from image and place on gray background."""
    seg_result = segmentation_model(keyword, image, debug=debug)
    if not seg_result['annotations']:
        raise ValueError(f"No object found for keyword: {keyword}")

    mask_rle = seg_result['annotations'][0]['segmentation']
    mask = mask_util.decode(mask_rle)
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    img_array = np.array(image)
    gray_bg = np.ones_like(img_array) * 128
    result_array = np.where(mask.astype(bool)[:, :, None], img_array, gray_bg)
    return Image.fromarray(result_array.astype(np.uint8))


def create_cropped_reference(keyword: str, image: Image.Image) -> Image.Image:
    """Create cropped reference by extracting object bbox with augmentation."""
    seg_result = segmentation_model(keyword, image)
    if not seg_result['annotations']:
        raise ValueError(f"No object found for keyword: {keyword}")

    bbox = seg_result['annotations'][0]['bbox']
    augmented = augment_bbox_for_crop(bbox, image.size)
    return image.crop(tuple(augmented))


def create_marked_reference(keyword: str, image: Image.Image) -> Image.Image:
    """Create marked reference by drawing red bbox with augmentation."""
    seg_result = segmentation_model(keyword, image)
    if not seg_result['annotations']:
        raise ValueError(f"No object found for keyword: {keyword}")

    bbox = seg_result['annotations'][0]['bbox']
    augmented = augment_bbox_for_draw(bbox, image.size)
    x1, y1, x2, y2 = augmented

    marked_image = image.copy()
    draw = ImageDraw.Draw(marked_image)
    for i in range(5):
        draw.rectangle([x1+i, y1+i, x2-i, y2-i], outline="red", width=1)
    return marked_image


# ============================================================================
# Type-Specific Processing Functions
# ============================================================================

def process_add_type(gpt_output: Dict[str, Any], input_image: Image.Image,
                     output_image: Image.Image, index: int, image_id: str = "",
                     debug: bool = False, images_dir: Path = None) -> List[tuple]:
    """Process 'add' type: segment from target, then personalize and/or edit."""
    segment_keyword = gpt_output.get("segment", "")
    if not segment_keyword:
        raise ValueError("segment keyword is required for 'add' type")

    segmented_object = extract_segmented_object(segment_keyword, output_image, debug=debug)
    reference_images = []
    output_w, output_h = output_image.size

    personalization_prompt = gpt_output.get("personalization", "")
    if personalization_prompt:
        ref1 = personalization_model(personalization_prompt, segmented_object,
                                    width=output_w, height=output_h, debug=debug)
        reference_images.append(("personalization", ref1))

    edit_prompt = gpt_output.get("edit", "")
    if edit_prompt:
        ref2 = edit_model(edit_prompt, segmented_object,
                         width=output_w, height=output_h, debug=debug)
        reference_images.append(("edit", ref2))

    if not reference_images:
        raise ValueError("No reference images generated for 'add' type")
    return reference_images


def process_replace_type(gpt_output: Dict[str, Any], input_image: Image.Image,
                         output_image: Image.Image, index: int, image_id: str = "",
                         debug: bool = False, images_dir: Path = None) -> List[tuple]:
    """Process 'replace' type (same as add)."""
    return process_add_type(gpt_output, input_image, output_image, index, image_id, debug, images_dir)


def process_remove_type(gpt_output: Dict[str, Any], input_image: Image.Image,
                        output_image: Image.Image, index: int, image_id: str = "",
                        debug: bool = False, images_dir: Path = None) -> List[tuple]:
    """Process 'remove' type: create cropped and marked references from input image."""
    reference_images = []
    segment_keyword = gpt_output.get("segment", "")
    if segment_keyword:
        reference_images.append(("cropped", create_cropped_reference(segment_keyword, input_image)))
        reference_images.append(("marked", create_marked_reference(segment_keyword, input_image)))
    return reference_images


def process_background_type(gpt_output: Dict[str, Any], input_image: Image.Image,
                            output_image: Image.Image, index: int, image_id: str = "",
                            debug: bool = False, images_dir: Path = None) -> List[tuple]:
    """Process 'background' type: remove foreground from target to get clean background."""
    edit_prompt = gpt_output.get("edit", "")
    if not edit_prompt:
        return []
    output_w, output_h = output_image.size
    bg_ref = edit_model(edit_prompt, output_image, width=output_w, height=output_h, debug=debug)
    return [("background", bg_ref)]


def process_style_type(gpt_output: Dict[str, Any], input_image: Image.Image,
                       output_image: Image.Image, index: int, image_id: str = "",
                       debug: bool = False, images_dir: Path = None) -> List[tuple]:
    """Process 'style' type: generate source image, then transfer style via GPT Image API."""
    generation_prompt = gpt_output.get("generation", "")
    if not generation_prompt:
        raise ValueError("generation prompt is required for 'style' type")

    edit_prompt = gpt_output.get("edit", "")
    if not edit_prompt:
        raise ValueError("edit prompt is required for 'style' type")

    output_w, output_h = output_image.size
    style_src = generation_model(generation_prompt, width=output_w, height=output_h, debug=debug)

    # Use GPT Image API for style transfer
    gpt_edit_prompt = edit_prompt.replace("__STYLE_SRC_IMAGE__", "image 1")
    gpt_edit_prompt = gpt_edit_prompt.replace("__STYLE_COND_IMAGE__", "image 2")

    styled = generate_image_with_gpt_api(
        images=[style_src, output_image],
        prompt=gpt_edit_prompt,
    )
    if styled is None:
        raise RuntimeError("GPT Image API failed for style transfer")
    return [("style", styled)]


def process_alter_type(gpt_output: Dict[str, Any], input_image: Image.Image,
                       output_image: Image.Image, index: int, image_id: str = "",
                       debug: bool = False, images_dir: Path = None) -> List[tuple]:
    """Process 'alter' type: generate source image, then transfer attribute via GPT Image API."""
    generation_prompt = gpt_output.get("generation", "")
    if not generation_prompt:
        raise ValueError("generation prompt is required for 'alter' type")

    edit_prompt = gpt_output.get("edit", "")
    if not edit_prompt:
        raise ValueError("edit prompt is required for 'alter' type")

    output_w, output_h = output_image.size
    alter_src = generation_model(generation_prompt, width=output_w, height=output_h, debug=debug)

    gpt_edit_prompt = edit_prompt.replace("__ALTER_SRC_IMAGE__", "image 1")
    gpt_edit_prompt = gpt_edit_prompt.replace("__ALTER_COND_IMAGE__", "image 2")

    altered = generate_image_with_gpt_api(
        images=[alter_src, output_image],
        prompt=gpt_edit_prompt,
    )
    if altered is None:
        raise RuntimeError("GPT Image API failed for attribute transfer")
    return [("alter", altered)]


# Type dispatcher mapping
TYPE_PROCESSORS = {
    "add": process_add_type,
    "replace": process_replace_type,
    "remove": process_remove_type,
    "background": process_background_type,
    "style": process_style_type,
    "alter": process_alter_type,
}
