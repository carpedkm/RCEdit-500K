"""
RCEdit-500K Pipeline - Part 2b: GPT Image Pool Processing

Processes style/alter entries using GPT-Image-1.5 API for style/attribute transfer.
Uses concurrent workers for maximum throughput with rate limit handling.

Usage:
    python multi_reference_part2b_gpt_pool.py \
        --input_csv outputs/part2a/style_alter_temp.csv \
        --data_dir /path/to/images \
        --output_dir outputs/part2b/references \
        --style_alter_temp_dir outputs/part2a/style_alter_temp \
        --csv_output_dir outputs/part2b
"""

import argparse
import csv
import os
import time
import base64
import io
import traceback
import threading
import multiprocessing as mp
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm
from dataclasses import dataclass, field
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from openai import OpenAI


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PipelineConfig:
    image_max_size: int = 1024
    num_workers: int = 12       # concurrent API requests
    rate_limit_pause: float = 60.0
    gpt_image_model: str = "gpt-image-1.5"
    debug: bool = False


# ============================================================================
# CSV Definition
# ============================================================================

INPUT_FIELDNAMES = [
    "id", "dataset", "input_path", "target_path", "prompt",
    "type", "assertion", "reason",
    "segment", "generation", "personalization", "edit",
    "new_instruction", "new_instruction_weak",
    "error", "generation_path",
]

OUTPUT_FIELDNAMES = [
    "id", "dataset", "input_path", "target_path", "prompt",
    "type", "assertion", "reason",
    "segment", "generation", "personalization", "edit",
    "new_instruction", "new_instruction_weak",
    "error", "reference_path",
]


# ============================================================================
# Image Utilities
# ============================================================================

def pil_to_file_like(img: Image.Image, format: str = "PNG") -> io.BytesIO:
    img = img.convert("RGBA") if format.upper() == "PNG" else img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    buf.name = f"image.{format.lower()}"
    return buf


def load_and_preprocess_image(image_path: Path, max_size: int = 1024) -> Image.Image:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w = max(16, (int(w * scale) // 16) * 16)
        new_h = max(16, (int(h * scale) // 16) * 16)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


# ============================================================================
# Main Processing
# ============================================================================

def process_style_alter_gpt_pool(
    input_csv: str,
    data_dir: str,
    output_dir: str,
    style_alter_temp_dir: str,
    csv_output_dir: str,
    config: PipelineConfig = None,
    resume: bool = False,
):
    if config is None:
        config = PipelineConfig()

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    style_alter_temp_dir = Path(style_alter_temp_dir)
    csv_output_dir = Path(csv_output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    client = OpenAI(api_key=api_key, max_retries=2)

    print(f"GPT Image Pool Processing")
    print(f"  Model: {config.gpt_image_model}")
    print(f"  Workers: {config.num_workers}")

    # Load input CSV
    all_entries = pd.read_csv(input_csv).to_dict('records')

    # Filter to style/alter with generation_path
    entries = [e for e in all_entries
               if str(e.get("type", "")).lower() in ["style", "alter"]
               and e.get("generation_path", "")]

    print(f"Style/Alter entries: {len(entries)}")
    if not entries:
        return

    output_csv_path = csv_output_dir / "style_alter_output_gpt.csv"

    # Resume
    processed_ids = set()
    if resume and output_csv_path.exists():
        try:
            processed_ids = set(pd.read_csv(output_csv_path)['id'].tolist())
        except Exception:
            pass

    entries = [e for e in entries if e.get("id", "") not in processed_ids]
    print(f"To process: {len(entries)}")
    if not entries:
        return

    stats = {"success": 0, "errors": 0, "skipped": 0}
    stats_lock = threading.Lock()
    batch_buffer = []
    batch_lock = threading.Lock()

    def flush_batch():
        with batch_lock:
            if not batch_buffer:
                return
            write_header = not output_csv_path.exists() or output_csv_path.stat().st_size == 0
            with open(output_csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES)
                if write_header:
                    writer.writeheader()
                writer.writerows(batch_buffer)
                f.flush()
                os.fsync(f.fileno())
            batch_buffer.clear()

    def process_one(entry):
        entry_id = entry.get("id", "")
        edit_type = str(entry.get("type", "")).lower()

        out = {k: entry.get(k, "") for k in OUTPUT_FIELDNAMES if k != "reference_path"}
        out["reference_path"] = ""

        # Validate
        gen_path_rel = entry.get("generation_path", "")
        target_rel = entry.get("target_path", "")
        edit_prompt = entry.get("edit", "")

        if not gen_path_rel or not target_rel or not edit_prompt:
            out["error"] = "Missing required fields"
            with stats_lock:
                stats["skipped"] += 1
            return out

        gen_path = style_alter_temp_dir / gen_path_rel
        target_path = data_dir / target_rel

        if not gen_path.exists() or not target_path.exists():
            out["error"] = f"Image not found"
            with stats_lock:
                stats["errors"] += 1
            return out

        # Format prompt
        if edit_type == "style":
            prompt = edit_prompt.replace("__STYLE_SRC_IMAGE__", "image 1").replace("__STYLE_COND_IMAGE__", "image 2")
        else:
            prompt = edit_prompt.replace("__ALTER_SRC_IMAGE__", "image 1").replace("__ALTER_COND_IMAGE__", "image 2")

        try:
            gen_img = load_and_preprocess_image(gen_path, config.image_max_size)
            target_img = load_and_preprocess_image(target_path, config.image_max_size)

            gen_file = pil_to_file_like(gen_img)
            target_file = pil_to_file_like(target_img)

            # Retry with backoff
            max_attempts = 5
            sleep_time = 2.0
            for attempt in range(max_attempts):
                try:
                    result = client.images.edit(
                        model=config.gpt_image_model,
                        image=[gen_file, target_file],
                        prompt=prompt,
                        input_fidelity="high",
                        quality="high",
                    )
                    image_bytes = base64.b64decode(result.data[0].b64_json)

                    # Save
                    safe_id = entry_id.replace("/", "_").replace("\\", "_")
                    fname = f"reference_{safe_id}_{edit_type}.png"
                    ref_path = output_dir / fname
                    ref_path.parent.mkdir(parents=True, exist_ok=True)

                    with open(ref_path, 'wb') as f:
                        f.write(image_bytes)
                        f.flush()
                        os.fsync(f.fileno())

                    out["reference_path"] = fname
                    out["error"] = ""
                    with stats_lock:
                        stats["success"] += 1
                    return out

                except Exception as e:
                    s = str(e).lower()
                    if "rate" in s or "429" in s or "too many" in s:
                        time.sleep(sleep_time)
                        sleep_time *= 2
                        gen_file.seek(0)
                        target_file.seek(0)
                        continue
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(sleep_time)
                    sleep_time *= 2
                    gen_file.seek(0)
                    target_file.seek(0)

        except Exception as e:
            out["error"] = f"API error: {e}"
            with stats_lock:
                stats["errors"] += 1
        return out

    # Process with thread pool
    pbar = tqdm(total=len(entries), desc="Processing")

    with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
        futures = {executor.submit(process_one, e): e for e in entries}
        for future in futures:
            try:
                result = future.result(timeout=300)
                with batch_lock:
                    batch_buffer.append(result)
                    if len(batch_buffer) >= 36:
                        flush_batch()
            except Exception as e:
                entry = futures[future]
                out = {k: entry.get(k, "") for k in OUTPUT_FIELDNAMES if k != "reference_path"}
                out["reference_path"] = ""
                out["error"] = str(e)
                with batch_lock:
                    batch_buffer.append(out)
                with stats_lock:
                    stats["errors"] += 1
            pbar.update(1)
            with stats_lock:
                pbar.set_postfix({'ok': stats["success"], 'err': stats["errors"]})

    pbar.close()
    flush_batch()

    print(f"\nDone. Success: {stats['success']}, Errors: {stats['errors']}, Skipped: {stats['skipped']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RCEdit Part 2b: GPT Image Pool Processing")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory for images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for reference images")
    parser.add_argument("--style_alter_temp_dir", type=str, required=True)
    parser.add_argument("--csv_output_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--rate_limit_pause", type=float, default=60.0)
    parser.add_argument("--gpt_model", type=str, choices=["gpt-image-1", "gpt-image-1.5"], default="gpt-image-1.5")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    config = PipelineConfig(
        num_workers=args.num_workers,
        rate_limit_pause=args.rate_limit_pause,
        gpt_image_model=args.gpt_model,
        debug=args.debug,
    )

    process_style_alter_gpt_pool(
        input_csv=args.input_csv,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        style_alter_temp_dir=args.style_alter_temp_dir,
        csv_output_dir=args.csv_output_dir,
        config=config,
        resume=args.resume,
    )
