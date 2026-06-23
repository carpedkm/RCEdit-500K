"""
RCEdit-500K Pipeline - Part 2a: Reference Generation

Generates reference images for all edit types:
- add/replace/remove/background: Full reference generation using Flux-Klein-9B + Grounded-SAM-2
- style/alter: Only generates the intermediate source image (Part 2b does the GPT Image transfer)

Supports multi-machine and multi-GPU processing.

Usage:
    python multi_reference_part2a_generation.py \
        --input_csv outputs/part1/gpt_analysis.csv \
        --data_dir /path/to/images \
        --output_dir outputs/part2a/references \
        --style_alter_temp_dir outputs/part2a/style_alter_temp \
        --csv_output_dir outputs/part2a
"""

import csv
import json
import os
import time
import traceback
import argparse
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Any
from tqdm import tqdm
from dataclasses import dataclass
import pandas as pd
import torch
import torch.multiprocessing as mp
from queue import Empty

from utils.multi_reference_utils import (
    preprocess_image_pair,
    save_image,
    TYPE_PROCESSORS,
    set_model_dtype,
    get_model_dtype_str,
    set_unload_model,
    get_unload_model,
    set_model_seed,
    get_model_seed,
    generation_model,
    set_gpu_id,
)


@dataclass
class PipelineConfig:
    image_max_size: int = 1024
    batch_save_interval: int = 20


INPUT_FIELDNAMES = [
    "id", "dataset", "input_path", "target_path", "prompt",
    "type", "assertion", "reason",
    "segment", "generation", "personalization", "edit",
    "new_instruction", "new_instruction_weak", "error",
]

OUTPUT_FIELDNAMES = INPUT_FIELDNAMES + ["reference_path"]
STYLE_ALTER_TEMP_FIELDNAMES = INPUT_FIELDNAMES + ["generation_path"]


def load_input_csv(csv_path: str) -> List[Dict[str, Any]]:
    return pd.read_csv(csv_path).to_dict('records')


def partition_data(data, machine_index, num_machines):
    if num_machines <= 1:
        return data
    return [item for i, item in enumerate(data) if i % num_machines == machine_index]


def load_and_preprocess_images(entry, data_dir: Path, config):
    input_rel = entry.get("input_path", "")
    target_rel = entry.get("target_path", "")
    if not input_rel or not target_rel:
        return None

    input_path = data_dir / input_rel
    target_path = data_dir / target_rel
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Target not found: {target_path}")

    inp = Image.open(input_path).convert("RGB")
    tgt = Image.open(target_path).convert("RGB")
    return preprocess_image_pair(inp, tgt, target_size=config.image_max_size)


def process_style_alter_generation_only(entry, data_dir, style_alter_temp_dir, config):
    """Style/alter: only generate the intermediate source image."""
    entry_id = entry.get("id", "")
    edit_type = str(entry.get("type", "")).lower()
    result = {k: entry.get(k, "") for k in INPUT_FIELDNAMES}
    result["generation_path"] = ""

    try:
        images = load_and_preprocess_images(entry, data_dir, config)
        if images is None:
            result["error"] = "Missing input/target path"
            return result
        _, target_image = images
    except Exception as e:
        result["error"] = f"Image loading error: {e}"
        return result

    gen_prompt = entry.get("generation", "")
    if not gen_prompt:
        result["error"] = f"generation prompt required for '{edit_type}'"
        return result

    try:
        output_w, output_h = target_image.size
        src_image = generation_model(gen_prompt, width=output_w, height=output_h)
    except Exception as e:
        result["error"] = f"Generation error: {e}"
        return result

    safe_id = entry_id.replace("/", "_").replace("\\", "_")
    gen_filename = f"generation_{safe_id}_{edit_type}.png"
    try:
        save_image(src_image, str(style_alter_temp_dir / gen_filename))
        result["generation_path"] = gen_filename
        result["error"] = ""
    except Exception as e:
        result["error"] = f"Save error: {e}"
    return result


def process_single_entry(entry, data_dir, output_dir, style_alter_temp_dir, config):
    """Process a single entry."""
    entry_id = entry.get("id", "")
    edit_type = str(entry.get("type", "")).lower()
    assertion = str(entry.get("assertion", "")).lower()

    if edit_type in ["style", "alter"]:
        if assertion != "true":
            result = {k: entry.get(k, "") for k in INPUT_FIELDNAMES}
            result["generation_path"] = ""
            result["error"] = f"Skipped: assertion={assertion}"
            return ("style_alter", result)
        result = process_style_alter_generation_only(entry, data_dir, style_alter_temp_dir, config)
        return ("style_alter", result)

    # Normal types
    result = {k: entry.get(k, "") for k in INPUT_FIELDNAMES}
    result["reference_path"] = ""

    if assertion != "true" or edit_type in ("others", ""):
        result["error"] = f"Skipped: type={edit_type}, assertion={assertion}"
        return ("normal", result)

    processor = TYPE_PROCESSORS.get(edit_type)
    if processor is None:
        result["error"] = f"Unknown type: {edit_type}"
        return ("normal", result)

    try:
        images = load_and_preprocess_images(entry, data_dir, config)
        if images is None:
            result["error"] = "Missing input/target path"
            return ("normal", result)
        input_image, target_image = images
    except Exception as e:
        result["error"] = f"Image loading error: {e}"
        return ("normal", result)

    gpt_output = {k: entry.get(k, "") for k in ["type", "assertion", "segment", "generation",
                                                   "personalization", "edit", "new_instruction", "new_instruction_weak"]}
    try:
        ref_images = processor(gpt_output=gpt_output, input_image=input_image,
                               output_image=target_image, index=0, image_id=entry_id)
    except Exception as e:
        result["error"] = f"Processor error: {e}"
        return ("normal", result)

    if not ref_images:
        result["error"] = "No reference images generated"
        return ("normal", result)

    ref_paths = []
    for label, ref_img in ref_images:
        safe_id = entry_id.replace("/", "_").replace("\\", "_")
        fname = f"reference_{safe_id}_{label}.png"
        try:
            save_image(ref_img, str(output_dir / fname))
            ref_paths.append(fname)
        except Exception as e:
            result["error"] = f"Save error: {e}"
            return ("normal", result)

    result["reference_path"] = ";".join(ref_paths)
    result["error"] = ""
    return ("normal", result)


def worker_process(gpu_id, task_queue, result_queue, data_dir, output_dir,
                   style_alter_temp_dir, config_dict, model_dtype, unload_model, model_seed):
    """Worker process for multi-GPU processing."""
    set_gpu_id(gpu_id)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    set_model_dtype(model_dtype)
    set_model_seed(model_seed)
    set_unload_model(unload_model)

    config = PipelineConfig(**config_dict)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    style_alter_temp_dir = Path(style_alter_temp_dir)

    while True:
        try:
            task = task_queue.get(timeout=1)
            if task is None:
                break
            idx, entry = task
            result_type, result = process_single_entry(entry, data_dir, output_dir,
                                                        style_alter_temp_dir, config)
            result_queue.put((idx, result_type, result))
        except Empty:
            continue
        except Exception as e:
            print(f"[GPU {gpu_id}] Error: {e}")
            traceback.print_exc()


def flush_batch_to_csv(batch, output_csv_path, fieldnames):
    if not batch:
        return
    write_header = not output_csv_path.exists() or output_csv_path.stat().st_size == 0
    with open(output_csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(batch)
        f.flush()
        os.fsync(f.fileno())


def process_reference_generation(
    input_csv, data_dir, output_dir, style_alter_temp_dir, csv_output_dir,
    machine_index=0, num_machines=1, num_gpus=1, config=None, resume=False
):
    if config is None:
        config = PipelineConfig()

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    style_alter_temp_dir = Path(style_alter_temp_dir)
    csv_output_dir = Path(csv_output_dir)

    for d in [output_dir, style_alter_temp_dir, csv_output_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Reference Generation (Part 2a) - Machine {machine_index + 1}/{num_machines}")

    all_entries = load_input_csv(input_csv)
    entries = partition_data(all_entries, machine_index, num_machines)
    print(f"Processing {len(entries)} entries")

    suffix = f"_m{machine_index}" if num_machines > 1 else ""
    output_csv = csv_output_dir / f"reference_output{suffix}.csv"
    sa_csv = csv_output_dir / f"style_alter_temp{suffix}.csv"

    # Resume
    processed_ids = set()
    sa_processed = set()
    if resume:
        if output_csv.exists():
            try:
                processed_ids = set(pd.read_csv(output_csv)['id'].tolist())
            except Exception:
                pass
        if sa_csv.exists():
            try:
                sa_processed = set(pd.read_csv(sa_csv)['id'].tolist())
            except Exception:
                pass

    entries_to_process = []
    for e in entries:
        eid = e.get("id", "")
        etype = str(e.get("type", "")).lower()
        if etype in ["style", "alter"]:
            if eid not in sa_processed:
                entries_to_process.append(e)
        else:
            if eid not in processed_ids:
                entries_to_process.append(e)

    if not entries_to_process:
        print("All done.")
        return

    stats = {"success": 0, "errors": 0, "skipped": 0, "sa_success": 0, "sa_errors": 0}
    start_time = time.time()

    normal_buf, sa_buf = [], []
    pbar = tqdm(enumerate(entries_to_process), total=len(entries_to_process), desc="Generating")

    for idx, entry in pbar:
        try:
            rt, result = process_single_entry(entry, data_dir, output_dir, style_alter_temp_dir, config)
            if rt == "style_alter":
                err = result.get("error", "")
                if err and "Skipped" not in err:
                    stats["sa_errors"] += 1
                else:
                    stats["sa_success"] += 1
                sa_buf.append(result)
                if len(sa_buf) >= config.batch_save_interval:
                    flush_batch_to_csv(sa_buf, sa_csv, STYLE_ALTER_TEMP_FIELDNAMES)
                    sa_buf.clear()
            else:
                err = result.get("error", "")
                if err:
                    if "Skipped" in err:
                        stats["skipped"] += 1
                    else:
                        stats["errors"] += 1
                else:
                    stats["success"] += 1
                normal_buf.append(result)
                if len(normal_buf) >= config.batch_save_interval:
                    flush_batch_to_csv(normal_buf, output_csv, OUTPUT_FIELDNAMES)
                    normal_buf.clear()
        except Exception as e:
            stats["errors"] += 1
            traceback.print_exc()

        pbar.set_postfix({'ok': stats["success"], 'err': stats["errors"], 'sa': stats["sa_success"]})

    if normal_buf:
        flush_batch_to_csv(normal_buf, output_csv, OUTPUT_FIELDNAMES)
    if sa_buf:
        flush_batch_to_csv(sa_buf, sa_csv, STYLE_ALTER_TEMP_FIELDNAMES)

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed/60:.1f} min. Success: {stats['success']}, SA: {stats['sa_success']}, Errors: {stats['errors']}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="RCEdit Part 2a: Reference Generation")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory for images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for reference images")
    parser.add_argument("--style_alter_temp_dir", type=str, required=True)
    parser.add_argument("--csv_output_dir", type=str, required=True)
    parser.add_argument("--machine_index", type=int, default=0)
    parser.add_argument("--num_machines", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--batch_interval", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--seed", type=str, default="42")
    parser.add_argument("--unload_model", type=lambda x: x.lower() in ('true', '1', 'yes'), default=True)

    args = parser.parse_args()

    set_model_dtype(args.dtype if args.dtype != "bfloat16" else None)
    if args.seed.lower() == "random":
        set_model_seed("random")
    else:
        set_model_seed(int(args.seed))
    set_unload_model(args.unload_model)

    config = PipelineConfig(batch_save_interval=args.batch_interval)
    process_reference_generation(
        input_csv=args.input_csv,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        style_alter_temp_dir=args.style_alter_temp_dir,
        csv_output_dir=args.csv_output_dir,
        machine_index=args.machine_index,
        num_machines=args.num_machines,
        num_gpus=args.num_gpus,
        config=config,
        resume=args.resume,
    )
