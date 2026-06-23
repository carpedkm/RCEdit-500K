"""
RCEdit-500K Pipeline - Part 1: GPT-4o Analysis

Single-call GPT-4o analysis using the unified prompt (gpt_prompt_forward.txt).
Determines edit type, constructs reference conditions, and generates instructions.

Usage:
    python multi_reference_part1_gpt_analysis.py \
        --metadata_csv data/metadata.csv \
        --data_dir /path/to/images \
        --output_dir outputs/part1
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

from utils.multi_reference_utils import (
    call_gpt_model,
    preprocess_image_pair,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PipelineConfig:
    model_name: str = "gpt-4o"
    image_max_size: int = 1024
    batch_save_interval: int = 100


GPT_OUTPUT_KEYS = [
    "type", "assertion", "reason",
    "segment", "generation", "personalization", "edit",
    "new_instruction", "new_instruction_weak",
]

CSV_FIELDNAMES = [
    "id", "dataset", "input_path", "target_path", "prompt",
] + GPT_OUTPUT_KEYS + ["error"]


# ============================================================================
# Data Loading
# ============================================================================

def load_metadata_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Load metadata CSV. Expected columns: id, input_path, target_path, prompt."""
    df = pd.read_csv(csv_path)
    return df.to_dict('records')


def partition_data(data: List[Any], machine_index: int, num_machines: int) -> List[Any]:
    """Partition data for distributed processing using round-robin."""
    if num_machines <= 1:
        return data
    partitioned = [item for i, item in enumerate(data) if i % num_machines == machine_index]
    print(f"  Machine {machine_index + 1}/{num_machines}: {len(partitioned)}/{len(data)} entries")
    return partitioned


def load_images(entry: Dict[str, Any], data_dir: Path, config: PipelineConfig) -> Optional[tuple]:
    """Load and preprocess input/target images from data_dir."""
    input_rel = entry.get("input_path", "")
    target_rel = entry.get("target_path", "")
    if not input_rel or not target_rel:
        return None

    input_path = data_dir / input_rel
    target_path = data_dir / target_rel

    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Target image not found: {target_path}")

    input_image = Image.open(input_path).convert("RGB")
    target_image = Image.open(target_path).convert("RGB")

    return preprocess_image_pair(input_image, target_image, target_size=config.image_max_size)


# ============================================================================
# Main Processing
# ============================================================================

def process_gpt_analysis(
    metadata_csv: str,
    data_dir: str,
    output_dir: str,
    machine_index: int = 0,
    num_machines: int = 1,
    config: PipelineConfig = None,
    debug: bool = False,
    resume: bool = False,
):
    if config is None:
        config = PipelineConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)

    print(f"GPT Analysis - Machine {machine_index + 1}/{num_machines}")
    print(f"Input CSV: {metadata_csv}")
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")

    all_entries = load_metadata_csv(metadata_csv)
    print(f"Total entries: {len(all_entries)}")
    if not all_entries:
        return

    entries = partition_data(all_entries, machine_index, num_machines)

    suffix = f"_m{machine_index}" if num_machines > 1 else ""
    output_csv_path = output_dir / f"gpt_analysis{suffix}.csv"

    # Resume support
    processed_ids = set()
    if resume and output_csv_path.exists():
        try:
            df = pd.read_csv(output_csv_path)
            processed_ids = set(df['id'].tolist())
            print(f"Resuming: {len(processed_ids)} already processed")
        except Exception:
            pass

    stats = {"total": len(entries), "processed": 0, "success": 0, "skipped": 0, "errors": 0}
    total_start_time = time.time()
    entry_times = []
    batch_buffer = []

    def flush_batch():
        if not batch_buffer:
            return
        write_header = not output_csv_path.exists() or output_csv_path.stat().st_size == 0
        with open(output_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerows(batch_buffer)
            f.flush()
            os.fsync(f.fileno())
        batch_buffer.clear()

    try:
        pbar = tqdm(enumerate(entries), total=len(entries), desc="GPT Analysis", unit="entry")
        for idx, entry in pbar:
            entry_start = time.time()
            entry_id = entry.get("id", f"entry_{idx}")

            if entry_id in processed_ids:
                stats["skipped"] += 1
                continue

            row = {
                "id": entry_id,
                "dataset": entry.get("dataset", ""),
                "input_path": entry.get("input_path", ""),
                "target_path": entry.get("target_path", ""),
                "prompt": entry.get("prompt", ""),
                "error": "",
            }
            for key in GPT_OUTPUT_KEYS:
                row[key] = ""

            # Load images
            try:
                images = load_images(entry, data_dir, config)
                if images is None:
                    row["error"] = "Missing input/target path"
                    batch_buffer.append(row)
                    stats["errors"] += 1
                    continue
                input_image, target_image = images
            except Exception as e:
                row["error"] = f"Image loading error: {str(e)}"
                batch_buffer.append(row)
                stats["errors"] += 1
                continue

            # Call GPT (single call)
            try:
                gpt_result = call_gpt_model(
                    input_image=input_image,
                    output_image=target_image,
                    prompt=entry.get("prompt", ""),
                    model_name=config.model_name,
                )
                for key in GPT_OUTPUT_KEYS:
                    value = gpt_result.get(key, "")
                    if isinstance(value, bool):
                        value = str(value).lower()
                    row[key] = value
                stats["success"] += 1
            except Exception as e:
                row["error"] = f"GPT error: {str(e)}"
                stats["errors"] += 1
                if debug:
                    traceback.print_exc()

            batch_buffer.append(row)
            processed_ids.add(entry_id)
            stats["processed"] += 1

            if len(batch_buffer) >= config.batch_save_interval:
                flush_batch()

            entry_time = time.time() - entry_start
            entry_times.append(entry_time)
            avg_time = sum(entry_times[-100:]) / len(entry_times[-100:])
            pbar.set_postfix({'ok': stats["success"], 'err': stats["errors"], 'avg': f'{avg_time:.1f}s'})

        pbar.close()
    finally:
        flush_batch()

    total_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"GPT ANALYSIS COMPLETE")
    print(f"Total: {stats['total']} | Success: {stats['success']} | Errors: {stats['errors']} | Skipped: {stats['skipped']}")
    print(f"Time: {total_time/60:.1f} min | Output: {output_csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RCEdit Part 1: GPT-4o Analysis")
    parser.add_argument("--metadata_csv", type=str, required=True, help="Path to input metadata CSV")
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory for images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results CSV")
    parser.add_argument("--machine_index", type=int, default=0)
    parser.add_argument("--num_machines", type=int, default=1)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--batch_interval", type=int, default=100)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()
    if args.machine_index < 0 or args.machine_index >= args.num_machines:
        parser.error(f"machine_index must be in [0, {args.num_machines - 1}]")

    config = PipelineConfig(model_name=args.model, batch_save_interval=args.batch_interval)
    process_gpt_analysis(
        metadata_csv=args.metadata_csv,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        machine_index=args.machine_index,
        num_machines=args.num_machines,
        config=config,
        debug=args.debug,
        resume=args.resume,
    )
