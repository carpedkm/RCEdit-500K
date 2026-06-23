"""
RCEdit-500K Pipeline - Post Filtering

5-dimensional VLM-based quality filtering using GPT-4o.
Evaluates each (input, reference, target, instruction) sample on:
  A. Reference Compatibility
  B. Reference Image Reasonableness
  C. Instruction Correctness
  D. Original Pair Correctness
  E. Similarity Check

Samples with all scores >= 70 pass.

Usage:
    python multi_reference_post_filtering.py \
        --input_csv outputs/part2a/reference_output.csv \
        --data_dir /path/to/images \
        --ref_dir outputs/part2a/references \
        --output_dir outputs/filtering
"""

import csv
import json
import os
import re
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
    get_openai_client,
    call_gpt_with_backoff,
    encode_image_from_pil,
)


@dataclass
class FilteringConfig:
    model_name: str = "gpt-4o"
    image_max_size: int = 1024
    batch_save_interval: int = 20
    passing_threshold: int = 70
    temperature: float = 0.1


SCORE_KEYS = [
    "A_reference_compatibility",
    "B_reference_image_reasonableness",
    "C_instruction_correctness",
    "D_original_pair_correctness",
    "E_Similarity_Check",
]

# Type-specific score keys (some types have fewer criteria)
TYPE_SCORE_KEYS = {
    "add": ["A_reference_image_reasonableness", "B_instruction_correctness",
            "C_original_pair_correctness", "D_Similarity_Check"],
    "replace": ["A_reference_image_reasonableness", "B_instruction_correctness",
                "C_original_pair_correctness", "D_Similarity_Check"],
    "remove": ["A_reference_image_reasonableness", "B_instruction_correctness",
               "C_original_pair_correctness"],
    "background": ["A_reference_image_reasonableness", "B_instruction_correctness",
                   "C_original_pair_correctness", "D_Similarity_Check"],
    "style": SCORE_KEYS,
    "alter": SCORE_KEYS,
}

PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "post_filtering_prompt.txt"
TYPE_PROMPT_DIR = Path(__file__).parent / "prompts" / "post_filtering_per_type"

INPUT_FIELDNAMES = [
    "id", "dataset", "input_path", "target_path", "prompt",
    "type", "assertion", "reason",
    "segment", "generation", "personalization", "edit",
    "new_instruction", "new_instruction_weak",
    "error", "reference_path",
]

REFERENCE_FIELDNAMES = ["reference_index", "current_reference_path"]

FILTER_FIELDNAMES = [
    "A_reference_compatibility_score", "A_reference_compatibility_reason",
    "B_reference_image_reasonableness_score", "B_reference_image_reasonableness_reason",
    "C_instruction_correctness_score", "C_instruction_correctness_reason",
    "D_original_pair_correctness_score", "D_original_pair_correctness_reason",
    "E_Similarity_Check_score", "E_Similarity_Check_reason",
    "average_score", "pass", "failure_modes", "filtering_error",
]

OUTPUT_FIELDNAMES = INPUT_FIELDNAMES + REFERENCE_FIELDNAMES + FILTER_FIELDNAMES


def load_prompt_template(task_type=None, use_type=False):
    if use_type and task_type:
        type_path = TYPE_PROMPT_DIR / f"{task_type.lower().strip()}.txt"
        if type_path.exists():
            return type_path.read_text(encoding='utf-8')
    return PROMPT_TEMPLATE_PATH.read_text(encoding='utf-8')


def load_image(path: Path, max_size=1024):
    if not path.exists():
        return None
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def extract_scores_from_response(raw_text, task_type=None, use_type=False):
    patterns = [
        r'\{[\s\S]*"A_reference_compatibility"[\s\S]*\}',
        r'\{[\s\S]*"A_reference_image_reasonableness"[\s\S]*\}',
    ]
    for pattern in patterns:
        m = re.search(pattern, raw_text)
        if m:
            json_str = m.group(0)
            brace_count = 0
            end_idx = 0
            for i, c in enumerate(json_str):
                if c == '{': brace_count += 1
                elif c == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            if end_idx > 0:
                try:
                    return json.loads(json_str[:end_idx])
                except json.JSONDecodeError:
                    pass
    return {}


def call_gpt_filtering(input_image, target_image, reference_image,
                       original_instruction, new_instruction, task_type,
                       model_name="gpt-4o", use_type=False, temperature=0.1):
    client = get_openai_client()
    template = load_prompt_template(task_type, use_type)
    prompt_text = template.replace("{task_type}", task_type) \
                          .replace("{original_instruction}", original_instruction) \
                          .replace("{new_instruction}", new_instruction)

    content = [{"type": "text", "text": prompt_text}]
    for img in [input_image, reference_image, target_image]:
        b64 = encode_image_from_pil(img)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}})

    messages = [{"role": "user", "content": content}]
    response = call_gpt_with_backoff(client, messages, model_name=model_name, temperature=temperature)
    return extract_scores_from_response(response.choices[0].message.content, task_type, use_type)


def calculate_average_score(scores, task_type=None, use_type=False):
    keys = TYPE_SCORE_KEYS.get(task_type.lower(), SCORE_KEYS) if use_type and task_type else SCORE_KEYS
    total, count = 0.0, 0
    for k in keys:
        if k in scores and isinstance(scores[k], dict):
            v = scores[k].get("score", 0)
            if isinstance(v, (int, float)):
                total += v
                count += 1
    return round(total / count, 2) if count else 0.0


def check_pass(scores, threshold=70, task_type=None, use_type=False):
    keys = TYPE_SCORE_KEYS.get(task_type.lower(), SCORE_KEYS) if use_type and task_type else SCORE_KEYS
    for k in keys:
        if k not in scores or not isinstance(scores[k], dict):
            return False
        v = scores[k].get("score", 0)
        if not isinstance(v, (int, float)) or v < threshold:
            return False
    return True


def process_filtering(
    input_csv, data_dir, ref_dir, output_dir,
    machine_index=0, num_machines=1, config=None, resume=False, use_type=False,
):
    if config is None:
        config = FilteringConfig()

    data_dir = Path(data_dir)
    ref_dir = Path(ref_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Post-Filtering - Machine {machine_index + 1}/{num_machines}")

    all_entries = pd.read_csv(input_csv).to_dict('records')
    valid = [e for e in all_entries if str(e.get("assertion", "")).lower().strip() == "true"]
    print(f"Valid entries: {len(valid)}")

    if num_machines > 1:
        entries = [e for i, e in enumerate(valid) if i % num_machines == machine_index]
    else:
        entries = valid

    suffix = f"_m{machine_index}" if num_machines > 1 else ""
    output_csv_path = output_dir / f"post_filtering_output{suffix}.csv"

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

    stats = {"success": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    batch_buffer = []

    def flush():
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

    start_time = time.time()
    pbar = tqdm(entries, desc="Filtering")

    try:
        for entry in pbar:
            ref_path_str = str(entry.get("reference_path", "")).strip()
            if not ref_path_str or ref_path_str == 'nan':
                row = {k: entry.get(k, "") for k in INPUT_FIELDNAMES}
                for k in REFERENCE_FIELDNAMES + FILTER_FIELDNAMES:
                    row[k] = ""
                row["filtering_error"] = "No reference"
                batch_buffer.append(row)
                stats["skipped"] += 1
                continue

            # Load images
            inp = load_image(data_dir / entry.get("input_path", ""), config.image_max_size)
            tgt = load_image(data_dir / entry.get("target_path", ""), config.image_max_size)
            if inp is None or tgt is None:
                row = {k: entry.get(k, "") for k in INPUT_FIELDNAMES}
                for k in REFERENCE_FIELDNAMES + FILTER_FIELDNAMES:
                    row[k] = ""
                row["filtering_error"] = "Failed to load images"
                batch_buffer.append(row)
                stats["errors"] += 1
                continue

            task_type = str(entry.get("type", "")).strip()
            original_inst = str(entry.get("prompt", "")).strip()
            new_inst = str(entry.get("new_instruction", "")).strip()

            ref_paths = [p.strip() for p in ref_path_str.split(';') if p.strip()]

            for ref_idx, rp in enumerate(ref_paths):
                ref_img = load_image(ref_dir / os.path.basename(rp), config.image_max_size)
                if ref_img is None:
                    continue

                ref_row = {k: entry.get(k, "") for k in INPUT_FIELDNAMES}
                for k in REFERENCE_FIELDNAMES + FILTER_FIELDNAMES:
                    ref_row[k] = ""
                ref_row["reference_index"] = ref_idx
                ref_row["current_reference_path"] = rp

                try:
                    scores = call_gpt_filtering(inp, tgt, ref_img, original_inst, new_inst,
                                                task_type, config.model_name, use_type, config.temperature)

                    for sk in SCORE_KEYS:
                        ref_row[f"{sk}_score"] = ""
                        ref_row[f"{sk}_reason"] = ""

                    for sk, sd in scores.items():
                        if not isinstance(sd, dict):
                            continue
                        for std_key in SCORE_KEYS:
                            if sk == std_key or sk.split('_')[0] == std_key.split('_')[0]:
                                ref_row[f"{std_key}_score"] = sd.get("score", "")
                                ref_row[f"{std_key}_reason"] = sd.get("reason", "")
                                break

                    ref_row["average_score"] = calculate_average_score(scores, task_type, use_type)
                    ref_row["pass"] = check_pass(scores, config.passing_threshold, task_type, use_type)
                    ref_row["failure_modes"] = json.dumps(scores.get("failure_modes", []))
                    ref_row["filtering_error"] = ""

                    stats["success"] += 1
                    if ref_row["pass"]:
                        stats["passed"] += 1
                    else:
                        stats["failed"] += 1
                except Exception as e:
                    ref_row["filtering_error"] = f"GPT error: {e}"
                    stats["errors"] += 1

                batch_buffer.append(ref_row)
                if len(batch_buffer) >= config.batch_save_interval:
                    flush()

            pbar.set_postfix({'pass': stats["passed"], 'fail': stats["failed"], 'err': stats["errors"]})
    finally:
        flush()

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"Passed: {stats['passed']}, Failed: {stats['failed']}, Errors: {stats['errors']}")
    if stats['success'] > 0:
        print(f"Pass rate: {stats['passed']/stats['success']*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RCEdit Post-Filtering")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory for source images")
    parser.add_argument("--ref_dir", type=str, required=True, help="Directory with reference images")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--machine_index", type=int, default=0)
    parser.add_argument("--num_machines", type=int, default=1)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--threshold", type=int, default=70)
    parser.add_argument("--batch_interval", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use_type", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.1)

    args = parser.parse_args()
    config = FilteringConfig(
        model_name=args.model,
        batch_save_interval=args.batch_interval,
        passing_threshold=args.threshold,
        temperature=args.temperature,
    )
    process_filtering(
        input_csv=args.input_csv,
        data_dir=args.data_dir,
        ref_dir=args.ref_dir,
        output_dir=args.output_dir,
        machine_index=args.machine_index,
        num_machines=args.num_machines,
        config=config,
        resume=args.resume,
        use_type=args.use_type,
    )
