<div align="center">

# RCEdit-500K

### Reference Completion for Image-Conditioned Image Editing

**ECCV 2026**

Jingxu Zhang<sup>1,2</sup>, Daneul Kim<sup>3</sup>, Yueming Pan<sup>2,4</sup>, Dong Chen<sup>2</sup>, Kai Qiu<sup>2</sup>,  
Yang Liu<sup>2</sup>, Yifan Yang<sup>2</sup>, Qi Dai<sup>2</sup>, Xiaoyan Sun<sup>1</sup>, Chong Luo<sup>1,2</sup>

<sup>1</sup>University of Science and Technology of China &nbsp; <sup>2</sup>Microsoft Research Asia  
<sup>3</sup>Seoul National University &nbsp; <sup>4</sup>Xi'an Jiaotong University

[![HuggingFace Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/carpedkm/RCEdit-500K)
[![Project Page](https://img.shields.io/badge/%F0%9F%8C%90-Project%20Page-blue)](https://carpedkm.github.io/RCEdit-500K)
[![GitHub](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/carpedkm/RCEdit-500K)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

This is the official code repository for **"RCEdit-500K: Reference Completion for Image-Conditioned Image Editing"** (ECCV 2026).

We reformulate image-conditioned image editing (ICIE) data construction as a **reference-completion** problem: high-quality text-conditioned image editing (TCIE) datasets already supply the input image, instruction, and edited target — the only missing component is a compatible reference image. Based on this insight, we introduce a scalable pipeline to construct **RCEdit-500K**, the first large-scale unified open ICIE dataset comprising 477K aligned quadruplets across six edit categories.

## Dataset

The RCEdit-500K dataset is available on HuggingFace:

**[https://huggingface.co/datasets/carpedkm/RCEdit-500K](https://huggingface.co/datasets/carpedkm/RCEdit-500K)**

Each sample is a quadruplet: `(input_image, reference_image, instruction, target_image)`.

### Supported Edit Types

| Type | Description | Reference Type |
|------|-------------|----------------|
| **add** | Add an object from reference to input | Concrete |
| **replace** | Replace an object with the one in reference | Concrete |
| **remove** | Remove the object indicated by reference | Concrete |
| **background** | Change background to match reference | Concrete |
| **style** | Transfer artistic style from reference | Abstract |
| **alter** | Transfer visual attributes from reference | Abstract |

## Data Generation Pipeline

This repository contains the full pipeline to reproduce RCEdit-500K or construct new ICIE datasets from any TCIE source data.

### Pipeline Overview

1. **GPT-4o Analysis** — Classifies edit type, generates reference synthesis prompts, and rewrites instructions (including weak-instruction augmentation)
2. **Reference Generation** — Type-specific reference image synthesis using Grounded-SAM-2 + Flux-Klein-9B
3. **GPT Image Processing** — Style/attribute transfer via GPT-Image-1.5 (for style/alter types only)
4. **Post-Filtering** — 5-dimensional VLM-based quality filtering (reference compatibility, reference quality, instruction correctness, original pair correctness, reference-target similarity)

### Type-Specific Reference Construction

| Type | Method | Models |
|------|--------|--------|
| **add/replace** | Segment target subject → personalize/re-background | Grounded-SAM-2, Flux-Klein-9B |
| **remove** | Localize subject → crop or bounding-box overlay | Grounded-SAM-2 |
| **background** | Inpaint foreground to isolate background | Flux-Klein-9B |
| **style/alter** | Generate intermediate image → attribute transfer | Flux-Klein-9B, GPT-Image-1.5 |

## Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (A100 recommended, V100 supported with `--dtype float16`)
- [OpenAI API key](https://platform.openai.com/api-keys) with access to GPT-4o and GPT-Image-1.5

### Setup

```bash
# Create conda environment
conda create -n rcedit python=3.10 -y
conda activate rcedit

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126

# Install SAM2 (https://github.com/facebookresearch/sam2)
git clone https://github.com/facebookresearch/sam2.git
pip install -e sam2

# Install Grounding DINO (https://github.com/IDEA-Research/GroundingDINO)
git clone https://github.com/IDEA-Research/GroundingDINO.git grounding_dino
pip install --no-build-isolation -e grounding_dino

# Install remaining dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### Download Model Checkpoints

```bash
# SAM2.1 checkpoints (https://github.com/facebookresearch/sam2#download-checkpoints)
mkdir -p checkpoints && cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
cd ..

# Grounding DINO checkpoints
mkdir -p gdino_checkpoints && cd gdino_checkpoints
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..

# Flux-Klein-9B will be automatically downloaded from HuggingFace on first use
```

## Usage

### Input Data Format

Prepare a metadata CSV with the following columns:

| Column | Description |
|--------|-------------|
| `id` | Unique entry identifier |
| `input_path` | Relative path to input image |
| `target_path` | Relative path to target/edited image |
| `prompt` | Text editing instruction |

Example:
```csv
id,input_path,target_path,prompt
sample_001,images/input_001.jpg,images/target_001.jpg,Add a cat on the sofa
sample_002,images/input_002.jpg,images/target_002.jpg,Change the sky to sunset
```

### Step 1: GPT-4o Analysis

```bash
python multi_reference_part1_gpt_analysis.py \
    --metadata_csv data/metadata.csv \
    --data_dir /path/to/images \
    --output_dir outputs/part1 \
    --model gpt-4o \
    --resume
```

### Step 2a: Reference Generation

```bash
python multi_reference_part2a_generation.py \
    --input_csv outputs/part1/gpt_analysis.csv \
    --data_dir /path/to/images \
    --output_dir outputs/part2a/references \
    --style_alter_temp_dir outputs/part2a/style_alter_temp \
    --csv_output_dir outputs/part2a \
    --num_gpus 1 \
    --dtype bfloat16 \
    --resume
```

### Step 2b: GPT Image Processing (Style/Alter only)

```bash
python multi_reference_part2b_gpt_pool.py \
    --input_csv outputs/part2a/style_alter_temp.csv \
    --data_dir /path/to/images \
    --output_dir outputs/part2b/references \
    --style_alter_temp_dir outputs/part2a/style_alter_temp \
    --csv_output_dir outputs/part2b \
    --gpt_model gpt-image-1.5 \
    --num_workers 12 \
    --resume
```

### Step 3: Post-Filtering

After completing Steps 2a and 2b, merge the output CSVs before running post-filtering:

```bash
# Merge reference_output.csv (add/replace/remove/background) and style_alter_output_gpt.csv (style/alter)
cat outputs/part2a/reference_output.csv > outputs/merged.csv
tail -n +2 outputs/part2b/style_alter_output_gpt.csv >> outputs/merged.csv
```

Then run filtering:

```bash
python multi_reference_post_filtering.py \
    --input_csv outputs/merged.csv \
    --data_dir /path/to/images \
    --ref_dir outputs/part2a/references \
    --output_dir outputs/filtering \
    --model gpt-4o \
    --threshold 70 \
    --resume
```

## Distributed Processing

All scripts support multi-machine distributed processing:

```bash
# Machine 0 of 4
python multi_reference_part1_gpt_analysis.py \
    --metadata_csv data/metadata.csv \
    --data_dir /path/to/images \
    --output_dir outputs/part1 \
    --machine_index 0 --num_machines 4
```

Part 2a also supports multi-GPU on a single machine:

```bash
python multi_reference_part2a_generation.py \
    --input_csv outputs/part1/gpt_analysis.csv \
    --data_dir /path/to/images \
    --output_dir outputs/part2a/references \
    --style_alter_temp_dir outputs/part2a/style_alter_temp \
    --csv_output_dir outputs/part2a \
    --num_gpus 4
```

## V100 Compatibility

For V100 GPUs (no bfloat16 support), use `--dtype float16`:

```bash
python multi_reference_part2a_generation.py \
    --input_csv outputs/part1/gpt_analysis.csv \
    --data_dir /path/to/images \
    --output_dir outputs/part2a/references \
    --style_alter_temp_dir outputs/part2a/style_alter_temp \
    --csv_output_dir outputs/part2a \
    --dtype float16
```

## Project Structure

```
RCEdit-500K/
├── multi_reference_part1_gpt_analysis.py   # Step 1: GPT-4o analysis
├── multi_reference_part2a_generation.py    # Step 2a: Reference generation
├── multi_reference_part2b_gpt_pool.py      # Step 2b: GPT Image style/alter
├── multi_reference_post_filtering.py       # Step 3: Post-filtering
├── utils/
│   └── multi_reference_utils.py            # Core utilities
├── prompts/
│   ├── gpt_prompt_forward.txt              # GPT-4o analysis prompt
│   ├── post_filtering_prompt.txt           # Post-filtering prompt
│   └── post_filtering_per_type/            # Type-specific filtering prompts
├── requirements.txt
├── LICENSE
└── README.md
```

### External Dependencies (cloned during setup)
- `sam2/` — [SAM2](https://github.com/facebookresearch/sam2) (Meta)
- `grounding_dino/` — [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) (IDEA-Research)

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{zhang2026rcedit,
  title={RCEdit-500K: Reference Completion for Image-Conditioned Image Editing},
  author={Zhang, Jingxu and Kim, Daneul and Pan, Yueming and Chen, Dong and Qiu, Kai and Liu, Yang and Yang, Yifan and Dai, Qi and Sun, Xiaoyan and Luo, Chong},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
