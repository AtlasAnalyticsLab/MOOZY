# MOOZY: A Patient-First Foundation Model for Computational Pathology

<!-- TODO: update arXiv URL when paper is posted -->
<p align="center">
  <a href="https://atlasanalyticslab.github.io/MOOZY/"><img src="https://img.shields.io/badge/Project-Page-4285F4?logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-B31B1B?logo=arxiv" alt="arXiv"></a>
  <a href="https://huggingface.co/AtlasAnalyticsLab/MOOZY"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow" alt="HuggingFace"></a>
  <a href="https://pypi.org/project/moozy/"><img src="https://img.shields.io/pypi/v/moozy?logo=pypi&logoColor=white&label=PyPI" alt="PyPI"></a>
  <a href="https://github.com/AtlasAnalyticsLab/MOOZY/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/AtlasAnalyticsLab/MOOZY/main/assets/overview_0.5.png" alt="MOOZY: a patient-first foundation model for computational pathology, whole-slide image encoding, and case-level representation learning">
</p>

**MOOZY is a foundation model for computational pathology that treats the patient case, not the individual slide, as the fundamental unit of representation.** It encodes one or more whole-slide images (WSIs) into a single 768-dimensional case-level embedding that captures dependencies across all slides from the same patient. Trained entirely on public data with 85.8M parameters (14x smaller than GigaPath), MOOZY outperforms larger models on classification and survival prediction tasks across diverse organs and cancer types.

---

## Table of Contents

- [Quick Start](#quick-start)
  - [Environment Setup](#environment-setup)
  - [Using the Output](#using-the-output)
- [Method Overview](#method-overview)
- [Training](#training)
  - [Scripts](#scripts)
  - [SLURM Jobs](#slurm-jobs)
- [Results](#results)
- [Acknowledgment](#acknowledgment)
- [Citation](#citation)
- [License](#license)

## Quick Start

```bash
pip install moozy
```

Model weights download automatically on first use. No access gates, no manual downloads, no HuggingFace approval.

```bash
# Encode a patient case from pre-extracted H5 feature files
moozy encode slide_1.h5 slide_2.h5 --output case_embedding.h5

# Encode directly from raw whole-slide images
moozy encode slide_1.svs slide_2.svs --output case_embedding.h5
```

Or use the Python API:

```python
from moozy.encoding import run_encoding

run_encoding(
    slide_paths=["slide_1.h5", "slide_2.h5"],
    output_path="case_embedding.h5",
)
```

The output H5 file contains a 768-d case-level embedding ready for downstream tasks: classification, survival prediction, or retrieval.

All encoding arguments (data, runtime, raw WSI options, mixed precision) are documented in [docs/encode.md](https://github.com/AtlasAnalyticsLab/MOOZY/blob/main/docs/encode.md).

### Environment Setup

```bash
conda create -n moozy python=3.12 -y
conda activate moozy
pip install moozy
```

<details>
<summary><b>venv</b></summary>

```bash
python -m venv moozy-env
source moozy-env/bin/activate
pip install moozy
```

</details>

<details>
<summary><b>uv</b></summary>

```bash
uv venv moozy-env
source moozy-env/bin/activate
uv pip install moozy
```

</details>

### Using the Output

The output is a standard H5 file. Load it with `h5py`:

```python
import h5py

with h5py.File("case_embedding.h5", "r") as f:
    embedding = f["features"][:]  # (768,) float32 case-level embedding

# Use the embedding for downstream tasks
# e.g., as input to a linear probe, k-NN, MLP probe, or clustering
```

## Method Overview

MOOZY is a two-stage pipeline that first learns slide-level representations through self-supervised learning, then aligns them with clinical meaning through multi-task supervision.

**Stage 1: Self-supervised slide encoder.** A vision transformer learns context-aware spatial representations from 77,134 unlabeled public histopathology slides (~1.67 billion patches across 23 anatomical sites) using masked self-distillation. No labels are used. The slide encoder captures tissue morphology, spatial context, and inter-region relationships across the whole slide.

**Stage 2: Patient-aware multi-task alignment.** The pretrained slide encoder is fine-tuned end-to-end with a case transformer that models dependencies across all slides from the same patient. A learnable [CASE] token aggregates per-slide embeddings into a single case-level representation. Multi-task supervision across 333 tasks (205 classification, 128 survival) from 56 public datasets provides broad clinical grounding. All task heads are discarded after training, leaving a general-purpose patient encoder.

For detailed model specifications, see the [model card](https://github.com/AtlasAnalyticsLab/MOOZY/blob/main/MODEL_CARD.md).

## Training

Both training stages are fully open-source and reproducible using only public data. All training arguments (data, model, optimization, checkpointing, logging, runtime) are documented in the [Stage 1](https://github.com/AtlasAnalyticsLab/MOOZY/blob/main/docs/stage_1.md) and [Stage 2](https://github.com/AtlasAnalyticsLab/MOOZY/blob/main/docs/stage_2.md) training docs.

### Scripts

For local multi-GPU training, use the launch scripts in [`scripts/`](https://github.com/AtlasAnalyticsLab/MOOZY/tree/main/scripts):

```bash
# Stage 1: Self-supervised pretraining
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/train_stage1.sh

# Stage 2: Multi-task alignment
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/train_stage2.sh
```

### SLURM Jobs

SLURM job templates are provided in [`slurm/`](https://github.com/AtlasAnalyticsLab/MOOZY/tree/main/slurm) for cluster environments:

| Script | Description |
|---|---|
| [`slurm/single_gpu.sh`](https://github.com/AtlasAnalyticsLab/MOOZY/blob/main/slurm/single_gpu.sh) | Single-GPU training |
| [`slurm/multi_gpu.sh`](https://github.com/AtlasAnalyticsLab/MOOZY/blob/main/slurm/multi_gpu.sh) | Multi-GPU training on one node |
| [`slurm/multi_node.sh`](https://github.com/AtlasAnalyticsLab/MOOZY/blob/main/slurm/multi_node.sh) | Multi-node distributed training |
| [`slurm/inference.sh`](https://github.com/AtlasAnalyticsLab/MOOZY/blob/main/slurm/inference.sh) | Patient encoding |

## Results

Frozen-feature MLP probe comparison against slide encoder baselines on eight held-out tasks. **Bold** indicates the best result per metric.

| Task | Metric | CHIEF | GigaPath | PRISM | Madeleine | TITAN | MOOZY |
|---|---|---|---|---|---|---|---|
| Residual Cancer Burden | F1 | 0.46 | 0.45 | 0.46 | 0.51 | 0.43 | **0.56** |
| | AUC | 0.60 | 0.55 | 0.58 | 0.63 | 0.58 | **0.74** |
| | Bal Acc | 0.44 | 0.40 | 0.43 | 0.48 | 0.38 | **0.51** |
| TP53 Mutation | F1 | 0.82 | 0.76 | 0.85 | 0.84 | **0.87** | **0.87** |
| | AUC | 0.81 | 0.76 | 0.85 | 0.85 | **0.91** | 0.86 |
| | Bal Acc | 0.83 | 0.76 | 0.84 | 0.84 | **0.88** | 0.86 |
| BAP1 Mutation | F1 | 0.86 | 0.84 | 0.80 | 0.85 | 0.84 | **0.89** |
| | AUC | 0.75 | 0.63 | 0.71 | 0.78 | **0.82** | 0.79 |
| | Bal Acc | 0.75 | 0.66 | 0.66 | 0.75 | 0.75 | **0.78** |
| ACVR2A Mutation | F1 | 0.89 | 0.80 | 0.85 | 0.89 | 0.87 | **0.91** |
| | AUC | 0.80 | 0.74 | 0.83 | 0.76 | 0.79 | **0.91** |
| | Bal Acc | 0.80 | 0.65 | 0.81 | 0.81 | 0.76 | **0.90** |
| Histologic Grade | F1 | 0.71 | 0.77 | 0.73 | 0.75 | 0.73 | **0.78** |
| | AUC | 0.71 | **0.77** | 0.67 | 0.74 | 0.71 | 0.75 |
| | Bal Acc | 0.73 | **0.77** | 0.73 | 0.74 | 0.73 | **0.77** |
| KRAS Mutation | F1 | 0.77 | 0.77 | 0.72 | 0.81 | 0.80 | **0.85** |
| | AUC | 0.76 | 0.72 | 0.61 | 0.70 | **0.80** | **0.80** |
| | Bal Acc | 0.74 | 0.76 | 0.63 | 0.77 | **0.81** | 0.79 |
| IDH Status | F1 | 0.92 | 0.94 | 0.91 | 0.92 | 0.94 | **0.97** |
| | AUC | 0.96 | 0.97 | 0.95 | 0.96 | 0.97 | **0.99** |
| | Bal Acc | 0.92 | 0.94 | 0.91 | 0.91 | 0.94 | **0.97** |
| Treatment Response | F1 | 0.53 | 0.51 | 0.57 | 0.49 | 0.49 | **0.58** |
| | AUC | **0.70** | 0.68 | 0.69 | 0.59 | 0.60 | 0.68 |
| | Bal Acc | 0.48 | 0.40 | **0.51** | 0.35 | 0.37 | 0.48 |

<!-- TODO: update arXiv link when paper is posted -->
<p align="center"><sub>Mean values from five-fold frozen-feature evaluation. Full results with confidence intervals are in the <a href="https://arxiv.org/abs/XXXX.XXXXX">paper</a>.</sub></p>

Across all eight tasks, MOOZY improves macro averages over TITAN by +7.4% weighted F1, +5.5% AUC, and +7.8% balanced accuracy, and over PRISM by +8.8% F1, +10.7% AUC, and +9.8% balanced accuracy, with 14x fewer parameters than GigaPath.

## Acknowledgment

This work was supported by NSERC-DG RGPIN-2022-05378 [M.S.H], Amazon Research Award [M.S.H], and Gina Cody RIF [M.S.H], FRQNT scholarship [Y.K]. Computational resources were provided in part by [Calcul Qu&eacute;bec](https://www.calculquebec.ca) and the [Digital Research Alliance of Canada](https://www.alliancecan.ca).

## Citation

If you find MOOZY useful, please cite:

<!-- TODO: update arXiv ID when paper is posted -->
```bibtex
@article{kotp2026moozy,
  title   = {MOOZY: A Patient-First Foundation Model for Computational Pathology},
  author  = {Kotp, Yousef and Trinh, Vincent Quoc-Huy and Pal, Christopher and Hosseini, Mahdi S.},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License

This project is licensed under [CC BY-NC-SA 4.0](https://github.com/AtlasAnalyticsLab/MOOZY/blob/main/LICENSE).
