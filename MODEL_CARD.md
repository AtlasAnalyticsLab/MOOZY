# Model Card: MOOZY

**Paper:** [arXiv:2603.27048](https://arxiv.org/abs/2603.27048)

## Model Description

MOOZY is a patient-first foundation model for computational pathology. It produces 768-dimensional embeddings at the slide and case (patient) level from whole-slide image feature grids.

The model has two components trained in sequence:

- **Slide encoder** (Stage 1): A 6-layer Vision Transformer (d=768, 12 heads, 2D ALiBi positional encoding, 4 register tokens) pretrained with masked self-distillation on 77,134 public slide feature grids at 20x and 40x magnification.
- **Case transformer** (Stage 2): A 3-layer transformer that aggregates slide embeddings into a single case embedding, fine-tuned with multi-task supervision over 333 clinical tasks (205 classification + 128 survival) from 56 public datasets.

Both stages use patch features from a frozen ViT-S/8 (Lunit DINO, 21.67M params, 384-D output).

| Component | Params |
|-----------|--------|
| Patch encoder (frozen) | 21.67M |
| Slide encoder | 42.8M |
| Case transformer | 21.3M |
| **Total** | **85.77M** |

## Training Data

All training data is public.

**Stage 1** (self-supervised): 77,134 slide feature grids (53,286 at 20x, 23,848 at 40x) extracted from ~1.67 billion patches across ~31.8 TB of raw WSI data, spanning 23 anatomical sites.

**Stage 2** (supervised): 41,089 cases (45,179 slides) across 333 tasks from 56 datasets: all 32 TCGA cohorts, all 10 CPTAC cohorts, REG, BC-Therapy, BRACS, CAMELYON17, DHMC Kidney, DHMC LUAD, EBRAINS, IMP Colorectum, IMP Cervix, MBC, MUT-HET-RCC, NADT Prostate, NAT-BRCA, and PANDA. Supervision covers classification labels and four survival endpoints (OS, DSS, DFI, PFI).

Anatomical sites: adrenal gland, bladder, brain, breast, cervix, colon/rectum, esophagus, eye, head and neck, kidney, liver/bile ducts, lung, lymph node, ovary, pancreas, prostate, skin, soft tissue, stomach, testis, thymus, thyroid, uterus.

## Training Procedure

- **Stage 1**: 200 epochs (14,400 steps, ~436 GPU-hours on 8 GPUs). Effective batch size 1,024. AdamW with cosine LR schedule. EMA teacher with cosine momentum (0.996 to 1.0). Multi-crop: 2 global (20x20) + 4 local (12x12) with block masking (ratio 0.1-0.5).
- **Stage 2**: 20 epochs (1,000 steps, ~512 GPU-hours on 8 GPUs). Effective batch size 1,024 cases (micro batch 1, 128 accumulation steps). AdamW with base LR 5e-5, cosine schedule, gradient clipping at 0.3. Classification: cross-entropy with label smoothing (0.03) and inverse-frequency weighting. Survival: discrete-time hazard NLL with adaptive binning.

## Intended Use

- Computational pathology research.
- Frozen-feature extraction for downstream slide-level and case-level tasks.
- As a pretrained backbone for fine-tuning on pathology classification and survival prediction.

## Out-of-Scope Use

- **Not a diagnostic tool.** Not validated for clinical decision-making. Any clinical use requires independent validation and regulatory approval.
- Not designed for non-pathology images (radiology, natural images, etc.).
- Not designed for stains other than H&E. Performance on IHC, IF, or special stains is untested.

## Limitations

- Trained exclusively on public data, which skews toward TCGA/CPTAC demographics (predominantly North American). Performance on underrepresented populations is unknown.
- Treatment Response (MBC) showed high variance across folds, indicating limited generalization for that specific task.

## Ethical Considerations

- Training data comes from public repositories (TCGA, CPTAC, etc.) collected under institutional review board protocols. Demographic representation is limited by the composition of these datasets.
- This model should not be used as a standalone diagnostic system. It is a research tool that produces embeddings for downstream analysis.
- Users deploying MOOZY on patient data should follow applicable data governance and privacy regulations.

## License

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Research and non-commercial use only.
