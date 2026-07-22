# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added


## [0.1.1] - 2026-07-21

### Changed

- Synchronized the repository documentation, model card, and citation metadata with the ECCV 2026 camera-ready paper.
- Expanded the Notes from the Authors with discussions of generalization, case aggregation, benchmarking, and Stage 2 hyperparameters.
- Updated the reported evaluation results and supporting analysis to cover all sixteen held-out tasks.


## [0.1.0] - 2026-03-31

### Added

- Two-stage training pipeline (self-supervised slide encoding + supervised case-level alignment)
- CLI interface (`moozy train stage1`, `moozy train stage2`, `moozy encode`)
- HuggingFace Hub integration for checkpoint downloads
- Multi-task supervised training with classification and survival objectives
- Slurm job templates for single-GPU, multi-GPU, and multi-node training
- CI workflow with ruff, mypy, and compile checks
- Training documentation for Stage 1, Stage 2, and encoding
