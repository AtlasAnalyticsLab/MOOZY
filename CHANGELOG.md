# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Two-stage training pipeline (self-supervised slide encoding + supervised case-level alignment)
- CLI interface (`moozy train stage1`, `moozy train stage2`, `moozy encode`)
- HuggingFace Hub integration for checkpoint downloads
- Multi-task supervised training with classification and survival objectives
- Slurm job templates for single-GPU, multi-GPU, and multi-node training
- CI workflow with ruff, mypy, and compile checks
- Training documentation for Stage 1, Stage 2, and encoding
