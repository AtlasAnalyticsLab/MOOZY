# Contributing to MOOZY

We welcome bug reports, documentation improvements, and feature suggestions.

## Reporting Issues

Open a [GitHub Issue](https://github.com/AtlasAnalyticsLab/MOOZY/issues)
with:
- A clear description of the problem
- Steps to reproduce (if applicable)
- Your environment (Python version, GPU, OS)

## Development Setup

```bash
git clone https://github.com/AtlasAnalyticsLab/MOOZY.git
cd MOOZY
make bootstrap   # installs dev deps + pre-commit hooks
make check        # runs format check, lint, typecheck, compile
```

## Code Style

- Formatting: [Ruff](https://docs.astral.sh/ruff/) (line length 120)
- Type checking: [mypy](https://mypy-lang.org/) on annotated modules
- Pre-commit hooks run automatically on `git commit`

## Pull Requests

1. Fork the repo and create a feature branch.
2. Run `make check` before pushing.
3. Open a PR with a clear description of the change.
