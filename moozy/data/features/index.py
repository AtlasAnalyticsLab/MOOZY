import logging
import os

logger = logging.getLogger(__name__)


def find_feature_multimap(
    feature_dirs: list[str] | os.PathLike,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Index all .h5 feature files by basename and by stem."""
    by_base: dict[str, list[str]] = {}
    by_stem: dict[str, list[str]] = {}

    for feature_dir in feature_dirs:
        if not os.path.isdir(feature_dir):
            continue
        for root, _, files in os.walk(feature_dir):
            for fname in files:
                if not fname.endswith(".h5"):
                    continue
                path = os.path.join(root, fname)
                stem = os.path.splitext(fname)[0]
                by_base.setdefault(fname, []).append(path)
                by_stem.setdefault(stem, []).append(path)

    for key in list(by_base.keys()):
        by_base[key] = sorted(set(by_base[key]))
    for key in list(by_stem.keys()):
        by_stem[key] = sorted(set(by_stem[key]))
    return by_base, by_stem


def list_feature_paths(feature_dirs: list[str] | os.PathLike) -> list[str]:
    """List all .h5 feature paths (no deduplication)."""
    by_base_multi, _ = find_feature_multimap(feature_dirs)
    all_paths: list[str] = []
    for paths in by_base_multi.values():
        all_paths.extend(paths)
    return sorted(all_paths)
