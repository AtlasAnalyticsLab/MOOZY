import csv
import logging
import os
import re
from typing import Iterable

import yaml

from moozy.config.tasks import TaskConfig, TaskSpec, TaskSupervisionInfo
from moozy.data.features import find_feature_multimap


def discover_task_csvs(task_dir: str) -> list[str]:
    """Recursively find all task.csv files under a directory."""
    task_dir = os.path.abspath(task_dir)
    if not os.path.isdir(task_dir):
        return []
    results = []
    for root, _dirs, files in os.walk(task_dir):
        if "task.csv" in files:
            results.append(os.path.join(root, "task.csv"))
    return sorted(results)


def task_name_from_path(task_path: str) -> str:
    """Derive a human-readable ``dataset_task`` name from a task CSV path."""
    dataset_name = os.path.basename(os.path.dirname(os.path.dirname(task_path)))
    task_id = os.path.basename(os.path.dirname(task_path))
    return f"{dataset_name}_{task_id}"


def sanitize_key(name: str, fallback: str = "unknown") -> str:
    """Replace non-alphanumeric characters (except ``_``, ``-``, ``.``) with ``_``."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return safe.strip("_") or fallback


def _resolve_paths(
    fname: str,
    by_base: dict[str, list[str]],
    by_stem: dict[str, list[str]],
) -> list[str]:
    if fname in by_base:
        return by_base[fname]
    return by_stem.get(os.path.splitext(fname)[0], [])


def load_task_config(task_path: str) -> TaskConfig:
    """Load the config.yaml that sits beside *task_path*.

    Every config.yaml has ``task_type`` (classification | survival) and
    ``sample_col``.  Survival tasks additionally need ``event_col`` and
    ``time_col`` (or ``task_col`` from which they are derived).
    """
    config_path = os.path.join(os.path.dirname(os.path.abspath(task_path)), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    task_type = str(cfg["task_type"]).strip().lower()
    sample_col = str(cfg.get("sample_col", "case_id")).strip().lower()
    task_col = cfg.get("task_col")

    result: TaskConfig = {
        "task_type": task_type,
        "sample_col": sample_col,
        "task_col": task_col,
    }
    if task_type == "survival":
        event_col = cfg.get("event_col")
        time_col = cfg.get("time_col")
        if event_col and time_col:
            result["event_col"] = str(event_col).strip()
            result["time_col"] = str(time_col).strip()
        else:
            result["event_col"] = f"{task_col}_event"
            result["time_col"] = f"{task_col}_days"
    return result


def _read_task_csv_rows(
    task_path: str,
    task_config: TaskConfig,
    by_base: dict[str, list[str]],
    by_stem: dict[str, list[str]],
    task_name: str,
    logger: logging.Logger,
) -> list[tuple]:
    """Read one task CSV into validated row tuples.

    Classification rows: ``(h5_path, label_str, case_id, slide_key)``
    Survival rows:       ``(h5_path, case_id, slide_key, event, time_val)``
    """
    task_type = task_config["task_type"]
    event_col = task_config.get("event_col")
    time_col = task_config.get("time_col")

    rows: list[tuple] = []
    unresolved = 0

    with open(task_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            fname = (row.get("filename") or "").strip()
            if not fname:
                continue
            h5_paths = _resolve_paths(fname, by_base, by_stem)
            if not h5_paths:
                unresolved += 1
                continue

            slide_key = os.path.basename(fname)
            case_raw = row.get("case_id")
            case_id = str(case_raw).strip() if case_raw is not None else ""
            if not case_id or case_id.lower() == "nan":
                case_id = os.path.splitext(os.path.basename(fname))[0]

            if task_type == "survival":
                event = int(str(row[event_col]).strip())
                time_val = float(row[time_col])
                if time_val < 0:
                    continue
                for h5_path in h5_paths:
                    rows.append((h5_path, case_id, slide_key, event, time_val))
            else:
                label = row.get("label")
                if label is None:
                    continue
                for h5_path in h5_paths:
                    rows.append((h5_path, str(label), case_id, slide_key))

    if unresolved > 0:
        logger.info("Task %s: %d rows had unresolved feature paths", task_name, unresolved)
    return rows


def _build_classification_task(
    task_name: str,
    task_key: str,
    rows: list[tuple],
    case_to_paths: dict[str, set],
    case_to_slide_paths: dict[str, dict[str, set]],
) -> tuple[TaskSpec, dict[str, int]] | None:
    """Build classification label maps, class weights, and task spec.

    Rows: ``(h5_path, label_str, case_id, slide_key)``.
    """
    file_to_label_raw: dict[str, str] = {}
    case_to_label_raw: dict[str, str] = {}
    for h5_path, lab, case_id, slide_key in rows:
        existing = file_to_label_raw.get(h5_path)
        if existing is not None and existing != lab:
            raise ValueError("Conflicting labels for %s in task %s: %s vs %s" % (h5_path, task_name, existing, lab))
        file_to_label_raw[h5_path] = lab
        existing_case = case_to_label_raw.get(case_id)
        if existing_case is not None and existing_case != lab:
            raise ValueError(
                "Conflicting labels for case_id %s in task %s: %s vs %s" % (case_id, task_name, existing_case, lab)
            )
        case_to_label_raw[case_id] = lab
        case_to_paths.setdefault(case_id, set()).add(h5_path)
        case_to_slide_paths.setdefault(case_id, {}).setdefault(slide_key, set()).add(h5_path)

    labels_unique = sorted(set(case_to_label_raw.values()))
    label_map = {lab: idx for idx, lab in enumerate(labels_unique)}
    num_classes = len(labels_unique)
    if num_classes == 0:
        return None

    case_to_label = {cid: label_map[lab] for cid, lab in case_to_label_raw.items()}

    counts = [0] * num_classes
    for idx in case_to_label.values():
        counts[idx] += 1
    total = sum(counts)
    class_weights = [float(total) / float(num_classes * c) if c > 0 else 0.0 for c in counts]

    task_spec: TaskSpec = {
        "name": task_name,
        "key": task_key,
        "num_classes": num_classes,
        "class_weights": class_weights,
        "label_map": label_map,
        "counts": counts,
        "task_type": "classification",
    }
    return task_spec, case_to_label


def _build_survival_task(
    task_name: str,
    task_key: str,
    rows: list[tuple],
    case_to_paths: dict[str, set],
    case_to_slide_paths: dict[str, dict[str, set]],
    logger: logging.Logger,
) -> tuple[TaskSpec, dict[str, int], dict[str, float]] | None:
    """Build survival event/time maps and task spec.

    Rows: ``(h5_path, case_id, slide_key, event, time_val)``.
    """
    case_to_event: dict[str, int] = {}
    case_to_time: dict[str, float] = {}

    for h5_path, case_id, slide_key, event, time_val in rows:
        existing_time = case_to_time.get(case_id)
        if existing_time is not None and existing_time != time_val:
            raise ValueError(
                "Conflicting survival times for case_id %s in task %s: %.6f vs %.6f"
                % (case_id, task_name, existing_time, time_val)
            )
        existing_event = case_to_event.get(case_id)
        if existing_event is not None and existing_event != event:
            raise ValueError(
                "Conflicting event values for case_id %s in task %s: %d vs %d"
                % (case_id, task_name, existing_event, event)
            )
        case_to_event[case_id] = event
        case_to_time[case_id] = time_val
        case_to_paths.setdefault(case_id, set()).add(h5_path)
        case_to_slide_paths.setdefault(case_id, {}).setdefault(slide_key, set()).add(h5_path)

    total = len(case_to_event)
    if total == 0:
        return None

    event_count = sum(1 for e in case_to_event.values() if e == 1)
    logger.info(
        "Survival task %s: %d cases (%d events, %d censored)",
        task_name,
        total,
        event_count,
        total - event_count,
    )
    task_spec: TaskSpec = {
        "name": task_name,
        "key": task_key,
        "num_classes": 1,
        "class_weights": [1.0],
        "label_map": {},
        "counts": [total],
        "task_type": "survival",
    }
    return task_spec, case_to_event, case_to_time


def load_task_supervision(
    feature_dirs: Iterable[str],
    task_csvs: list[str],
    logger: logging.Logger,
) -> TaskSupervisionInfo | None:
    """Load task CSVs and build per-task label/event/time mappings."""
    by_base, by_stem = find_feature_multimap(feature_dirs)
    tasks: list[TaskSpec] = []
    case_labels: list[dict[str, int]] = []
    case_events: list[dict[str, int]] = []
    case_times: list[dict[str, float]] = []
    case_to_paths: dict[str, set] = {}
    case_to_slide_paths: dict[str, dict[str, set]] = {}
    seen_keys: dict[str, int] = {}

    for task_path in task_csvs:
        task_name = task_name_from_path(task_path)
        task_config = load_task_config(task_path)
        task_type = task_config["task_type"]

        rows = _read_task_csv_rows(task_path, task_config, by_base, by_stem, task_name, logger)
        if not rows:
            logger.warning("No valid rows for task %s; skipping.", task_name)
            continue

        task_key_base = sanitize_key(task_name)
        suffix = seen_keys.get(task_key_base, 0)
        seen_keys[task_key_base] = suffix + 1
        task_key = task_key_base if suffix == 0 else f"{task_key_base}_{suffix}"

        if task_type == "survival":
            result = _build_survival_task(task_name, task_key, rows, case_to_paths, case_to_slide_paths, logger)
            if result is None:
                continue
            task_spec, case_to_event, case_to_time = result
            tasks.append(task_spec)
            case_labels.append({})
            case_events.append(case_to_event)
            case_times.append(case_to_time)
        else:
            result = _build_classification_task(task_name, task_key, rows, case_to_paths, case_to_slide_paths)
            if result is None:
                continue
            task_spec, case_to_label = result
            tasks.append(task_spec)
            case_labels.append(case_to_label)
            case_events.append({})
            case_times.append({})

    if not tasks:
        return None

    return {
        "tasks": tasks,
        "case_labels": case_labels,
        "case_events": case_events,
        "case_times": case_times,
        "case_to_paths": case_to_paths,
        "case_to_slide_paths": {
            case_id: {slide_key: sorted(paths) for slide_key, paths in sorted(slide_map.items())}
            for case_id, slide_map in sorted(case_to_slide_paths.items())
        },
    }
