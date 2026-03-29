from collections import defaultdict
from typing import Any

import numpy as np


def _build_case_matrix(
    case_ids: list[str],
    mappings: list[dict[str, Any]],
    *,
    fill_value: int | float,
    dtype,
    cast_value,
) -> np.ndarray:
    matrix = np.full((len(case_ids), len(mappings)), fill_value, dtype=dtype)
    case_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, case_id in enumerate(case_ids):
        case_to_indices[case_id].append(idx)

    for task_idx, mapping in enumerate(mappings):
        for case_id, value in mapping.items():
            for row in case_to_indices.get(case_id, []):
                matrix[row, task_idx] = cast_value(value)
    return matrix


def build_case_label_matrix(
    case_ids: list[str],
    case_labels: list[dict[str, int]],
) -> np.ndarray:
    return _build_case_matrix(
        case_ids,
        case_labels,
        fill_value=-1,
        dtype=np.int64,
        cast_value=int,
    )


def build_case_event_matrix(
    case_ids: list[str],
    case_events: list[dict[str, int]],
) -> np.ndarray:
    return _build_case_matrix(
        case_ids,
        case_events,
        fill_value=-1,
        dtype=np.int64,
        cast_value=int,
    )


def build_case_time_matrix(
    case_ids: list[str],
    case_times: list[dict[str, float]],
) -> np.ndarray:
    return _build_case_matrix(
        case_ids,
        case_times,
        fill_value=-1.0,
        dtype=np.float32,
        cast_value=float,
    )
