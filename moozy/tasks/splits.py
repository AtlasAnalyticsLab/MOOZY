import numpy as np


def split_train_val_indices_task_stratified(
    labels: np.ndarray,
    val_ratio: float,
    seed: int,
    *,
    events: np.ndarray | None = None,
    task_types: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], int]]:
    """Best-effort per-task stratified train/val split for multi-task labels."""
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("labels must be 2-D for task-stratified split")

    n = int(labels.shape[0])
    if n < 2:
        return np.arange(n, dtype=np.int64), np.empty((0,), dtype=np.int64), {}

    vr = float(val_ratio)
    if not (0.0 < vr < 1.0):
        raise ValueError("val_ratio must be in (0, 1) for stratified split")

    val_len = max(1, int(n * vr))
    val_len = min(val_len, n - 1)

    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    num_tasks = int(labels.shape[1])
    if events is not None:
        events = np.asarray(events)
    if task_types is None:
        task_types = ["classification"] * num_tasks

    targets: dict[tuple[int, int], int] = {}
    for task_idx in range(num_tasks):
        task_type = task_types[task_idx] if task_idx < len(task_types) else "classification"
        if task_type == "survival" and events is not None:
            task_events = events[:, task_idx]
            valid = task_events >= 0
            if not valid.any():
                continue
            for event_val in (0, 1):
                idx = np.flatnonzero(task_events == event_val)
                if idx.size <= 1:
                    continue
                target = int(round(idx.size * vr))
                targets[(task_idx, int(event_val))] = max(1, min(target, idx.size - 1))
            continue

        task_labels = labels[:, task_idx]
        valid = task_labels >= 0
        if not valid.any():
            continue
        for label in np.unique(task_labels[valid]):
            idx = np.flatnonzero(task_labels == label)
            if idx.size <= 1:
                continue
            target = int(round(idx.size * vr))
            targets[(task_idx, int(label))] = max(1, min(target, idx.size - 1))

    if not targets:
        order = np.arange(n, dtype=np.int64)
        rng.shuffle(order)
        return order[val_len:], order[:val_len], {}

    target_keys = set(targets.keys())
    case_keys: list[list[tuple[int, int]]] = []
    for row_idx in range(n):
        keys: list[tuple[int, int]] = []
        for task_idx in range(num_tasks):
            task_type = task_types[task_idx] if task_idx < len(task_types) else "classification"
            if task_type == "survival" and events is not None:
                event_val = events[row_idx, task_idx]
                if event_val < 0:
                    continue
                key = (task_idx, int(event_val))
            else:
                label = labels[row_idx, task_idx]
                if label < 0:
                    continue
                key = (task_idx, int(label))
            if key in target_keys:
                keys.append(key)
        case_keys.append(keys)

    order = np.arange(n, dtype=np.int64)
    rng.shuffle(order)
    order = sorted(order.tolist(), key=lambda idx: len(case_keys[idx]), reverse=True)

    remaining = dict(targets)
    val_selected: list[int] = []
    train_selected: list[int] = []
    for idx in order:
        if len(val_selected) >= val_len:
            train_selected.append(idx)
            continue
        needed = sum(1 for key in case_keys[idx] if remaining.get(key, 0) > 0)
        if needed > 0:
            val_selected.append(idx)
            for key in case_keys[idx]:
                if remaining.get(key, 0) > 0:
                    remaining[key] -= 1
        else:
            train_selected.append(idx)

    if len(val_selected) < val_len and train_selected:
        rng.shuffle(train_selected)
        fill = train_selected[: max(0, val_len - len(val_selected))]
        val_selected.extend(fill)
        train_selected = train_selected[len(fill) :]

    train_idx = np.asarray(train_selected, dtype=np.int64)
    val_idx = np.asarray(val_selected, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx, remaining
