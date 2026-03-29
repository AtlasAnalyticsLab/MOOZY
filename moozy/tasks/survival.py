import logging

import numpy as np

from moozy.config.tasks import TaskSupervisionInfo


def build_survival_bin_edges(
    task_info: TaskSupervisionInfo,
    train_events: np.ndarray,
    train_times: np.ndarray,
    *,
    target_bins: int,
    min_bins: int,
    max_bins: int,
    logger: logging.Logger | None = None,
) -> dict[str, list[float]]:
    """Build per-task quantile bin edges for discrete survival training."""
    if train_events is None or train_times is None:
        return {}

    events = np.asarray(train_events)
    times = np.asarray(train_times)
    if events.ndim != 2 or times.ndim != 2:
        raise ValueError(
            "Expected 2-D train_events/train_times for survival bin construction, "
            f"got events={events.shape}, times={times.shape}."
        )

    req_bins = int(max(min_bins, min(max_bins, target_bins)))
    upscale_events_per_bin = max(1, int(3 * req_bins))
    bin_edges_by_key: dict[str, list[float]] = {}
    max_time_by_task: list[tuple[str, str, float]] = []
    compact_rows: list[str] = []
    survival_tasks = 0

    for idx, task in enumerate(task_info.get("tasks", [])):
        task_type = str(task.get("task_type", "classification")).strip().lower()
        if task_type != "survival":
            continue
        survival_tasks += 1
        key = str(task.get("key", f"task_{idx}"))
        name = str(task.get("name", key))
        valid = (events[:, idx] >= 0) & (times[:, idx] >= 0)
        valid_count = int(valid.sum())
        if valid_count <= 0:
            bin_edges_by_key[key] = []
            compact_rows.append(f"{key}: valid=0 events=0 bins=1")
            continue

        ev = events[:, idx][valid]
        tm = times[:, idx][valid].astype(np.float64, copy=False)
        event_times = tm[ev == 1]
        event_count = int(event_times.shape[0])

        if event_count < req_bins:
            task_bins = max(min_bins, max(1, event_count))
        else:
            extra_bins = int((event_count - req_bins) // upscale_events_per_bin)
            task_bins = min(max_bins, req_bins + extra_bins)
        task_bins = int(min(task_bins, valid_count))

        ref_times = event_times if event_times.shape[0] >= 2 else tm
        if task_bins <= 1 or ref_times.shape[0] <= 1:
            edges = np.empty((0,), dtype=np.float32)
        else:
            quantiles = np.linspace(0.0, 1.0, num=task_bins + 1, dtype=np.float64)[1:-1]
            raw_edges = np.quantile(ref_times, quantiles)
            edges = np.unique(np.asarray(raw_edges, dtype=np.float32))
        eff_bins = int(edges.shape[0]) + 1
        bin_edges_by_key[key] = [float(value) for value in edges.tolist()]
        max_time_by_task.append((key, name, float(tm.max())))
        compact_rows.append(f"{key}: valid={valid_count} events={event_count} bins={eff_bins}")

    if logger:
        logger.info(
            (
                "Configured discrete survival bins for %d survival tasks "
                "(target_bins=%d, min_bins=%d, max_bins=%d; adaptive_upscale=+1 bin/%d events above target)."
            ),
            survival_tasks,
            req_bins,
            int(min_bins),
            int(max_bins),
            int(upscale_events_per_bin),
        )
        if compact_rows:
            logger.info("Survival bin preview: %s", ", ".join(compact_rows[:8]))
        if max_time_by_task:
            max_vals = np.array([row[2] for row in max_time_by_task], dtype=np.float64)
            median_max = float(np.median(max_vals))
            if median_max > 0.0:
                for key, name, tmax in max_time_by_task:
                    if tmax < (0.1 * median_max):
                        logger.warning(
                            "Survival task %s (%s) has unusually small time scale (max=%.3f vs median max=%.3f). "
                            "Per-task quantile bins will be used.",
                            key,
                            name,
                            tmax,
                            median_max,
                        )
    return bin_edges_by_key
