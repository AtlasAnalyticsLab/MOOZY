import logging


def log_task_coverage(
    logger: logging.Logger,
    task_info,
    labels,
    *,
    events,
    times,
) -> None:
    num_samples = labels.shape[0]
    for idx, task in enumerate(task_info["tasks"]):
        task_type = str(task.get("task_type", "classification")).strip().lower()
        if task_type == "survival":
            valid = (events[:, idx] >= 0) & (times[:, idx] >= 0)
            labeled = int(valid.sum())
            event_count = 0
            censored_count = 0
            if labeled > 0:
                event_vals = events[:, idx][valid]
                event_count = int((event_vals == 1).sum())
                censored_count = int((event_vals == 0).sum())
            logger.info(
                "  %s: %d/%d labeled, type=survival, events=%d, censored=%d",
                task["name"],
                labeled,
                num_samples,
                event_count,
                censored_count,
            )
            continue

        labeled = int((labels[:, idx] >= 0).sum())
        logger.info(
            "  %s: %d/%d labeled, type=classification, classes=%d",
            task["name"],
            labeled,
            num_samples,
            int(task["num_classes"]),
        )
