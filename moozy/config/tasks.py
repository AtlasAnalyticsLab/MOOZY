from typing import TypedDict


class TaskConfig(TypedDict, total=False):
    task_type: str
    sample_col: str
    task_col: str | None
    event_col: str
    time_col: str


class TaskSpec(TypedDict):
    name: str
    key: str
    num_classes: int
    class_weights: list[float]
    label_map: dict[str, int]
    counts: list[int]
    task_type: str


class TaskSupervisionInfo(TypedDict):
    tasks: list[TaskSpec]
    case_labels: list[dict[str, int]]
    case_events: list[dict[str, int]]
    case_times: list[dict[str, float]]
    case_to_paths: dict[str, set[str]]
    case_to_slide_paths: dict[str, dict[str, list[str]]]
