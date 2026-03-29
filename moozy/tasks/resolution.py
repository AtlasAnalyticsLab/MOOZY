from moozy.config.tasks import TaskSupervisionInfo
from moozy.data.stage2 import CaseRecord


def build_supervised_cases(
    task_info: TaskSupervisionInfo,
) -> tuple[list[CaseRecord], list[str]]:
    case_to_paths = task_info.get("case_to_paths") or {}
    case_to_slide_paths = task_info.get("case_to_slide_paths") or {}
    if not case_to_paths and not case_to_slide_paths:
        raise ValueError("No labeled cases found for the provided task CSVs.")

    cases: list[CaseRecord] = []
    dropped_empty_cases: list[str] = []
    if case_to_slide_paths:
        for case_id in sorted(case_to_slide_paths.keys()):
            slide_map = case_to_slide_paths[case_id]
            path_options = []
            for _, options in sorted(slide_map.items()):
                sorted_options = sorted(options)
                if sorted_options:
                    path_options.append(sorted_options)
            if not path_options:
                dropped_empty_cases.append(case_id)
                continue
            cases.append(
                {
                    "case_id": case_id,
                    "paths": [options[0] for options in path_options],
                    "path_options": path_options,
                }
            )
    else:
        for case_id in sorted(case_to_paths.keys()):
            paths = sorted(case_to_paths[case_id])
            if not paths:
                dropped_empty_cases.append(case_id)
                continue
            cases.append({"case_id": case_id, "paths": paths})

    return cases, dropped_empty_cases
