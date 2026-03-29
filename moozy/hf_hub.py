import os

_HF_REPO_ID = "AtlasAnalyticsLab/MOOZY"
_HF_CHECKPOINT = "moozy.pt"


def ensure_checkpoint() -> str:
    """Download the MOOZY checkpoint from HuggingFace if not already cached."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=_HF_REPO_ID, filename=_HF_CHECKPOINT)


def ensure_tasks_dir() -> str:
    """Download the bundled task definitions from HuggingFace if not already cached."""
    from huggingface_hub import snapshot_download

    snapshot_dir = snapshot_download(repo_id=_HF_REPO_ID, allow_patterns=["tasks/**"])
    return os.path.join(snapshot_dir, "tasks")
