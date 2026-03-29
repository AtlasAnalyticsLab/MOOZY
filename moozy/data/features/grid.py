import numpy as np

SLIDE_GRID_STEP_TOLERANCE = 0.25


def build_grid_from_coords(
    features: np.ndarray,
    coords: np.ndarray,
    expected_step: float,
    step_tolerance: float = SLIDE_GRID_STEP_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild a uniform spatial grid from unordered features and coordinates."""
    if expected_step is None:
        raise ValueError("expected_step (patch size) must be provided to enforce uniform spacing.")

    feat_dim = features.shape[1]
    step_x = float(expected_step)
    step_y = float(expected_step)
    tol = float(step_tolerance)

    unique_x = np.unique(coords[:, 0])
    unique_y = np.unique(coords[:, 1])
    min_x, max_x = float(unique_x.min()), float(unique_x.max())
    min_y, max_y = float(unique_y.min()), float(unique_y.max())

    w = int(np.floor((max_x - min_x) / step_x + 0.5)) + 1
    h = int(np.floor((max_y - min_y) / step_y + 0.5)) + 1

    col_idx = np.round((coords[:, 0] - min_x) / step_x).astype(int)
    row_idx = np.round((coords[:, 1] - min_y) / step_y).astype(int)
    snapped_x = min_x + col_idx * step_x
    snapped_y = min_y + row_idx * step_y

    if np.abs(snapped_x - coords[:, 0]).max() > tol * step_x or np.abs(snapped_y - coords[:, 1]).max() > tol * step_y:
        raise ValueError(
            f"Coordinate spacing deviates from uniform grid beyond tolerance "
            f"(tol={tol}, step=({step_x:.2f},{step_y:.2f}))"
        )

    grid = np.zeros((h, w, feat_dim), dtype=features.dtype)
    for i in range(features.shape[0]):
        r = row_idx[i]
        c = col_idx[i]
        if r < 0 or r >= h or c < 0 or c >= w:
            raise ValueError(f"Coordinate index out of bounds after snapping: {(r, c)} vs grid {(h, w)}")
        grid[r, c] = features[i]

    xs_axis = (min_x + np.arange(w) * step_x).astype(np.int64)
    ys_axis = (min_y + np.arange(h) * step_y).astype(np.int64)
    return grid, xs_axis, ys_axis
