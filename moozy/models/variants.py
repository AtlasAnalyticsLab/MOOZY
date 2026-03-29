ENCODER_VARIANTS: dict[str, dict[str, int]] = {
    "tiny": {
        "n_layers": 12,
        "d_model": 192,
        "n_heads": 3,
        "dim_feedforward": 768,
    },
    "small": {
        "n_layers": 12,
        "d_model": 384,
        "n_heads": 6,
        "dim_feedforward": 1536,
    },
    "base": {
        "n_layers": 12,
        "d_model": 768,
        "n_heads": 12,
        "dim_feedforward": 3072,
    },
    "base_half_depth": {
        "n_layers": 6,
        "d_model": 768,
        "n_heads": 12,
        "dim_feedforward": 3072,
    },
    "base_quarter_depth": {
        "n_layers": 3,
        "d_model": 768,
        "n_heads": 12,
        "dim_feedforward": 3072,
    },
    "large": {
        "n_layers": 24,
        "d_model": 1024,
        "n_heads": 16,
        "dim_feedforward": 4096,
    },
    "large_half_depth": {
        "n_layers": 12,
        "d_model": 1024,
        "n_heads": 16,
        "dim_feedforward": 4096,
    },
    "large_quarter_depth": {
        "n_layers": 6,
        "d_model": 1024,
        "n_heads": 16,
        "dim_feedforward": 4096,
    },
}
