import torch
import torch.nn as nn

from moozy.training.loss import DiscreteHazardLoss

from .case_transformer import CaseAggregator
from .heads import build_task_head
from .stage2_encoding import encode_case_batch
from .stage2_tasks import run_stage2_task_heads


class MOOZY(nn.Module):
    """Slide encoder with multi-task heads for supervised fine-tuning."""

    def __init__(
        self,
        slide_encoder: nn.Module,
        task_names: list[str] | None = None,
        task_keys: list[str] | None = None,
        task_num_classes: list[int] | None = None,
        task_class_weights: list[list[float]] | None = None,
        task_types: list[str] | None = None,
        classification_head_type: str = "mlp",
        survival_head_type: str = "mlp",
        head_dropout: float = 0.0,
        label_smoothing: float = 0.0,
        case_transformer_layers: int = 2,
        case_transformer_heads: int = 8,
        case_transformer_ffn_dim: int = 0,
        case_transformer_dropout: float = 0.1,
        case_transformer_layerscale_init: float = 1e-5,
        case_transformer_layer_drop: float = 0.0,
        case_transformer_qk_norm: bool = False,
        case_transformer_num_registers: int = 0,
        survival_num_bins: int = 8,
        survival_bin_edges: dict[str, list[float]] | None = None,
    ):
        super().__init__()
        self.slide_encoder = slide_encoder
        self.classification_head_type = str(classification_head_type).lower()
        self.survival_head_type = str(survival_head_type).lower()
        if self.classification_head_type not in ("linear", "mlp"):
            raise ValueError(f"Unsupported classification_head_type: {self.classification_head_type}")
        if self.survival_head_type not in ("linear", "mlp"):
            raise ValueError(f"Unsupported survival_head_type: {self.survival_head_type}")
        self.survival_num_bins = max(1, survival_num_bins)
        self.head_dropout = head_dropout
        self.label_smoothing = label_smoothing
        self.feature_dropout = nn.Dropout(self.head_dropout) if self.head_dropout > 0.0 else nn.Identity()
        ffn_dim = case_transformer_ffn_dim if case_transformer_ffn_dim > 0 else None
        self.case_transformer = CaseAggregator(
            d_model=self.slide_encoder.d_model,
            num_layers=case_transformer_layers,
            num_heads=case_transformer_heads,
            dim_feedforward=ffn_dim,
            dropout=case_transformer_dropout,
            layerscale_init=case_transformer_layerscale_init,
            layer_drop=case_transformer_layer_drop,
            qk_norm=case_transformer_qk_norm,
            num_registers=case_transformer_num_registers,
        )

        self.discrete_hazard_loss_fn = DiscreteHazardLoss(reduction="mean")
        self.survival_edge_buffer_names: dict[str, str] = {}
        provided_survival_edges = survival_bin_edges or {}

        self.task_heads = nn.ModuleDict()
        self.task_names: list[str] = []
        self.task_keys: list[str] = []
        self.task_types: list[str] = []
        self.task_head_types: list[str] = []
        self.task_weight_names: list[str | None] = []
        if task_num_classes and task_names:
            for idx, num_classes in enumerate(task_num_classes):
                key = task_keys[idx] if task_keys and idx < len(task_keys) else f"task_{idx}"
                task_type = task_types[idx] if task_types and idx < len(task_types) else "classification"
                if task_type == "survival":
                    edge_values = provided_survival_edges.get(key)
                    if edge_values is not None:
                        edge_tensor = torch.as_tensor(edge_values, dtype=torch.float32).reshape(-1)
                        if edge_tensor.numel() > 0:
                            edge_tensor = torch.unique(edge_tensor.sort().values)
                        out_dim = int(edge_tensor.numel()) + 1
                    else:
                        edge_tensor = torch.empty((0,), dtype=torch.float32)
                        out_dim = int(self.survival_num_bins)
                    out_dim = max(1, out_dim)
                    edge_name = f"survival_bin_edges_{idx}"
                    self.register_buffer(edge_name, edge_tensor)
                    self.survival_edge_buffer_names[key] = edge_name
                else:
                    out_dim = int(num_classes)
                head_type = self.survival_head_type if task_type == "survival" else self.classification_head_type
                self.task_heads[key] = build_task_head(
                    d_model=self.slide_encoder.d_model,
                    num_classes=out_dim,
                    head_type=head_type,
                )
                self.task_names.append(task_names[idx])
                self.task_keys.append(key)
                self.task_types.append(task_type)
                self.task_head_types.append(head_type)
                if task_class_weights and idx < len(task_class_weights):
                    weight_tensor = torch.tensor(task_class_weights[idx], dtype=torch.float32)
                    weight_name = f"task_weight_{idx}"
                    self.register_buffer(weight_name, weight_tensor)
                    self.task_weight_names.append(weight_name)
                else:
                    self.task_weight_names.append(None)

    def set_activation_checkpointing(self, enabled: bool = True) -> None:
        """Enable or disable activation checkpointing for stage-2 forward paths."""
        enabled = bool(enabled)
        self.slide_encoder.set_activation_checkpointing(enabled)
        self.case_transformer.set_activation_checkpointing(enabled)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cls_out, task_labels, task_events, task_times, sample_count = encode_case_batch(
            self.slide_encoder,
            self.case_transformer,
            batch,
        )

        task_weights = [
            getattr(self, weight_name, None) if weight_name is not None else None
            for weight_name in self.task_weight_names
        ]
        task_outputs = run_stage2_task_heads(
            cls_out=cls_out,
            feature_dropout=self.feature_dropout,
            task_heads=self.task_heads,
            task_types=self.task_types,
            task_head_types=self.task_head_types,
            task_weights=task_weights,
            task_labels=task_labels,
            task_events=task_events,
            task_times=task_times,
            label_smoothing=self.label_smoothing,
            hazard_loss_fn=self.discrete_hazard_loss_fn,
            survival_bin_edges={
                key: getattr(self, edge_name) if edge_name is not None else None
                for key, edge_name in self.survival_edge_buffer_names.items()
            },
        )

        return {
            "loss": task_outputs["loss_task"],
            "loss_task": task_outputs["loss_task"],
            "loss_cls_component": task_outputs["loss_cls_component"],
            "loss_surv_component": task_outputs["loss_surv_component"],
            "task_count_cls": task_outputs["task_count_cls"],
            "task_count_surv": task_outputs["task_count_surv"],
            "cls": cls_out,
            "logits": task_outputs["logits"],
            "task_labels": task_labels,
            "task_events": task_events,
            "task_times": task_times,
            "sample_count": sample_count,
        }
