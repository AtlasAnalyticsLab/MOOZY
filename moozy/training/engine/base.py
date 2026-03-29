from typing import Any

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


class BaseEngine:
    """Loop skeleton.  Receives fully-constructed objects, owns only iteration.

    Subclasses implement :meth:`run` (the loop) and :meth:`training_step`
    (a single forward/backward pass).
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: Any,
        callbacks: list[Any] = None,  # type: ignore[assignment]
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.callbacks: list[Any] = callbacks if callbacks is not None else []

    @property
    def raw_model(self) -> nn.Module:
        """Return the unwrapped model (strips DDP)."""
        return self.model.module if isinstance(self.model, DDP) else self.model

    def fire(self, hook: str, **kwargs: Any) -> None:
        """Invoke *hook* on every callback that implements it."""
        for cb in self.callbacks:
            fn = getattr(cb, hook, None)
            if fn is not None:
                fn(**kwargs)

    def run(self) -> Any:
        """Execute the training loop.  Subclasses must override."""
        raise NotImplementedError
