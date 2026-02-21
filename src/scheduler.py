"""Learning rate, weight decay, and momentum schedulers for T-JEPA."""

import math

from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineSchedule(LRScheduler):
    """Linear warmup followed by cosine decay to 0."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, last_epoch: int = -1) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        step = self.last_epoch
        if step < self.warmup_steps:
            # Linear warmup
            scale = step / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [base_lr * scale for base_lr in self.base_lrs]


class CosineWDSchedule:
    """Cosine schedule for weight decay from start_wd to end_wd."""

    def __init__(self, optimizer, start_wd: float, end_wd: float, total_steps: int) -> None:
        self.optimizer = optimizer
        self.start_wd = start_wd
        self.end_wd = end_wd
        self.total_steps = total_steps
        self._step = 0

    def step(self) -> None:
        progress = self._step / max(1, self.total_steps)
        wd = self.end_wd + 0.5 * (self.start_wd - self.end_wd) * (1.0 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            if group.get("apply_wd", True):
                group["weight_decay"] = wd
        self._step += 1


def momentum_schedule(ema_start: float, ema_end: float, total_steps: int):
    """Generate momentum values via cosine schedule from ema_start to ema_end.

    Args:
        ema_start: Initial EMA momentum (e.g. 0.996).
        ema_end: Final EMA momentum (e.g. 1.0).
        total_steps: Total number of training steps.

    Yields:
        Momentum value for each step.
    """
    for step in range(total_steps):
        progress = step / max(1, total_steps - 1)
        m = ema_end - (ema_end - ema_start) * (math.cos(math.pi * progress) + 1) / 2
        yield m
