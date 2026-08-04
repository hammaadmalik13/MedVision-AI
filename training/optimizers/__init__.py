"""Optimizer and scheduler factories."""

from typing import Any

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau


class Ranger(optim.Optimizer):
    """Ranger optimizer combining RAdam and Lookahead."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        alpha: float = 0.5,
        k: int = 6,
        betas: tuple[float, float] = (0.95, 0.999),
        eps: float = 1e-5,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, alpha=alpha, k=k)
        super().__init__(params, defaults)
        self._step_count = 0
        self._slow_params: list[dict[str, Any]] = []

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["slow_buffer"] = p.data.clone()

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group["eps"])
                step_size = group["lr"]
                update = exp_avg / denom
                p.data.add_(update, alpha=-step_size)

                if state["step"] % group["k"] == 0:
                    alpha = group["alpha"]
                    state["slow_buffer"].mul_(1 - alpha).add_(p.data, alpha=alpha)
                    p.data.copy_(state["slow_buffer"])

        self._step_count += 1
        return loss


OPTIMIZER_REGISTRY = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
    "ranger": Ranger,
}

SCHEDULER_REGISTRY = {
    "cosine": CosineAnnealingLR,
    "reduce_on_plateau": ReduceLROnPlateau,
    "onecycle": OneCycleLR,
}


def get_optimizer(name: str, params, **kwargs) -> optim.Optimizer:
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer: {name}")
    return OPTIMIZER_REGISTRY[name](params, **kwargs)


def get_scheduler(name: str, optimizer: optim.Optimizer, **kwargs):
    if name not in SCHEDULER_REGISTRY:
        raise ValueError(f"Unknown scheduler: {name}")
    if name == "reduce_on_plateau":
        return SCHEDULER_REGISTRY[name](optimizer, **kwargs)
    return SCHEDULER_REGISTRY[name](optimizer, **kwargs)
