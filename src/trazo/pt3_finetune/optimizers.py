"""Utility-based Perturbed Gradient Descent (UPGD).

Elsayed and Mahmood (2024). Each weight accumulates a running "utility"
(-grad * weight); high-utility weights are protected from both the gradient
update and the injected noise, which is what limits catastrophic forgetting when
fine-tuning on a small regional dataset.

Adapted from the WRI/Kerner Lab active-learning experiments. Changes from that
version:

* The global utility maximum is taken over absolute values and guarded against
  zero. In the original, a non-positive maximum flipped the sign of every
  scaled utility, which silently inverted the protection the optimizer exists to
  provide.
* Parameters with ``requires_grad=False`` are skipped rather than decayed, so
  UPGD composes with last-layer freezing and LoRA.
"""

from __future__ import annotations

import torch


class UPGD(torch.optim.Optimizer):
    """Utility-based perturbed gradient descent.

    Args:
        params: Parameters to optimize.
        lr: Learning rate.
        weight_decay: Decoupled weight decay applied to trainable parameters.
        beta_utility: EMA decay for the per-weight utility trace.
        sigma: Standard deviation of the perturbation noise.
    """

    def __init__(self, params, lr=1e-5, weight_decay=1e-3, beta_utility=0.999, sigma=1e-3):
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 0.0 <= beta_utility < 1.0:
            raise ValueError(f"beta_utility must be in [0, 1), got {beta_utility}")
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            beta_utility=beta_utility,
            sigma=sigma,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        device = self.param_groups[0]["params"][0].device
        global_max_util = torch.tensor(0.0, device=device)

        # Pass 1: update the utility trace and find the global maximum.
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                state["step"] += 1

                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(
                    -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                )

                bias_correction = 1 - group["beta_utility"] ** state["step"]
                current_max = (avg_utility / bias_correction).abs().max()
                global_max_util = torch.maximum(global_max_util, current_max)

        # A zero maximum means no signal yet (first step, or all-zero grads).
        # Falling through with a zero divisor would produce NaNs.
        if not torch.isfinite(global_max_util) or global_max_util <= 0:
            global_max_util = torch.tensor(1.0, device=device)

        # Pass 2: scaled, utility-gated update.
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue
                state = self.state[p]
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                noise = torch.randn_like(p.grad) * group["sigma"]

                scaled_utility = torch.sigmoid_(
                    (state["avg_utility"] / bias_correction) / global_max_util
                )
                p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                    (p.grad.data + noise) * (1 - scaled_utility),
                    alpha=-2.0 * group["lr"],
                )

        return loss
