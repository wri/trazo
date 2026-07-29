"""Low-rank adaptation (LoRA) for the convolutional layers of a U-Net.

Adapted from Microsoft's LoRA reference implementation:

    Copyright (c) Microsoft Corporation. All rights reserved.
    Licensed under the MIT License (MIT).
    https://github.com/microsoft/LoRA

Adapted for smp U-Nets with EfficientNet encoders, via the WRI/Kerner Lab
active-learning experiments. Changes from that version:

* The wrapper no longer re-registers the wrapped conv's parameters on itself.
  In the original this was load-bearing by accident: the alias kept the
  pre-injection state-dict keys resolvable, so a checkpoint loaded after
  injection still reached the real conv weights. It also made every base weight
  appear twice in ``named_parameters()``, so any caller optimizing
  ``model.parameters()`` on a LoRA-wrapped model would double-count them.
  Here the checkpoint is loaded *before* injection instead (see
  :mod:`trazo.pt3_finetune.trainers`), which gets the same result without the
  duplicate parameters.
* ``lora_dropout`` is actually applied. In the original it was accepted,
  documented and set to 0.1 in the shipped config, but ``forward`` never
  referenced it, so every LoRA run effectively used dropout 0.
* ``inject_lora`` returns the number of wrapped layers and raises when that is
  zero, instead of silently returning an unmodified model that trains nothing.
* Dead ``Conv1d``/``Conv2d``/``Conv3d``/``Conv2dSamePadLoRA`` subclasses were
  removed; they passed a class where an instance was expected and could not run.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # EfficientNet encoders use a custom same-padding conv.
    from efficientnet_pytorch.utils import Conv2dStaticSamePadding
except Exception:  # pragma: no cover - optional dependency
    Conv2dStaticSamePadding = ()


class ConvLoRA(nn.Module):
    """Wrap a conv layer with a trainable low-rank update.

    The base convolution is left untouched and frozen; the LoRA branch is a
    rank-``r`` factorization of the weight delta, applied as a second (bias-free)
    convolution so the wrapped layer's stride, padding, dilation and grouping all
    still apply, and so dropout on the LoRA input means what it says.
    """

    def __init__(self, conv: nn.Module, r: int = 0, lora_alpha: float = 1.0,
                 lora_dropout: float = 0.0, merge_weights: bool = False) -> None:
        super().__init__()
        self.conv = conv
        self.r = int(r)
        self.lora_alpha = float(lora_alpha)
        self.merge_weights = bool(merge_weights)
        self.merged = False
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        kernel = conv.kernel_size
        k = kernel[0] if isinstance(kernel, tuple) else kernel
        fan_in = conv.in_channels // conv.groups

        if self.r > 0:
            self.lora_A = nn.Parameter(conv.weight.new_zeros(self.r, fan_in * k * k))
            self.lora_B = nn.Parameter(conv.weight.new_zeros(conv.out_channels, self.r))
            self.scaling = self.lora_alpha / self.r
            conv.weight.requires_grad = False
            if getattr(conv, "bias", None) is not None:
                conv.bias.requires_grad = False
            self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def _delta(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A).view_as(self.conv.weight) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.conv(x)
        if self.r == 0 or self.merged:
            return base

        xd = self.lora_dropout(x)
        pad_mod = getattr(self.conv, "static_padding", None) or getattr(
            self.conv, "_static_padding", None
        )
        if pad_mod is not None:
            # EfficientNet's Conv2dStaticSamePadding pads before convolving; the
            # LoRA branch has to see exactly the same padded input.
            xd = pad_mod(xd)

        return base + F.conv2d(
            xd,
            self._delta(),
            None,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )


def _wrap(conv: nn.Module, **kwargs) -> nn.Module:
    if isinstance(conv, ConvLoRA) or hasattr(conv, "lora_A"):
        return conv
    return ConvLoRA(conv, **kwargs)


def inject_lora(
    model: nn.Module,
    r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    merge_weights: bool = False,
    target_parts: Sequence[str] = ("encoder", "decoder", "segmentation_head"),
) -> nn.Module:
    """Wrap the convolutions of ``model`` in LoRA adapters, in place.

    Raises:
        RuntimeError: if no layer was wrapped. A LoRA run where nothing was
            adapted trains zero parameters and looks like a very bad model
            rather than like a bug, so it fails loudly instead.
    """
    kwargs = dict(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                  merge_weights=merge_weights)
    wrapped = 0

    # EfficientNet encoders keep their convs behind `.conv` attributes on
    # composite modules, so they need naming-aware handling.
    encoder = getattr(model, "encoder", None)
    if encoder is not None and "efficientnet" in type(encoder).__name__.lower():
        for attr in ("_conv_stem", "_conv_head"):
            sub = getattr(encoder, attr, None)
            if sub is not None and hasattr(sub, "conv"):
                sub.conv = _wrap(sub.conv, **kwargs)
                wrapped += 1
        for blk in getattr(encoder, "_blocks", []):
            for attr in ("_expand_conv", "_depthwise_conv", "_project_conv",
                         "_se_reduce", "_se_expand"):
                sub = getattr(blk, attr, None)
                if sub is not None and hasattr(sub, "conv"):
                    sub.conv = _wrap(sub.conv, **kwargs)
                    wrapped += 1

    # Everything else: a generic sweep over plain Conv2d children.
    def sweep(module: nn.Module) -> int:
        count = 0
        for name, child in list(module.named_children()):
            if isinstance(child, ConvLoRA):
                continue
            if isinstance(child, nn.Conv2d) and not isinstance(child, Conv2dStaticSamePadding):
                setattr(module, name, _wrap(child, **kwargs))
                count += 1
            else:
                count += sweep(child)
        return count

    for part in target_parts:
        sub = getattr(model, part, None)
        if sub is not None and not (part == "encoder" and wrapped):
            wrapped += sweep(sub)

    if wrapped == 0:
        raise RuntimeError(
            "inject_lora wrapped no layers. The model architecture is not one "
            "this adapter understands; fine-tuning would train zero parameters."
        )
    print(f"[lora] wrapped {wrapped} convolution(s) with rank-{r} adapters")
    return model


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> int:
    """Freeze everything except LoRA parameters. Returns trainable count."""
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    if bias == "all":
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True
    elif bias == "lora_only":
        for name, param in model.named_parameters():
            if "lora_" in name and "bias" in name:
                param.requires_grad = True
    elif bias != "none":
        raise ValueError(f"bias must be 'none', 'all' or 'lora_only', got {bias!r}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total else 0.0
    print(f"[lora] trainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)")
    return trainable


def lora_state_dict(model: nn.Module) -> dict:
    """Just the LoRA tensors, for saving an adapter without the base weights."""
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "ConvLoRA",
    "inject_lora",
    "mark_only_lora_as_trainable",
    "lora_state_dict",
    "count_trainable",
]
