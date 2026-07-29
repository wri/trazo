"""Model merging: task arithmetic and MagMax.

Fine-tuning on a small region buys target-region accuracy and pays for it in
catastrophic forgetting. In the technical note's cross-region evaluation, full
fine-tuning on Chiquitania samples dropped India pixel IoU from 96.0% to 0.1%.
Merging the fine-tuned weights back toward the base model recovers most of that:
MagMax scored best on Kenya (65.9%) and India (43.4%) while keeping 79.2% on the
target region, versus 84.4% for full fine-tuning.

Two operations:

**Task arithmetic** (Ilharco et al. 2022). A task vector is the weight delta
``theta_finetuned - theta_base``. Vectors add, negate and scale, so capabilities
can be combined or removed without retraining.

**MagMax** (Marczak et al. 2024). Given several task vectors, keep the
element-wise largest-magnitude entry across them, then apply that single merged
vector to the base weights.

The experiment code this is derived from called a ``src.task_vectors`` module
that was never committed, so the notebook could not be run. This is a standalone
implementation with the same interface.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch

StateDict = Dict[str, torch.Tensor]


def load_state_dict(path: str | Path) -> StateDict:
    """Load a state dict from a Lightning checkpoint or a bare .pth."""
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not contain a state dict")
    return {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}


def _mergeable_keys(base: StateDict, other: StateDict) -> List[str]:
    """Float tensors present in both with matching shapes.

    Integer buffers (``num_batches_tracked`` and friends) are excluded: taking
    differences of them is meaningless and silently corrupts BatchNorm state.
    """
    keys = []
    for k, v in base.items():
        if k not in other:
            continue
        if not torch.is_floating_point(v):
            continue
        if other[k].shape != v.shape:
            continue
        keys.append(k)
    return keys


class TaskVector:
    """The weight delta between a fine-tuned checkpoint and its base."""

    def __init__(
        self,
        base_path: Optional[str | Path] = None,
        finetuned_path: Optional[str | Path] = None,
        vector: Optional[StateDict] = None,
    ) -> None:
        if vector is not None:
            self.vector = vector
            return
        if base_path is None or finetuned_path is None:
            raise ValueError("provide either `vector` or both checkpoint paths")

        base = load_state_dict(base_path)
        finetuned = load_state_dict(finetuned_path)
        keys = _mergeable_keys(base, finetuned)
        if not keys:
            raise ValueError(
                f"no comparable float tensors between {base_path} and {finetuned_path}"
            )
        self.vector = {k: (finetuned[k] - base[k]).float() for k in keys}

    def __add__(self, other: "TaskVector | int") -> "TaskVector":
        if isinstance(other, int) and other == 0:  # supports sum([...])
            return self
        merged: StateDict = {}
        for k, v in self.vector.items():
            if k in other.vector and other.vector[k].shape == v.shape:
                merged[k] = v + other.vector[k]
            else:
                merged[k] = v.clone()
        for k, v in other.vector.items():
            merged.setdefault(k, v.clone())
        return TaskVector(vector=merged)

    __radd__ = __add__

    def __neg__(self) -> "TaskVector":
        return TaskVector(vector={k: -v for k, v in self.vector.items()})

    def __mul__(self, scalar: float) -> "TaskVector":
        return TaskVector(vector={k: v * float(scalar) for k, v in self.vector.items()})

    __rmul__ = __mul__

    def norm(self) -> float:
        return float(torch.sqrt(sum((v.double() ** 2).sum() for v in self.vector.values())))

    def apply_to(self, base_path: str | Path, scaling_coef: float = 1.0) -> StateDict:
        """Return ``base + scaling_coef * vector`` as a state dict."""
        base = load_state_dict(base_path)
        out = {k: v.clone() for k, v in base.items()}
        applied = 0
        for k, delta in self.vector.items():
            if k in out and out[k].shape == delta.shape:
                out[k] = out[k] + scaling_coef * delta.to(out[k].dtype)
                applied += 1
        if applied == 0:
            raise ValueError(f"task vector shares no keys with {base_path}")
        print(f"[merge] applied {applied} tensor deltas at scaling_coef={scaling_coef}")
        return out


def magmax(task_vectors: Sequence[TaskVector]) -> TaskVector:
    """Element-wise maximum-magnitude selection across task vectors."""
    if not task_vectors:
        raise ValueError("magmax needs at least one task vector")
    if len(task_vectors) == 1:
        return task_vectors[0]

    merged: StateDict = {}
    for tv in task_vectors:
        for k, v in tv.vector.items():
            current = merged.get(k)
            if current is None:
                merged[k] = v.clone()
            elif current.shape == v.shape:
                take = v.abs() > current.abs()
                merged[k] = torch.where(take, v, current)
    return TaskVector(vector=merged)


def merge_checkpoints(
    base: str | Path,
    finetuned: Iterable[str | Path],
    method: str = "magmax",
    scaling_coef: float = 1.0,
) -> StateDict:
    """Merge one or more fine-tuned checkpoints back into a base checkpoint."""
    vectors = [TaskVector(base, path) for path in finetuned]
    if method == "magmax":
        combined = magmax(vectors)
    elif method == "sum":
        combined = vectors[0]
        for tv in vectors[1:]:
            combined = combined + tv
    else:
        raise ValueError(f"method must be 'magmax' or 'sum', got {method!r}")
    return combined.apply_to(base, scaling_coef=scaling_coef)


def save_state_dict(state: StateDict, path: str | Path) -> None:
    """Write a state dict as a Lightning-loadable checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "pytorch-lightning_version": "2.0.0"}, path)
    print(f"[merge] wrote {path}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge fine-tuned checkpoints back toward a base checkpoint using "
            "MagMax or task-vector addition, to limit catastrophic forgetting."
        )
    )
    parser.add_argument("--base", required=True, help="Base checkpoint (e.g. the FTW model).")
    parser.add_argument(
        "--finetuned", required=True, nargs="+",
        help="One or more fine-tuned checkpoints to merge in.",
    )
    parser.add_argument("--output", required=True, help="Where to write the merged checkpoint.")
    parser.add_argument(
        "--method", choices=["magmax", "sum"], default="magmax",
        help="magmax keeps the largest-magnitude delta per weight (default).",
    )
    parser.add_argument(
        "--scaling-coef", type=float, default=1.0,
        help="Scale applied to the merged task vector (default: 1.0).",
    )
    args = parser.parse_args(argv)

    state = merge_checkpoints(
        args.base, args.finetuned, method=args.method, scaling_coef=args.scaling_coef
    )
    save_state_dict(state, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
