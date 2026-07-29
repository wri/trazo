"""Fine-tuning task: the Step 4 trainer plus a choice of adaptation strategy.

Four strategies, matching the technical note's fine-tuning comparison:

==============  ==========================================================
``full``        Update every weight. Strongest on the target region in the
                note's Chiquitania experiment (84.4% pixel IoU), and the
                most prone to catastrophic forgetting elsewhere.
``lastlayer``   Freeze encoder and decoder; update the segmentation head
                only. Cheapest, and underfits.
``lora``        Freeze the base weights; train rank-``r`` adapters.
``upgd``        Update every weight with utility-based perturbed gradient
                descent, which protects high-utility weights.
==============  ==========================================================

Model merging (MagMax, task arithmetic) is a post-hoc operation on finished
checkpoints and lives in :mod:`trazo.pt3_finetune.merge`.

Two behaviours here are deliberate corrections of the experiment code this is
derived from:

* The pretrained checkpoint is loaded **before** LoRA injection or freezing, and
  a load that matches no keys is a hard error. The original injected LoRA first
  and loaded with ``strict=False``, which happened to work only because the
  wrapper aliased the wrapped conv's parameters onto itself; without that
  accident the pretrained weights would have been dropped in silence. Loading
  first removes the dependency on the accident, and the hard error means a
  genuinely mismatched checkpoint can never be mistaken for a bad model.
* The optimizer is selected by the ``optimizer`` hparam. In the original,
  ``configure_optimizers`` always constructed UPGD with the AdamW line commented
  out, so the full / last-layer / LoRA arms of the comparison were all secretly
  UPGD runs too.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..pt4_train.trainers import CustomSemanticSegmentationTask
from .lora import inject_lora, mark_only_lora_as_trainable
from .optimizers import UPGD

STRATEGIES = ("full", "lastlayer", "lora", "upgd")


class FineTuneTask(CustomSemanticSegmentationTask):
    """Semantic segmentation task with fine-tuning strategies."""

    def __init__(
        self,
        *args: Any,
        pretrained_ckpt: Optional[str] = None,
        strategy: str = "full",
        optimizer: Optional[str] = None,
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        upgd_weight_decay: float = 1e-3,
        upgd_beta_utility: float = 0.999,
        upgd_sigma: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            pretrained_ckpt: Checkpoint to adapt (e.g. the FTW 3-class model).
            strategy: One of ``full``, ``lastlayer``, ``lora``, ``upgd``.
            optimizer: ``adamw`` or ``upgd``. Defaults to ``upgd`` for the
                ``upgd`` strategy and ``adamw`` otherwise.
            lora_rank: Rank of the LoRA adapters.
            lora_alpha: LoRA scaling; the update is scaled by alpha / rank.
            lora_dropout: Dropout applied to the LoRA branch input.
            upgd_weight_decay: Decoupled weight decay for UPGD.
            upgd_beta_utility: EMA decay of the UPGD utility trace.
            upgd_sigma: Perturbation noise scale for UPGD.
        """
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
        if optimizer is None:
            optimizer = "upgd" if strategy == "upgd" else "adamw"
        if optimizer not in ("adamw", "upgd"):
            raise ValueError(f"optimizer must be 'adamw' or 'upgd', got {optimizer!r}")

        super().__init__(*args, **kwargs)
        self.save_hyperparameters(
            {
                "pretrained_ckpt": pretrained_ckpt,
                "strategy": strategy,
                "optimizer": optimizer,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "upgd_weight_decay": upgd_weight_decay,
                "upgd_beta_utility": upgd_beta_utility,
                "upgd_sigma": upgd_sigma,
            }
        )
        self._apply_strategy()

    # ------------------------------------------------------------------
    # checkpoint loading
    # ------------------------------------------------------------------
    def load_pretrained(self, ckpt_path: str) -> None:
        """Load a pretrained checkpoint into the un-adapted model.

        Raises:
            RuntimeError: if the checkpoint matches none of the model's
                parameters. Silently training from scratch while reporting a
                fine-tune is the single most expensive failure mode here.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        state = {k: v for k, v in state.items() if "lora_" not in k}

        own = self.state_dict()
        matched = {
            k: v for k, v in state.items()
            if k in own and own[k].shape == v.shape
        }
        if not matched:
            raise RuntimeError(
                f"No parameters in {ckpt_path} matched this model. Check that "
                "in_channels, num_classes, model and backbone agree with the "
                "checkpoint you are adapting."
            )

        result = self.load_state_dict(matched, strict=False)
        skipped = len(state) - len(matched)
        print(
            f"[finetune] loaded {len(matched)}/{len(own)} tensors from {ckpt_path}"
            + (f"; {skipped} checkpoint tensor(s) did not match" if skipped else "")
        )
        if result.unexpected_keys:
            print(f"[finetune] unexpected keys ignored: {len(result.unexpected_keys)}")

    # ------------------------------------------------------------------
    # strategy
    # ------------------------------------------------------------------
    def _apply_strategy(self) -> None:
        strategy = self.hparams["strategy"]
        ckpt_path = self.hparams["pretrained_ckpt"]

        # Order matters: weights first, adaptation second.
        if ckpt_path:
            self.load_pretrained(ckpt_path)
        elif strategy != "full":
            print(
                f"[finetune] WARNING: strategy '{strategy}' without "
                "pretrained_ckpt adapts randomly initialized weights."
            )

        if strategy == "lastlayer":
            head = getattr(self.model, "segmentation_head", None)
            if head is None:
                raise RuntimeError(
                    "strategy 'lastlayer' needs a model with a segmentation_head "
                    f"(got {type(self.model).__name__})."
                )
            for param in self.model.parameters():
                param.requires_grad = False
            for param in head.parameters():
                param.requires_grad = True
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"[finetune] last-layer: {trainable:,} trainable parameters")

        elif strategy == "lora":
            inject_lora(
                self.model,
                r=self.hparams["lora_rank"],
                lora_alpha=self.hparams["lora_alpha"],
                lora_dropout=self.hparams["lora_dropout"],
            )
            mark_only_lora_as_trainable(self.model, bias="none")

        # 'full' and 'upgd' train everything; they differ only in the optimizer.

    # ------------------------------------------------------------------
    # optimization
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError(
                "No trainable parameters. Check the fine-tuning strategy: "
                f"'{self.hparams['strategy']}' froze the whole model."
            )

        name = self.hparams["optimizer"]
        if name == "upgd":
            optimizer = UPGD(
                params,
                lr=self.hparams["lr"],
                weight_decay=self.hparams["upgd_weight_decay"],
                beta_utility=self.hparams["upgd_beta_utility"],
                sigma=self.hparams["upgd_sigma"],
            )
        else:
            optimizer = AdamW(params, lr=self.hparams["lr"], amsgrad=True)

        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.hparams["patience"], eta_min=1e-6
        )
        print(
            f"[finetune] strategy={self.hparams['strategy']} optimizer={name} "
            f"trainable_tensors={len(params)}"
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor},
        }


__all__ = ["FineTuneTask", "STRATEGIES"]
