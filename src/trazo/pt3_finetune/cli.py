#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Step 3 CLI: fine-tune a pretrained checkpoint on a new region.

Subcommands:
  run       Fine-tune with a strategy (full, lastlayer, lora, upgd)
  merge     Merge fine-tuned checkpoints back toward the base (MagMax)
  configs   List the bundled starter configs and print their paths

Backwards compatible with the earlier placeholder: calling this with only
``--config`` / ``--checkpoint`` / ``--data-dir`` and no subcommand runs ``run``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

CONFIG_DIR = Path(__file__).resolve().parent / "configs"

STRATEGY_CONFIGS = {
    "full": "finetune_full.yaml",
    "lastlayer": "finetune_lastlayer.yaml",
    "lora": "finetune_lora.yaml",
    "upgd": "finetune_upgd.yaml",
}


def default_config_for(strategy: str) -> Path:
    return CONFIG_DIR / STRATEGY_CONFIGS[strategy]


def build_finetune_args(
    config: str,
    strategy: str = "full",
    checkpoint: Optional[str] = None,
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_epochs: Optional[int] = None,
    lr: Optional[float] = None,
    extra: Optional[List[str]] = None,
) -> List[str]:
    """Assemble the argv handed to LightningCLI.

    Every flag this CLI accepts becomes an override here, so nothing the user
    types is quietly dropped.
    """
    args = ["fit", f"--config={config}", f"--model.init_args.strategy={strategy}"]
    if checkpoint:
        args.append(f"--model.init_args.pretrained_ckpt={checkpoint}")
    if data_dir:
        args.append(f"--data.init_args.root={data_dir}")
    if output_dir:
        args.append(f"--trainer.default_root_dir={output_dir}")
    if max_epochs is not None:
        args.append(f"--trainer.max_epochs={max_epochs}")
    if lr is not None:
        args.append(f"--model.init_args.lr={lr}")
    args.extend(extra or [])
    return args


def run_finetune(args: argparse.Namespace) -> int:
    from lightning.pytorch.cli import LightningCLI
    from torchgeo.trainers import BaseTask

    config = args.config or str(default_config_for(args.strategy))
    if not Path(config).is_file():
        print(f"error: config not found: {config}", file=sys.stderr)
        return 1

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    cli_args = build_finetune_args(
        config,
        strategy=args.strategy,
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_epochs=args.max_epochs,
        lr=args.lr,
        extra=list(args.extra or []),
    )

    print(f"[pt3_finetune] strategy: {args.strategy}")
    print(f"[pt3_finetune] config:   {config}")
    print(f"[pt3_finetune] args:     {cli_args}")

    # Same GDAL settings Step 4 uses for remote reads.
    os.environ.update(
        {
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "AWS_NO_SIGN_REQUEST": "YES",
            "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": "200000000",
            "GDAL_SWATH_SIZE": "200000000",
            "VSI_CURL_CACHE_SIZE": "200000000",
        }
    )

    LightningCLI(
        model_class=BaseTask,
        seed_everything_default=0,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        args=cli_args,
    )
    print("[pt3_finetune] finished")
    return 0


def run_merge(args: argparse.Namespace) -> int:
    from .merge import merge_checkpoints, save_state_dict

    state = merge_checkpoints(
        args.base, args.finetuned, method=args.method, scaling_coef=args.scaling_coef
    )
    save_state_dict(state, args.output)
    return 0


def list_configs(_args: argparse.Namespace) -> int:
    print(f"Bundled Step 3 configs in {CONFIG_DIR}:\n")
    for strategy, name in STRATEGY_CONFIGS.items():
        path = CONFIG_DIR / name
        mark = " " if path.is_file() else "!"
        print(f" {mark} {strategy:<10} {path}")
    print(
        "\nCopy one, edit the data paths and class weights, then:\n"
        "  trazo-pt3-finetune run --strategy <name> --config my_config.yaml \\\n"
        "      --checkpoint ftw_3class.ckpt --data-dir /data/region "
        "--output-dir /models/ft"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trazo-pt3-finetune",
        description=(
            "Fine-tune a pretrained field boundary checkpoint on a new region. "
            "Strategies: full, lastlayer, lora, upgd. Merge finished "
            "checkpoints with MagMax to limit catastrophic forgetting."
        ),
    )
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Fine-tune a checkpoint.")
    run.add_argument(
        "--strategy", choices=list(STRATEGY_CONFIGS), default="full",
        help=(
            "full: update all weights (strongest on the target region). "
            "lastlayer: segmentation head only. lora: rank-r adapters. "
            "upgd: all weights, utility-gated optimizer. Default: full."
        ),
    )
    run.add_argument(
        "--config", default=None,
        help="YAML config. Defaults to the bundled config for the strategy.",
    )
    run.add_argument(
        "--checkpoint", default=None,
        help="Pretrained checkpoint to adapt (e.g. the FTW 3-class model).",
    )
    run.add_argument("--data-dir", default=None, help="Dataset root. Overrides the config.")
    run.add_argument(
        "--output-dir", default=None,
        help="Where logs and checkpoints go. Overrides the config.",
    )
    run.add_argument("--max-epochs", type=int, default=None, help="Override trainer.max_epochs.")
    run.add_argument("--lr", type=float, default=None, help="Override the learning rate.")
    run.add_argument(
        "extra", nargs="*",
        help="Extra LightningCLI overrides, e.g. --data.init_args.batch_size=16",
    )
    run.set_defaults(func=run_finetune)

    merge = sub.add_parser("merge", help="Merge checkpoints (MagMax or task-vector sum).")
    merge.add_argument("--base", required=True, help="Base checkpoint.")
    merge.add_argument("--finetuned", required=True, nargs="+", help="Fine-tuned checkpoints.")
    merge.add_argument("--output", required=True, help="Output checkpoint path.")
    merge.add_argument("--method", choices=["magmax", "sum"], default="magmax")
    merge.add_argument("--scaling-coef", type=float, default=1.0)
    merge.set_defaults(func=run_merge)

    configs = sub.add_parser("configs", help="List the bundled starter configs.")
    configs.set_defaults(func=list_configs)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Backwards compatibility with the flags the placeholder CLI accepted.
    if argv and argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        argv = ["run"] + argv

    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
