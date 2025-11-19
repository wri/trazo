#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 5 CLI: Inference and post processing.

This CLI currently exposes:

  1. tilepairs
     Select best Sentinel-2 scene pairs per tile for a given AOI and year,
     then write 8-band stacks and a summary CSV.

  2. batch-infer (placeholder)
     Future hook for running model inference over precomputed stacks.
"""

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tkt-pt5-infer",
        description="Step 5: Inference and post processing utilities.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Subcommands for Step 5.",
    )

    # ------------------------------------------------------------------
    # tilepairs: delegate to tkt.pt5_inference.tilepairs
    # ------------------------------------------------------------------
    p_pairs = subparsers.add_parser(
        "tilepairs",
        help=(
            "Select best Sentinel-2 scene pairs per tile for an AOI and year, "
            "and write 8-band stacks plus a summary CSV."
        ),
    )
    # We do not duplicate all options here. Instead, we forward remaining
    # arguments directly to tkt.pt5_inference.tilepairs.main(argv).
    # So the user can pass all the same flags as if they ran the module.
    p_pairs.set_defaults(command="tilepairs")

    # ------------------------------------------------------------------
    # batch-infer: placeholder for future FTW model inference
    # ------------------------------------------------------------------
    p_infer = subparsers.add_parser(
        "batch-infer",
        help="Batch inference over precomputed stacks (placeholder for now).",
    )
    p_infer.add_argument(
        "--config",
        default=None,
        help="Optional config file for batch inference.",
    )
    p_infer.add_argument(
        "--checkpoint",
        default=None,
        help="Model checkpoint or directory of checkpoints.",
    )
    p_infer.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing input stack GeoTIFFs.",
    )
    p_infer.add_argument(
        "--output-dir",
        default=None,
        help="Directory where inference outputs will be written.",
    )
    p_infer.set_defaults(command="batch-infer")

    return parser


def main(argv=None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    parser = build_parser()

    # We allow subcommands to own their arguments. So we parse known args here
    # and forward the remaining args to the underlying implementation.
    args, remaining = parser.parse_known_args(argv)

    if not args.command:
        parser.print_help()
        return

    if args.command == "tilepairs":
        # Delegate to tkt.pt5_inference.tilepairs.main(remaining_args)
        from . import tilepairs

        print("[pt5_inference] Running tilepairs selector/stacker...")
        tilepairs.main(remaining)
        return

    if args.command == "batch-infer":
        # Placeholder behavior for now
        print("[pt5_inference] batch-infer placeholder.")
        print("config    :", args.config)
        print("checkpoint:", args.checkpoint)
        print("input-dir :", args.input_dir)
        print("output-dir:", args.output_dir)
        return

    # Fallback
    parser.print_help()


if __name__ == "__main__":
    main()
