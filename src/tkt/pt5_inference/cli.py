#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 5 CLI: Inference and post processing.

This CLI exposes three subcommands:

  1. tilepairs
     Select best Sentinel-2 scene pairs per tile for a given AOI and year,
     then write 8-band stacks and a summary CSV.
     (Delegates to: tkt.pt5_inference.tilepairs)

  2. multi-infer
     Run multi-model batch inference over precomputed 8-band stacks, where
     a directory of checkpoints is applied to a directory of TIFF stacks.
     (Delegates to: tkt.pt5_inference.multimodelinference)

  3. batch-infer-legacy
     Legacy batch inference hook for older scripts.
     (Delegates to: tkt.pt5_inference.batchinference, if it defines a main())
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
    # Do not duplicate all options here; forward remaining args to tilepairs.main()
    p_pairs.set_defaults(command="tilepairs")

    # ------------------------------------------------------------------
    # multi-infer: delegate to tkt.pt5_inference.multimodelinference
    # ------------------------------------------------------------------
    p_multi = subparsers.add_parser(
        "multi-infer",
        help=(
            "Run multi-model batch inference over precomputed stack TIFFs, "
            "using a directory of checkpoints."
        ),
    )
    # Same pattern: let multimodelinference.main parse its own arguments
    p_multi.set_defaults(command="multi-infer")

    # ------------------------------------------------------------------
    # batch-infer-legacy: delegate to tkt.pt5_inference.batchinference
    # ------------------------------------------------------------------
    p_legacy = subparsers.add_parser(
        "batch-infer-legacy",
        help="Legacy batch inference helper (older behavior).",
    )
    # Any arguments are forwarded; legacy module may or may not use them
    p_legacy.set_defaults(command="batch-infer-legacy")

    return parser


def main(argv=None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    parser = build_parser()

    # Parse only known args here; pass through the rest to the underlying modules.
    args, remaining = parser.parse_known_args(argv)

    if not args.command:
        parser.print_help()
        return

    if args.command == "tilepairs":
        from . import tilepairs

        print("[pt5_inference] Running tilepairs selector/stacker...")
        tilepairs.main(remaining)
        return

    if args.command == "multi-infer":
        from . import multimodelinference

        print("[pt5_inference] Running multi-model batch inference...")
        multimodelinference.main(remaining)
        return

    if args.command == "batch-infer-legacy":
        from . import batchinference

        print("[pt5_inference] Running legacy batch inference...")
        # Be defensive in case batchinference has no main()
        if hasattr(batchinference, "main"):
            batchinference.main(remaining)
        else:
            print(
                "[WARN] tkt.pt5_inference.batchinference has no main() function. "
                "Update it or remove this subcommand."
            )
        return

    # Fallback: should not get here, but just in case.
    parser.print_help()


if __name__ == "__main__":
    main()
