#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 5 CLI: Inference and post processing.

Subcommands:

  tilepairs
      AOI based selector and stack writer.
      Delegates to trazo.pt5_inference.tilepairs.main(argv).

  tilepairs-advanced
      Advanced selector that also handles inference and cleanup.
      Delegates to trazo.pt5_inference.tilepairs_advanced.main(argv).

  tilepairs-tilelist
      Tile list based selector using SOS/EOS and progressive cloud constraints.
      Delegates to trazo.pt5_inference.tilepairs_tilelist.main(argv).

  multi-infer
      Multi model batch inference over a folder of TIFFs.
      Delegates to trazo.pt5_inference.multimodelinference.main(argv).

  batch-infer
      Batch inference utility (single or multiple checkpoints).
      Delegates to trazo.pt5_inference.batchinference.main(argv).
"""

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trazo-pt5-infer",
        description="Step 5: Inference and post processing utilities.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Subcommands for Step 5.",
    )

    # tilepairs (AOI based stacks, selection + 8 band stacks only)
    p_pairs = subparsers.add_parser(
        "tilepairs",
        help=(
            "AOI based Sentinel-2 scene pairing and 8 band stack creation "
            "(delegates to trazo.pt5_inference.tilepairs)."
        ),
    )
    p_pairs.set_defaults(command="tilepairs")

    # tilepairs-advanced (selection + inference + cleanup)
    p_pairs_adv = subparsers.add_parser(
        "tilepairs-advanced",
        help=(
            "Advanced Sentinel-2 scene pairing that can also run inference and "
            "cleanup temporary stacks "
            "(delegates to trazo.pt5_inference.tilepairs_advanced)."
        ),
    )
    p_pairs_adv.set_defaults(command="tilepairs-advanced")

    # tilepairs-tilelist (tile list + SOS/EOS)
    p_pairs_tile = subparsers.add_parser(
        "tilepairs-tilelist",
        help=(
            "Tile list based Sentinel-2 scene pairing using SOS/EOS derived "
            "date windows "
            "(delegates to trazo.pt5_inference.tilepairs_tilelist)."
        ),
    )
    p_pairs_tile.set_defaults(command="tilepairs-tilelist")

    # multi-infer (multi model inference over TIFFs)
    p_multi = subparsers.add_parser(
        "multi-infer",
        help=(
            "Multi model batch inference over a directory of TIFFs "
            "(delegates to trazo.pt5_inference.multimodelinference)."
        ),
    )
    p_multi.set_defaults(command="multi-infer")

    # batch-infer (general batch inference utility)
    p_batch = subparsers.add_parser(
        "batch-infer",
        help=(
            "Batch inference utility "
            "(delegates to trazo.pt5_inference.batchinference)."
        ),
    )
    p_batch.set_defaults(command="batch-infer")

    return parser


def main(argv=None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)

    if not args.command:
        parser.print_help()
        return

    if args.command == "tilepairs":
        from . import tilepairs

        print("[pt5_inference] Running AOI based tilepairs selector and stacker...")
        tilepairs.main(remaining)
        return

    if args.command == "tilepairs-advanced":
        from . import tilepairs_advanced

        print("[pt5_inference] Running advanced tilepairs selector and inference...")
        tilepairs_advanced.main(remaining)
        return

    if args.command == "tilepairs-tilelist":
        from . import tilepairs_tilelist

        print("[pt5_inference] Running tile list based tilepairs selector...")
        tilepairs_tilelist.main(remaining)
        return

    if args.command == "multi-infer":
        from . import multimodelinference

        print("[pt5_inference] Running multi model inference over stacks...")
        multimodelinference.main(remaining)
        return

    if args.command == "batch-infer":
        from . import batchinference

        print("[pt5_inference] Running batch inference utility...")
        batchinference.main(remaining)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
