#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 2 CLI: Data preparation utilities (chips, masks, parquet).

Subcommands:
  pair-stacks   Pair window_a and window_b into 8 band stacks
  resize-256    Resize 8 band stacks to 256x256 into sized256
  chips-bboxes  Create 256x256 chip bounding box GeoJSONs
  make-masks    Create window_a/b and instance and semantic masks
  chips-parquet Build per chip GeoParquet metadata
  scale-u16     Scale chips to uint16 in [0,10000]
  export-hkl    Export FTW-style .hkl files from windows and masks
"""

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tkt-pt2-dataprep",
        description="Step 2: Data preparation utilities (chips, masks, parquet).",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommands for Step 2.")

    subparsers.add_parser("pair-stacks", help="Pair window_a and window_b into 8 band stacks.")
    subparsers.add_parser("resize-256", help="Resize 8 band stacks to 256x256 into sized256.")
    subparsers.add_parser("chips-bboxes", help="Create 256x256 chip bounding box GeoJSONs.")
    subparsers.add_parser("make-masks", help="Create window_a/b and instance and semantic masks.")
    subparsers.add_parser("chips-parquet", help="Build per chip GeoParquet metadata.")
    subparsers.add_parser("scale-u16", help="Scale chips to uint16 in [0,10000].")
    subparsers.add_parser("export-hkl", help="Export FTW-style .hkl files from windows and masks.")

    return parser


def main(argv=None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)

    if not args.command:
        parser.print_help()
        return

    if args.command == "pair-stacks":
        from . import pair_stacks as mod
        mod.main(remaining)
        return

    if args.command == "resize-256":
        from . import resize_chips_256 as mod
        mod.main(remaining)
        return

    if args.command == "chips-bboxes":
        from . import chips_to_bboxes as mod
        mod.main(remaining)
        return

    if args.command == "make-masks":
        from . import make_masks_and_windows as mod
        mod.main(remaining)
        return

    if args.command == "chips-parquet":
        from . import build_chips_parquet as mod
        mod.main(remaining)
        return

    if args.command == "scale-u16":
        from . import scale_uint16 as mod
        mod.main(remaining)
        return

    if args.command == "export-hkl":
        from . import export_hkl as mod
        mod.main(remaining)
        return


if __name__ == "__main__":
    main()

