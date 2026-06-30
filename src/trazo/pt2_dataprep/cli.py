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
  export-zarr   Export FTW-style Zarr files from windows and masks
"""

import sys

_USAGE = """\
usage: trazo-pt2-dataprep {pair-stacks,resize-256,chips-bboxes,make-masks,chips-parquet,scale-u16,export-hkl,export-zarr} [args ...]

Step 2: Data preparation utilities (chips, masks, parquet).

Subcommands:
  pair-stacks   Pair window_a and window_b into 8-band stacks.
                Run: trazo-pt2-dataprep pair-stacks --help
  resize-256    Resize 8-band stacks to 256x256.
                Run: trazo-pt2-dataprep resize-256 --help
  chips-bboxes  Create 256x256 chip bounding box GeoJSONs.
                Run: trazo-pt2-dataprep chips-bboxes --help
  make-masks    Create window_a/b, instance and semantic masks.
                Run: trazo-pt2-dataprep make-masks --help
  chips-parquet Build per-chip GeoParquet metadata.
                Run: trazo-pt2-dataprep chips-parquet --help
  scale-u16     Scale chips to uint16 in [0, 10000].
                Run: trazo-pt2-dataprep scale-u16 --help
  export-hkl    Export FTW-style .hkl files from windows and masks.
                Run: trazo-pt2-dataprep export-hkl --help
  export-zarr   Export FTW-style Zarr files from windows and masks.
                Run: trazo-pt2-dataprep export-zarr --help
"""

_DISPATCH = {
    "pair-stacks":   lambda: _run("pair_stacks"),
    "resize-256":    lambda: _run("resize_chips_256"),
    "chips-bboxes":  lambda: _run("chips_to_bboxes"),
    "make-masks":    lambda: _run("make_masks_and_windows"),
    "chips-parquet": lambda: _run("build_chips_parquet"),
    "scale-u16":     lambda: _run("scale_uint16"),
    "export-hkl":    lambda: _run("export_hkl"),
    "export-zarr":   lambda: _run("export_zarr"),
}


def _run(module_name):
    import importlib, inspect
    mod = importlib.import_module("trazo.pt2_dataprep." + module_name)
    sig = inspect.signature(mod.main)
    if sig.parameters:
        mod.main(sys.argv[1:])
    else:
        mod.main()


def main(argv=None):
    if argv is not None:
        sys.argv = [sys.argv[0]] + list(argv)

    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(_USAGE)
        return

    cmd, rest = args[0], args[1:]

    if cmd not in _DISPATCH:
        print("error: unknown subcommand '" + cmd + "'\n")
        print(_USAGE)
        sys.exit(1)

    sys.argv = [sys.argv[0]] + rest
    _DISPATCH[cmd]()


if __name__ == "__main__":
    main()
