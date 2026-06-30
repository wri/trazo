import sys

from .gridding import main as gridding_main

_USAGE = """\
usage: trazo-pt1-create {grid,chips} [args ...]

Step 1: Create training data.

Subcommands:
  grid   Create a training grid from field boundary datasets.
         Run: trazo-pt1-create grid --help
  chips  Download Sentinel-2 chips for planting/harvest windows.
         Run: trazo-pt1-create chips --help
"""


def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(_USAGE)
        return

    cmd, rest = args[0], args[1:]

    if cmd == "grid":
        # Replace argv so gridding's argparse sees its own args
        sys.argv = [sys.argv[0]] + rest
        gridding_main()

    elif cmd == "chips":
        # Lazy import: only requires pt1 optional extras when actually used
        from . import plantingharvest
        sys.argv = [sys.argv[0]] + rest
        plantingharvest.main()

    else:
        print(f"error: unknown subcommand '{cmd}'\n")
        print(_USAGE)
        sys.exit(1)
