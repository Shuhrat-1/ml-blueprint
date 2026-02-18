import argparse

from . import __version__


def main() -> None:
    parser = argparse.ArgumentParser(prog="mlb", description="ML Blueprint CLI")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    args = parser.parse_args()

    if args.version:
        print(__version__)
    else:
        parser.print_help()