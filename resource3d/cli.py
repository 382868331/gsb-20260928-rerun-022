from __future__ import annotations

import argparse

from .errors import ImportErrorBundle
from .library import ResourceLibrary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="resource3d")
    subparsers = parser.add_subparsers(dest="command", required=True)
    import_parser = subparsers.add_parser("import")
    import_parser.add_argument("--library", required=True)
    import_parser.add_argument("--manifest", required=True)
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--library", required=True)
    export_parser.add_argument("--asset", required=True)
    export_parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    library = ResourceLibrary(args.library)
    try:
        if args.command == "import":
            print(library.import_batch(args.manifest))
        else:
            print(library.export_inspection(args.asset, args.output))
    except ImportErrorBundle as exc:
        for error in exc.errors:
            print(f"error: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
