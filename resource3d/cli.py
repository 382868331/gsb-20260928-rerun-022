from __future__ import annotations

import argparse
import json

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
    diff_parser = subparsers.add_parser("diff")
    diff_parser.add_argument("--library", required=True)
    diff_parser.add_argument("--old-asset", required=True)
    diff_parser.add_argument("--new-asset", required=True)
    diff_parser.add_argument("--output", required=True)
    retarget_parser = subparsers.add_parser("retarget")
    retarget_parser.add_argument("--library", required=True)
    retarget_parser.add_argument("--asset", required=True)
    retarget_parser.add_argument("--plan", required=True)
    retarget_parser.add_argument("--review")
    retarget_parser.add_argument("--commit", action="store_true")
    retarget_parser.add_argument("--output")
    review_parser = subparsers.add_parser("tile-review")
    review_parser.add_argument("--library", required=True)
    review_parser.add_argument("--asset", required=True)
    review_parser.add_argument("--plan", required=True)
    review_parser.add_argument("--review", required=True)
    review_parser.add_argument("--decisions")
    review_parser.add_argument("--save-as")
    review_parser.add_argument("--output")
    rollback_parser = subparsers.add_parser("rollback")
    rollback_parser.add_argument("--library", required=True)
    rollback_parser.add_argument("--asset", required=True)
    args = parser.parse_args(argv)
    library = ResourceLibrary(args.library)
    try:
        if args.command == "import":
            print(library.import_batch(args.manifest))
        elif args.command == "export":
            print(library.export_inspection(args.asset, args.output))
        elif args.command == "retarget":
            if args.review:
                report, published = library.apply_reviewed_tile_plan(
                    args.asset, args.plan, args.review, commit=args.commit,
                )
                if not report.get("valid", True):
                    for error in report.get("errors", []):
                        print(f"error: {error}")
                    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
                    if args.output:
                        with open(args.output, "w", encoding="utf-8") as handle:
                            handle.write(payload)
                    return 2
            elif args.commit:
                _, report = library.apply_retarget(args.asset, args.plan)
            else:
                report = library.plan_retarget(args.asset, args.plan)
            payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            if args.output:
                with open(args.output, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                print(args.output)
            else:
                print(payload, end="")
        elif args.command == "tile-review":
            decisions = None
            if args.decisions:
                with open(args.decisions, "r", encoding="utf-8") as handle:
                    decisions = json.load(handle)
            report = library.save_tile_review(
                args.asset, args.plan, args.review, decisions=decisions, save_as=args.save_as,
            )
            payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            if args.output:
                with open(args.output, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                print(args.output)
            else:
                print(payload, end="")
        elif args.command == "rollback":
            print(library.retarget_source(args.asset))
        else:
            report = library.compare_assets(args.old_asset, args.new_asset)
            with open(args.output, "w", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            print(args.output)
    except ImportErrorBundle as exc:
        for error in exc.errors:
            print(f"error: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
