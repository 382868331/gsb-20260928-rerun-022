"""Command line interface for the r3d resource service."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .inspect import export_scene, verify_closure
from .store import Library


def _print_json(value: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False))
    else:
        committed = "COMMITTED" if value.get("committed") else "REJECTED"
        print(f"batch: {value['batch']} -> {committed}")
        for scene_id in value.get("scenes", []):
            print(f"  scene {scene_id}")
        if value.get("newResources"):
            print(f"  new resources: {value['newResources']}")
        if value.get("reusedResources"):
            print(f"  reused resources: {value['reusedResources']}")
        for issue in value.get("issues", []):
            print(f"  ERROR {issue}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="r3d", description="3D resource import service")
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser("import", help="import one material batch")
    import_parser.add_argument("--library", required=True)
    import_parser.add_argument("--batch", required=True)
    import_parser.add_argument("--json", action="store_true")

    verify_parser = subparsers.add_parser("verify", help="verify scene dependency closure")
    verify_parser.add_argument("--library", required=True)
    verify_parser.add_argument("scene")

    export_parser = subparsers.add_parser("export", help="export an inspectable scene bundle")
    export_parser.add_argument("--library", required=True)
    export_parser.add_argument("--scene", required=True)
    export_parser.add_argument("--output", required=True)

    list_parser = subparsers.add_parser("list", help="list published scenes")
    list_parser.add_argument("--library", required=True)

    args = parser.parse_args(argv)
    library = Library(Path(args.library))

    if args.command == "import":
        report = library.import_batch(Path(args.batch))
        _print_json(report.to_dict(), args.json)
        return 0 if report.committed else 2
    if args.command == "verify":
        closure = verify_closure(library, args.scene)
        print(json.dumps({"ok": closure.ok, "resources": closure.resources, "issues": closure.issues}, indent=2))
        return 0 if closure.ok else 2
    if args.command == "export":
        try:
            bundle = export_scene(library, args.scene, Path(args.output))
        except KeyError as exc:
            print(f"export failed: {exc.args[0]}", file=sys.stderr)
            return 2
        print(f"exported {bundle['sceneId']} to {args.output}")
        return 0
    if args.command == "list":
        manifest = library.read_manifest()
        for entry in manifest.get("scenes", []):
            print(entry["id"])
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
