"""Content-addressed resource library with whole-batch atomic publishing."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .gltf import Issue, load_gltf
from .model import canonical_json

MANIFEST_NAME = "manifest.json"
RESOURCES_DIRNAME = "resources"
BLOBS_DIRNAME = "blobs"

GLTF_SUFFIXES = {".gltf", ".glb"}


@dataclass
class BatchReport:
    batch: str
    committed: bool = False
    scene_ids: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    new_resources: dict[str, int] = field(default_factory=dict)
    reused_resources: dict[str, int] = field(default_factory=dict)
    new_blobs: int = 0
    reused_blobs: int = 0
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "batch": self.batch,
            "committed": self.committed,
            "scenes": self.scene_ids,
            "files": self.files,
            "newResources": self.new_resources,
            "reusedResources": self.reused_resources,
            "newBlobs": self.new_blobs,
            "reusedBlobs": self.reused_blobs,
            "issues": self.issues,
        }


class Library:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.resources_dir = self.root / RESOURCES_DIRNAME
        self.blobs_dir = self.root / BLOBS_DIRNAME
        self.manifest_path = self.root / MANIFEST_NAME

    def initialize(self) -> None:
        self.resources_dir.mkdir(parents=True, exist_ok=True)
        self.blobs_dir.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            self._atomic_bytes(self.manifest_path, canonical_json({"version": 1, "scenes": []}))

    def read_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {"version": 1, "scenes": []}
        return json.loads(self.manifest_path.read_text("utf-8"))

    def resource_path(self, resource_id: str) -> Path:
        kind = resource_id.split("-", 1)[0]
        return self.resources_dir / kind / f"{resource_id}.json"

    def blob_path(self, blob_id: str) -> Path:
        return self.blobs_dir / f"{blob_id}.bin"

    def load_resource(self, resource_id: str) -> dict:
        return json.loads(self.resource_path(resource_id).read_text("utf-8"))

    def has_resource(self, resource_id: str) -> bool:
        return self.resource_path(resource_id).exists()

    @staticmethod
    def _atomic_bytes(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def import_batch(self, batch_dir: Path | str) -> BatchReport:
        """Validate every file first; publish resources only if the batch is clean."""
        batch_path = Path(batch_dir)
        report = BatchReport(batch=str(batch_path))
        if not batch_path.is_dir():
            report.issues.append(f"batch directory not found: {batch_path}")
            return report

        files = sorted(
            path for path in batch_path.rglob("*") if path.suffix.lower() in GLTF_SUFFIXES
        )
        if not files:
            report.issues.append("batch contains no .gltf or .glb files")
            return report

        combined_resources: dict[str, dict] = {}
        combined_blobs: dict[str, bytes] = {}
        for path in files:
            loaded = load_gltf(path, batch_path.resolve())
            report.files.append(str(path.relative_to(batch_path)))
            report.issues.extend(str(issue) for issue in loaded.issues)
            if loaded.ok:
                for resource_id, payload in loaded.resources.items():
                    existing = combined_resources.get(resource_id)
                    if existing is not None and existing != payload:
                        report.issues.append(
                            f"hash collision or inconsistent payload for {resource_id}"
                        )
                    combined_resources[resource_id] = payload
                combined_blobs.update(loaded.blobs)
                report.scene_ids.extend(loaded.roots)

        if report.issues:
            return report
        return self._commit(batch_path, combined_resources, combined_blobs, report)

    def _commit(self, batch_path, resources, blobs, report: BatchReport) -> BatchReport:
        self.initialize()
        # Stage all bytes first. Nothing is visible until each write succeeds.
        staged_resources: list[tuple[Path, bytes]] = []
        staged_blobs: list[tuple[Path, bytes]] = []
        for resource_id, payload in sorted(resources.items()):
            target = self.resource_path(resource_id)
            if target.exists():
                kind = payload["type"]
                report.reused_resources[kind] = report.reused_resources.get(kind, 0) + 1
            else:
                kind = payload["type"]
                report.new_resources[kind] = report.new_resources.get(kind, 0) + 1
                staged_resources.append((target, canonical_json(payload)))
        for blob_id, data in sorted(blobs.items()):
            target = self.blob_path(blob_id)
            if target.exists():
                report.reused_blobs += 1
            else:
                report.new_blobs += 1
                staged_blobs.append((target, data))

        try:
            for target, data in staged_blobs + staged_resources:
                self._atomic_bytes(target, data)
            manifest = self.read_manifest()
            known = {entry["id"] for entry in manifest.get("scenes", [])}
            for scene_id in report.scene_ids:
                if scene_id not in known:
                    manifest.setdefault("scenes", []).append(
                        {"id": scene_id, "sourceBatch": batch_path.name}
                    )
                    known.add(scene_id)
            self._atomic_bytes(self.manifest_path, canonical_json(manifest))
        except OSError as exc:
            report.issues.append(f"publish failed: {exc}")
            return report
        report.committed = True
        return report
