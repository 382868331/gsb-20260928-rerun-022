"""Verify dependency closure and export a human/machine-checkable scene graph."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .model import canonical_json
from .store import Library
from .transform import IDENTITY, multiply, transform_point


@dataclass
class ClosureReport:
    ok: bool = True
    resources: dict[str, int] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


def verify_closure(library: Library, scene_id: str) -> ClosureReport:
    """Walk a scene and ensure every declared dependency is present on disk."""
    report = ClosureReport()
    visited: set[str] = set()

    def note(kind: str) -> None:
        report.resources[kind] = report.resources.get(kind, 0) + 1

    def require(rid: str, referrer: str) -> dict | None:
        if rid in visited:
            return library.load_resource(rid) if library.has_resource(rid) else None
        if not library.has_resource(rid):
            report.ok = False
            report.issues.append(f"{referrer} references missing resource {rid}")
            return None
        visited.add(rid)
        return library.load_resource(rid)

    def visit_node(node_id: str, referrer: str, world_stack: list[list[float]]):
        payload = require(node_id, referrer)
        if payload is None:
            return
        note("node")
        world = multiply(world_stack[-1], payload["matrix"]) if world_stack else list(IDENTITY)
        for binding in payload.get("meshes", []):
            geometry = require(binding["geometry"], node_id)
            material = require(binding["material"], node_id)
            if geometry is not None:
                note("geometry")
                if geometry.get("triangleCount", 0) == 0:
                    report.ok = False
                    report.issues.append(f"geometry {binding['geometry']} has no triangles")
            if material is not None:
                note("material")
                image_id = material.get("pbr", {}).get("baseColorImage")
                if image_id is not None:
                    image = require(image_id, binding["material"])
                    if image is not None:
                        note("image")
                        blob_path = library.blob_path(image["blobId"])
                        if not blob_path.is_file():
                            report.ok = False
                            report.issues.append(f"missing blob {image['blobId']} for {image_id}")
                        elif blob_path.stat().st_size != image["byteLength"]:
                            report.ok = False
                            report.issues.append(f"blob size mismatch for {image_id}")
        for child in payload.get("children", []):
            visit_node(child, node_id, world_stack + [world])

    scene = require(scene_id, "scene query")
    if scene is None:
        return report
    note("scene")
    for root in scene.get("roots", []):
        visit_node(root, scene_id, [])
    return report


def export_scene(library: Library, scene_id: str, output_path: Path | str) -> dict:
    """Emit a single JSON document containing full closure and checksums."""
    closure = verify_closure(library, scene_id)
    if not closure.ok:
        raise KeyError("; ".join(closure.issues))
    scene = library.load_resource(scene_id)
    exported: dict[str, dict] = {}
    geometry_stats: dict[str, dict] = {}
    world_matrices: dict[str, list[float]] = {}
    world_points: dict[str, list[list[float]]] = {}
    ordered_nodes: list[str] = []

    def pull(rid: str) -> dict:
        if rid not in exported:
            exported[rid] = library.load_resource(rid)
        return exported[rid]

    def walk(node_id: str, parent_world: list[float]):
        node = pull(node_id)
        world = multiply(parent_world, node["matrix"])
        world_matrices[node_id] = world
        ordered_nodes.append(node_id)
        for binding in node.get("meshes", []):
            geometry = pull(binding["geometry"])
            pull(binding["material"])
            material = exported[binding["material"]]
            image = material.get("pbr", {}).get("baseColorImage")
            if image:
                pull(image)
            stats = geometry_stats.setdefault(binding["geometry"], {
                "vertexCount": geometry["vertexCount"],
                "triangleCount": geometry["triangleCount"],
                "instances": 0,
                "materials": set(),
            })
            stats["instances"] += 1
            stats["materials"].add(binding["material"])
        for child in node.get("children", []):
            walk(child, world)

    for root in scene.get("roots", []):
        walk(root, IDENTITY)

    # World-space samples for every geometry instance make transforms testable.
    for node_id in ordered_nodes:
        node = exported[node_id]
        for binding in node.get("meshes", []):
            geometry = exported[binding["geometry"]]
            world = world_matrices[node_id]
            key = f"{node_id}|{binding['geometry']}"
            world_points[key] = [transform_point(world, p) for p in geometry["positions"][:4]]

    bundle = {
        "format": "r3d-export",
        "version": 1,
        "sceneId": scene_id,
        "scene": scene,
        "resources": exported,
        "nodeOrder": ordered_nodes,
        "worldMatrices": world_matrices,
        "worldPointSamples": world_points,
        "geometryStats": [
            {
                "geometryId": geometry_id,
                "vertexCount": stats["vertexCount"],
                "triangleCount": stats["triangleCount"],
                "instances": stats["instances"],
                "materials": sorted(stats["materials"]),
            }
            for geometry_id, stats in sorted(geometry_stats.items())
        ],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(bundle, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        "utf-8",
    )
    return bundle
