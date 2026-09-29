"""Internal, format-neutral 3D resource model.

Resource kinds
--------------
geometry: triangulated mesh data (positions, normals, uvs, triangle indices)
material: PBR material referencing optional image resources
image:    binary image blob reference
node:     transform + children + optional geometry/material bindings
scene:    ordered collection of root node ids

Every *structured* resource (geometry/material/node/scene) is serialized as
canonical JSON and identified by ``sha256(canonical_json(payload))``. Binary
image data is identified by the hash of its raw bytes.  Identity therefore
depends only on content: importing the same asset twice never creates a
duplicate, regardless of which source format produced it.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

GEOMETRY = "geometry"
MATERIAL = "material"
IMAGE = "image"
NODE = "node"
SCENE = "scene"

# Binary blobs are namespaced separately from structured resource hashes.
BLOB_PREFIX = "blob"


class ResourceError(ValueError):
    """Raised when a resource payload violates the model invariants."""


def _round_float(value: float) -> float:
    # Round-trip through repr keeps JSON stable while removing float noise.
    return float(repr(float(value)))


def _clean_numbers(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return _round_float(value)
    if isinstance(value, list):
        return [_clean_numbers(item) for item in value]
    if isinstance(value, dict):
        return {key: _clean_numbers(value[key]) for key in sorted(value)}
    return value


def canonical_json(payload: dict[str, Any]) -> bytes:
    """Deterministic serialization shared by all resources in every format."""
    return json.dumps(
        _clean_numbers(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def resource_id(kind: str, payload: dict[str, Any]) -> str:
    """Stable content id for a structured resource."""
    digest = hashlib.sha256(canonical_json(payload)).hexdigest()
    return f"{kind}-{digest}"


def blob_id(data: bytes) -> str:
    """Stable content id for a binary blob."""
    digest = hashlib.sha256(data).hexdigest()
    return f"{BLOB_PREFIX}-{digest}"


def _vec3(value: Iterable[Any], name: str) -> list[float]:
    values = [float(item) for item in value]
    if len(values) != 3:
        raise ResourceError(f"{name} must have exactly 3 components")
    return values


def _vec4(value: Iterable[Any], name: str) -> list[float]:
    values = [float(item) for item in value]
    if len(values) != 4:
        raise ResourceError(f"{name} must have exactly 4 components")
    return values


def _mat4(value: Iterable[Any], name: str) -> list[float]:
    values = [_round_float(item) for item in value]
    if len(values) != 16:
        raise ResourceError(f"{name} must be a column-major 4x4 matrix")
    return values


def make_geometry(
    *,
    positions: list[list[float]],
    triangles: list[list[int]],
    normals: list[list[float]] | None = None,
    uvs: list[list[float]] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Create a triangulated geometry resource; returns (id, payload)."""
    vertex_count = len(positions)
    if vertex_count == 0:
        raise ResourceError("geometry must contain at least one vertex")
    if any(len(point) != 3 for point in positions):
        raise ResourceError("positions must be vec3")
    if not triangles:
        raise ResourceError("geometry must contain at least one triangle")
    for triangle in triangles:
        if len(triangle) != 3:
            raise ResourceError("triangles must contain 3 vertex indices")
        for index in triangle:
            if not isinstance(index, int) or not (0 <= index < vertex_count):
                raise ResourceError("triangle index out of range")
    if normals is not None:
        if len(normals) != vertex_count or any(len(n) != 3 for n in normals):
            raise ResourceError("normals must be vec3 per vertex")
    if uvs is not None:
        if len(uvs) != vertex_count or any(len(uv) != 2 for uv in uvs):
            raise ResourceError("uvs must be vec2 per vertex")

    payload: dict[str, Any] = {
        "type": GEOMETRY,
        "version": 1,
        "vertexCount": vertex_count,
        "triangleCount": len(triangles),
        "positions": [_vec3(p, "position") for p in positions],
        "triangles": [[int(i) for i in tri] for tri in triangles],
    }
    if normals is not None:
        payload["normals"] = [_vec3(n, "normal") for n in normals]
    if uvs is not None:
        payload["uvs"] = [
            [_round_float(uv[0]), _round_float(uv[1])] for uv in uvs
        ]
    return resource_id(GEOMETRY, payload), payload


def make_material(
    *,
    base_color_factor: Iterable[float] = (1.0, 1.0, 1.0, 1.0),
    base_color_image: str | None = None,
    metallic_factor: float = 1.0,
    roughness_factor: float = 1.0,
    name: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Create a PBR material referencing an optional image id."""
    metallic = _round_float(metallic_factor)
    roughness = _round_float(roughness_factor)
    if not 0.0 <= metallic <= 1.0 or not 0.0 <= roughness <= 1.0:
        raise ResourceError("metallic/roughness factors must be within [0, 1]")
    pbr: dict[str, Any] = {
        "baseColorFactor": _vec4(base_color_factor, "baseColorFactor"),
        "metallicFactor": metallic,
        "roughnessFactor": roughness,
    }
    if base_color_image is not None:
        if not base_color_image.startswith(f"{IMAGE}-"):
            raise ResourceError("baseColorImage must reference an image id")
        pbr["baseColorImage"] = base_color_image
    payload: dict[str, Any] = {"type": MATERIAL, "version": 1, "pbr": pbr}
    if name:
        payload["name"] = str(name)
    return resource_id(MATERIAL, payload), payload


def make_image(*, media_type: str, data: bytes) -> tuple[str, dict[str, Any]]:
    """Create an image resource backed by a binary blob."""
    if not data:
        raise ResourceError("image data must not be empty")
    payload = {
        "type": IMAGE,
        "version": 1,
        "mediaType": media_type,
        "byteLength": len(data),
        "blobId": blob_id(data),
    }
    return resource_id(IMAGE, payload), payload


def make_node(
    *,
    matrix: Iterable[float],
    children: Iterable[str] = (),
    meshes: Iterable[dict[str, str]] = (),
    name: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Create a hierarchy node.

    ``matrix`` is a column-major 4x4 local transform. ``meshes`` maps a
    geometry id to the material id used to draw it.
    """
    child_ids = list(children)
    for child in child_ids:
        if not child.startswith(f"{NODE}-"):
            raise ResourceError("node children must reference node ids")
    mesh_bindings: list[dict[str, str]] = []
    for binding in meshes:
        geometry = binding.get("geometry")
        material = binding.get("material")
        if not geometry or not geometry.startswith(f"{GEOMETRY}-"):
            raise ResourceError("mesh binding requires a geometry id")
        if not material or not material.startswith(f"{MATERIAL}-"):
            raise ResourceError("mesh binding requires a material id")
        mesh_bindings.append({"geometry": geometry, "material": material})
    payload: dict[str, Any] = {
        "type": NODE,
        "version": 1,
        "matrix": _mat4(matrix, "matrix"),
    }
    if name:
        payload["name"] = str(name)
    if mesh_bindings:
        payload["meshes"] = mesh_bindings
    if child_ids:
        payload["children"] = child_ids
    return resource_id(NODE, payload), payload


def make_scene(
    *,
    roots: Iterable[str],
    up_axis: str = "Y",
    name: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Create a scene resource rooted at the given node ids."""
    root_ids = list(roots)
    for root in root_ids:
        if not root.startswith(f"{NODE}-"):
            raise ResourceError("scene roots must reference node ids")
    if up_axis not in {"X", "Y", "Z"}:
        raise ResourceError("upAxis must be X, Y or Z")
    payload: dict[str, Any] = {
        "type": SCENE,
        "version": 1,
        "upAxis": up_axis,
        "roots": root_ids,
    }
    if name:
        payload["name"] = str(name)
    return resource_id(SCENE, payload), payload
