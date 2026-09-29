from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .hashing import content_id


@dataclass(frozen=True)
class Geometry:
    positions: tuple[float, ...]
    triangle_indices: tuple[int, ...]
    normals: tuple[float, ...] | None = None

    def id(self) -> str:
        return content_id("geo", self.data())

    def data(self) -> dict[str, Any]:
        data = {
            "positions": list(self.positions),
            "triangle_indices": list(self.triangle_indices),
        }
        if self.normals is not None:
            data["normals"] = list(self.normals)
        return data


@dataclass(frozen=True)
class Material:
    name: str
    base_color_factor: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    metallic_factor: float = 1.0
    roughness_factor: float = 1.0
    alpha_mode: Literal["OPAQUE", "MASK", "BLEND"] = "OPAQUE"
    double_sided: bool = False

    def id(self) -> str:
        return content_id("mat", self.data())

    def data(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_color_factor": list(self.base_color_factor),
            "metallic_factor": self.metallic_factor,
            "roughness_factor": self.roughness_factor,
            "alpha_mode": self.alpha_mode,
            "double_sided": self.double_sided,
        }


@dataclass(frozen=True)
class Mesh:
    name: str
    geometry_id: str
    material_bindings: tuple[tuple[str, int, int], ...]

    def id(self) -> str:
        return content_id("msh", self.data())

    def data(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "geometry_id": self.geometry_id,
            "material_bindings": [
                {"material_id": material_id, "face_offset": offset, "face_count": count}
                for material_id, offset, count in self.material_bindings
            ],
        }


@dataclass(frozen=True)
class Node:
    name: str
    transform: tuple[float, ...]
    mesh_id: str | None = None
    children: tuple[str, ...] = field(default_factory=tuple)

    def id(self) -> str:
        return content_id("ndo", self.data())

    def data(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "transform": list(self.transform),
            "mesh_id": self.mesh_id,
            "children": list(self.children),
        }


@dataclass(frozen=True)
class Scene:
    name: str
    root_node_ids: tuple[str, ...]

    def id(self) -> str:
        return content_id("scn", self.data())

    def data(self) -> dict[str, Any]:
        return {"name": self.name, "root_node_ids": list(self.root_node_ids)}


@dataclass(frozen=True)
class Asset:
    source_format: str
    scene_ids: tuple[str, ...]

    def id(self) -> str:
        return content_id("ast", self.data())

    def data(self) -> dict[str, Any]:
        return {"source_format": self.source_format, "scene_ids": list(self.scene_ids)}
