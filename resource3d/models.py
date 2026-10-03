from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .hashing import content_id


@dataclass(frozen=True)
class Geometry:
    positions: tuple[float, ...]
    triangle_indices: tuple[int, ...]
    normals: tuple[float, ...] | None = None
    skin_joints: tuple[int, ...] | None = None
    skin_weights: tuple[float, ...] | None = None

    def id(self) -> str:
        return content_id("geo", self.data())

    def data(self) -> dict[str, Any]:
        data = {
            "positions": list(self.positions),
            "triangle_indices": list(self.triangle_indices),
        }
        if self.normals is not None:
            data["normals"] = list(self.normals)
        if self.skin_joints is not None:
            data["skin_joints"] = list(self.skin_joints)
        if self.skin_weights is not None:
            data["skin_weights"] = list(self.skin_weights)
        return data


@dataclass(frozen=True)
class Image:
    name: str
    mime_type: str
    payload: bytes

    def id(self) -> str:
        return content_id("img", self.data())

    def data(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mime_type": self.mime_type,
            "payload": self.payload.hex(),
        }


@dataclass(frozen=True)
class Texture:
    name: str
    image_id: str
    usage: Literal["color", "data", "normal", "occlusion", "emissive"]
    tex_coord: int = 0
    tile_position: tuple[int, int] | None = None
    tile_layout: tuple[int, int] | None = None
    tile_edges: tuple[tuple[str, ...], ...] | None = None

    def id(self) -> str:
        return content_id("tex", self.data())

    def data(self) -> dict[str, Any]:
        data = {
            "name": self.name,
            "image_id": self.image_id,
            "usage": self.usage,
            "tex_coord": self.tex_coord,
        }
        if self.tile_position is not None and self.tile_layout is not None:
            data["tile"] = {
                "position": list(self.tile_position),
                "layout": list(self.tile_layout),
            }
            if self.tile_edges is not None:
                data["tile"]["edges"] = [list(edge) for edge in self.tile_edges]
        return data


@dataclass(frozen=True)
class Material:
    name: str
    base_color_factor: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    metallic_factor: float = 1.0
    roughness_factor: float = 1.0
    alpha_mode: Literal["OPAQUE", "MASK", "BLEND"] = "OPAQUE"
    double_sided: bool = False
    color_texture_id: str | None = None
    data_texture_id: str | None = None
    normal_texture_id: str | None = None
    occlusion_texture_id: str | None = None
    emissive_texture_id: str | None = None
    emissive_factor: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal_scale: float = 1.0
    occlusion_strength: float = 1.0
    tile_textures: tuple[tuple[str, tuple[str, ...]], ...] = ()

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
            "color_texture_id": self.color_texture_id,
            "data_texture_id": self.data_texture_id,
            "normal_texture_id": self.normal_texture_id,
            "occlusion_texture_id": self.occlusion_texture_id,
            "emissive_texture_id": self.emissive_texture_id,
            "emissive_factor": list(self.emissive_factor),
            "normal_scale": self.normal_scale,
            "occlusion_strength": self.occlusion_strength,
            "tile_textures": [
                {"usage": usage, "texture_ids": list(texture_ids)}
                for usage, texture_ids in self.tile_textures
            ],
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
    skin_id: str | None = None

    def id(self) -> str:
        return content_id("ndo", self.data())

    def data(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "transform": list(self.transform),
            "mesh_id": self.mesh_id,
            "children": list(self.children),
            "skin_id": self.skin_id,
        }


@dataclass(frozen=True)
class Skin:
    name: str
    joint_node_ids: tuple[str, ...]
    inverse_bind_matrices: tuple[tuple[float, ...], ...]

    def id(self) -> str:
        return content_id("skn", self.data())

    def data(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "joint_node_ids": list(self.joint_node_ids),
            "inverse_bind_matrices": [list(matrix) for matrix in self.inverse_bind_matrices],
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
