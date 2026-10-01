from __future__ import annotations

import struct
from dataclasses import replace
from pathlib import Path

from .errors import ImportErrorBundle
from .hashing import sha256_bytes
from .models import (
    Asset,
    Image,
    Material,
    Mesh,
    Node,
    RetargetRecord,
    Scene,
    Texture,
)

TEXTURE_USAGE_FIELDS = (
    ("color", "color_texture_id"),
    ("data", "data_texture_id"),
    ("normal", "normal_texture_id"),
    ("occlusion", "occlusion_texture_id"),
    ("emissive", "emissive_texture_id"),
)
USAGE_NAMES = tuple(name for name, _ in TEXTURE_USAGE_FIELDS)
MIME_BY_SUFFIX = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def image_dimensions(image: Image) -> tuple[int, int]:
    payload = image.payload
    if image.mime_type == "image/png":
        if len(payload) < 24 or payload[:8] != b"\x89PNG\r\n\x1a\n" or payload[12:16] != b"IHDR":
            raise ImportErrorBundle([f"image {image.name!r}: corrupt PNG header"])
        width, height = struct.unpack(">II", payload[16:24])
        return width, height
    if image.mime_type == "image/jpeg":
        if len(payload) < 4 or payload[:2] != b"\xff\xd8":
            raise ImportErrorBundle([f"image {image.name!r}: corrupt JPEG header"])
        offset = 2
        while offset + 9 < len(payload):
            if payload[offset] != 0xFF:
                offset += 1
                continue
            marker = payload[offset + 1]
            if marker in (0xC0, 0xC1, 0xC2):
                height, width = struct.unpack(">HH", payload[offset + 5 : offset + 9])
                return width, height
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                offset += 2
                continue
            (segment_length,) = struct.unpack(">H", payload[offset + 2 : offset + 4])
            offset += 2 + segment_length
        raise ImportErrorBundle([f"image {image.name!r}: JPEG size marker not found"])
    raise ImportErrorBundle([f"image {image.name!r}: unsupported mime type {image.mime_type!r}"])


def _check_size_relation(original: tuple[int, int], replacement: tuple[int, int], label: str) -> str | None:
    orig_w, orig_h = original
    new_w, new_h = replacement
    if new_w <= 0 or new_h <= 0:
        return f"{label}: replacement dimensions must be positive, got {new_w}x{new_h}"
    if new_w >= orig_w or new_h >= orig_h:
        return f"{label}: replacement {new_w}x{new_h} is not smaller than original {orig_w}x{orig_h}"
    if orig_w * new_h != orig_h * new_w:
        return f"{label}: aspect ratio mismatch between {orig_w}x{orig_h} and {new_w}x{new_h}"
    if orig_w % new_w != 0 or orig_h % new_h != 0 or orig_w // new_w != orig_h // new_h:
        return f"{label}: {orig_w}x{orig_h} is not an integer multiple of {new_w}x{new_h}"
    return None


class RetargetPlanner:
    """Plan and apply per-object texture resolution replacement."""

    def __init__(self, library, asset_id: str, spec: dict, spec_dir: Path):
        self.library = library
        self.asset_id = asset_id
        self.spec = spec
        self.spec_dir = Path(spec_dir).resolve()
        self.errors: list[str] = []
        self.objects = library._closure(asset_id)
        self.images = {oid: obj for oid, obj in self.objects.items() if isinstance(obj, Image)}
        self.textures = {oid: obj for oid, obj in self.objects.items() if isinstance(obj, Texture)}
        self.materials = {oid: obj for oid, obj in self.objects.items() if isinstance(obj, Material)}
        self.meshes = {oid: obj for oid, obj in self.objects.items() if isinstance(obj, Mesh)}
        self.nodes = {oid: obj for oid, obj in self.objects.items() if isinstance(obj, Node)}
        # node_id -> list of (replacement_index, usage, material_id, old_texture_id)
        self.adoptions: dict[str, list[tuple[int, str, str, str]]] = {}
        self.replacements: dict[int, dict] = {}

    def plan(self) -> dict:
        replacements = self.spec.get("replacements")
        if not isinstance(replacements, list) or not replacements:
            raise ImportErrorBundle(["retarget spec must list at least one replacement"])
        for index, entry in enumerate(replacements):
            self._plan_replacement(index, entry)
        changes = self._change_report()
        return {
            "asset_id": self.asset_id,
            "changes": changes,
            "errors": list(self.errors),
        }

    def apply(self) -> str:
        plan = self.plan()
        if self.errors:
            raise ImportErrorBundle(self.errors)
        new_objects: dict[str, object] = {}
        record_entries: list[dict] = []
        texture_map: dict[str, str] = {}
        for repl in self.replacements.values():
            new_objects[repl["new_image"].id()] = repl["new_image"]
            for old_texture_id in repl["texture_ids"]:
                old_texture = self.textures[old_texture_id]
                new_texture = Texture(
                    name=old_texture.name,
                    image_id=repl["new_image"].id(),
                    usage=old_texture.usage,
                    tex_coord=old_texture.tex_coord,
                )
                texture_map[old_texture_id] = new_texture.id()
                new_objects[new_texture.id()] = new_texture
        material_map: dict[tuple[str, tuple[tuple[str, str], ...]], str] = {}
        mesh_map: dict[tuple[str, tuple[tuple[str, str], ...]], str] = {}
        node_mesh_override: dict[str, str] = {}
        for node_id, adoptions in self.adoptions.items():
            node = self.nodes[node_id]
            assert node.mesh_id is not None
            mesh = self.meshes[node.mesh_id]
            # (material_id, usage) replacements for this node's mesh
            swaps: dict[tuple[str, str], str] = {}
            for repl_index, usage, material_id, old_texture_id in adoptions:
                swaps[(material_id, usage)] = texture_map[old_texture_id]
            new_bindings = []
            mesh_changed = False
            for material_id, face_offset, face_count in mesh.material_bindings:
                material = self.materials[material_id]
                material_changes = {
                    field: swaps[(material_id, usage)]
                    for usage, field in TEXTURE_USAGE_FIELDS
                    if (material_id, usage) in swaps
                }
                if not material_changes:
                    new_bindings.append((material_id, face_offset, face_count))
                    continue
                mesh_changed = True
                key = (material_id, tuple(sorted(material_changes.items())))
                if key not in material_map:
                    new_material = replace(material, **material_changes)
                    material_map[key] = new_material.id()
                    new_objects[new_material.id()] = new_material
                new_bindings.append((material_map[key], face_offset, face_count))
            if not mesh_changed:
                continue
            mesh_key = (node.mesh_id, tuple(sorted(new_bindings)))
            if mesh_key not in mesh_map:
                new_mesh = Mesh(mesh.name, mesh.geometry_id, tuple(new_bindings))
                mesh_map[mesh_key] = new_mesh.id()
                new_objects[new_mesh.id()] = new_mesh
            node_mesh_override[node_id] = mesh_map[mesh_key]
        # Rebuild changed nodes and their ancestors so parents reference the
        # new child ids; untouched subtrees keep their content-addressed ids.
        final_node_ids: dict[str, str] = {}

        def resolve_node(node_id: str) -> str:
            if node_id in final_node_ids:
                return final_node_ids[node_id]
            node = self.nodes[node_id]
            children = tuple(resolve_node(child_id) for child_id in node.children)
            mesh_id = node_mesh_override.get(node_id, node.mesh_id)
            if children == node.children and mesh_id == node.mesh_id:
                final_node_ids[node_id] = node_id
            else:
                new_node = replace(node, mesh_id=mesh_id, children=children)
                new_objects[new_node.id()] = new_node
                final_node_ids[node_id] = new_node.id()
            return final_node_ids[node_id]

        base_asset = self.objects[self.asset_id]
        assert isinstance(base_asset, Asset)
        new_scene_ids = []
        for scene_id in base_asset.scene_ids:
            scene = self.objects[scene_id]
            assert isinstance(scene, Scene)
            new_roots = tuple(resolve_node(root_id) for root_id in scene.root_node_ids)
            if new_roots == scene.root_node_ids:
                new_scene_ids.append(scene_id)
                continue
            new_scene = Scene(scene.name, new_roots)
            new_objects[new_scene.id()] = new_scene
            new_scene_ids.append(new_scene.id())
        for repl in self.replacements.values():
            for entry in repl["record_entries"]:
                record_entries.append(entry)
        record_entries.sort(key=lambda entry: (entry["material"], entry["usage"], entry["node"]))
        record = RetargetRecord(self.asset_id, tuple(record_entries))
        new_objects[record.id()] = record
        new_asset = Asset(base_asset.source_format, tuple(new_scene_ids), record.id())
        new_objects[new_asset.id()] = new_asset
        existing = self.library._load_index()
        merged = {
            **existing.get("objects", {}),
            **{key: self.library._record(value) for key, value in new_objects.items()},
        }
        assets = {**existing.get("assets", {}), new_asset.id(): f"retarget:{self.asset_id}"}
        self.library._commit(new_objects, {"version": 1, "assets": assets, "objects": merged})
        return new_asset.id()

    def _plan_replacement(self, index: int, entry: dict) -> None:
        label = f"replacements[{index}]"
        if not isinstance(entry, dict):
            self.errors.append(f"{label}: must be an object")
            return
        image_name = entry.get("image")
        image_ids = [oid for oid, img in self.images.items() if img.name == image_name]
        if not image_ids:
            self.errors.append(f"{label}: image {image_name!r} not found in asset")
            return
        image_id = image_ids[0]
        original = self.images[image_id]
        file_value = entry.get("file")
        if not isinstance(file_value, str) or not file_value:
            self.errors.append(f"{label}: replacement file is required")
            return
        path = (self.spec_dir / file_value).resolve()
        if path != self.spec_dir and self.spec_dir not in path.parents:
            self.errors.append(f"{label}: replacement file {file_value!r} escapes the spec directory")
            return
        if not path.is_file():
            self.errors.append(f"{label}: replacement file {file_value!r} is missing")
            return
        mime_type = MIME_BY_SUFFIX.get(path.suffix.lower())
        if mime_type is None:
            self.errors.append(f"{label}: replacement file {file_value!r} is not PNG or JPEG")
            return
        if mime_type != original.mime_type:
            self.errors.append(
                f"{label}: replacement mime {mime_type!r} does not match original {original.mime_type!r}"
            )
            return
        new_image = Image(path.stem, mime_type, path.read_bytes())
        try:
            size_error = _check_size_relation(
                image_dimensions(original), image_dimensions(new_image), label
            )
        except ImportErrorBundle as exc:
            self.errors.extend(exc.errors)
            return
        if size_error:
            self.errors.append(size_error)
            return
        usages = entry.get("usages")
        explicit_usages = usages is not None
        if usages is None:
            usages = list(USAGE_NAMES)
        if not isinstance(usages, list) or any(u not in USAGE_NAMES for u in usages):
            self.errors.append(f"{label}: usages must be a subset of {list(USAGE_NAMES)!r}")
            return
        node_names = entry.get("objects")
        if not isinstance(node_names, list) or not node_names:
            self.errors.append(f"{label}: objects must list at least one node name")
            return
        target_node_ids: list[str] = []
        for name in node_names:
            matches = [oid for oid, node in self.nodes.items() if node.name == name]
            if not matches:
                self.errors.append(f"{label}: object {name!r} not found in asset")
                continue
            target_node_ids.extend(matches)
        if not target_node_ids:
            return
        texture_ids: set[str] = set()
        record_entries: list[dict] = []
        matched_usages: set[str] = set()
        for node_id in target_node_ids:
            node = self.nodes[node_id]
            if node.mesh_id is None:
                self.errors.append(f"{label}: object {node.name!r} has no mesh")
                continue
            mesh = self.meshes[node.mesh_id]
            for material_id, _, _ in mesh.material_bindings:
                material = self.materials[material_id]
                for usage, field in TEXTURE_USAGE_FIELDS:
                    if usage not in usages:
                        continue
                    texture_id = getattr(material, field)
                    if not texture_id:
                        continue
                    texture = self.textures[texture_id]
                    if texture.image_id != image_id:
                        continue
                    matched_usages.add(usage)
                    texture_ids.add(texture_id)
                    self.adoptions.setdefault(node_id, []).append(
                        (index, usage, material_id, texture_id)
                    )
                    record_entries.append({
                        "node": node.name,
                        "material": material.name,
                        "usage": usage,
                        "image": original.name,
                        "old_image_id": image_id,
                        "new_image_id": new_image.id(),
                        "old_texture_id": texture_id,
                        "replacement": {
                            "name": path.name,
                            "sha256": sha256_bytes(new_image.payload),
                        },
                    })
        for usage in usages:
            if explicit_usages and usage not in matched_usages:
                self.errors.append(
                    f"{label}: image {image_name!r} is not used as {usage!r} by the selected objects"
                )
        self.replacements[index] = {
            "new_image": new_image,
            "texture_ids": sorted(texture_ids),
            "record_entries": record_entries,
        }

    def _change_report(self) -> list[dict]:
        report: dict[tuple[str, str], dict] = {}
        for node_id, adoptions in self.adoptions.items():
            node = self.nodes[node_id]
            for repl_index, usage, material_id, old_texture_id in adoptions:
                material = self.materials[material_id]
                key = (material.name, node.name)
                entry = report.setdefault(key, {
                    "material": material.name,
                    "node": node.name,
                    "switches": [],
                })
                repl = self.replacements.get(repl_index)
                entry["switches"].append({
                    "usage": usage,
                    "old_texture_id": old_texture_id,
                    "old_image_id": self.textures[old_texture_id].image_id,
                    "new_image_id": repl["new_image"].id() if repl else None,
                })
        for entry in report.values():
            entry["switches"].sort(key=lambda item: item["usage"])
        return [report[key] for key in sorted(report)]
