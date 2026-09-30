from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .errors import ImportErrorBundle
from .gltf import GltfImporter
from .hashing import canonical_json, sha256_bytes
from .models import Asset, Geometry, Image, Material, Mesh, Node, Scene, Skin, Texture

TEXTURE_USAGES = ("color", "data", "normal", "occlusion", "emissive")


class ResourceLibrary:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.objects_dir = self.root / "objects"
        self.index_path = self.root / "index.json"

    def import_batch(self, manifest: Path | str) -> str:
        manifest = Path(manifest).resolve()
        asset, objects = GltfImporter(manifest, manifest.parent).load()
        existing = self._load_index()
        merged = {**existing.get("objects", {}), **{key: self._record(value) for key, value in objects.items()}}
        assets = {**existing.get("assets", {}), asset.id(): manifest.name}
        self._commit(objects, {"version": 1, "assets": assets, "objects": merged})
        return asset.id()

    def load_object(self, object_id: str):
        record = self._load_index()["objects"][object_id]
        path = self.objects_dir / record["path"]
        data = json.loads(path.read_text(encoding="utf-8"))
        kind = object_id.split("_", 1)[0]
        if kind == "geo":
            return Geometry(
                tuple(data["positions"]),
                tuple(data["triangle_indices"]),
                tuple(data["normals"]) if data.get("normals") is not None else None,
                tuple(data["skin_joints"]) if data.get("skin_joints") is not None else None,
                tuple(data["skin_weights"]) if data.get("skin_weights") is not None else None,
            )
        if kind == "img":
            return Image(data["name"], data["mime_type"], bytes.fromhex(data["payload"]))
        if kind == "tex":
            return Texture(data["name"], data["image_id"], data["usage"], data.get("tex_coord", 0))
        if kind == "mat":
            return Material(
                name=data["name"],
                base_color_factor=tuple(data["base_color_factor"]),
                metallic_factor=data["metallic_factor"],
                roughness_factor=data["roughness_factor"],
                emissive_factor=tuple(data.get("emissive_factor", [0.0, 0.0, 0.0])),
                alpha_mode=data["alpha_mode"],
                double_sided=data["double_sided"],
                color_texture_id=data.get("color_texture_id"),
                data_texture_id=data.get("data_texture_id"),
                normal_texture_id=data.get("normal_texture_id"),
                occlusion_texture_id=data.get("occlusion_texture_id"),
                emissive_texture_id=data.get("emissive_texture_id"),
            )
        if kind == "msh":
            return Mesh(
                data["name"],
                data["geometry_id"],
                tuple((b["material_id"], b["face_offset"], b["face_count"]) for b in data["material_bindings"]),
            )
        if kind == "ndo":
            return Node(
                data["name"],
                tuple(data["transform"]),
                data.get("mesh_id"),
                tuple(data["children"]),
                data.get("skin_id"),
            )
        if kind == "skn":
            return Skin(
                data["name"],
                tuple(data["joint_node_ids"]),
                tuple(tuple(matrix) for matrix in data["inverse_bind_matrices"]),
            )
        if kind == "scn":
            return Scene(data["name"], tuple(data["root_node_ids"]))
        if kind == "ast":
            return Asset(data["source_format"], tuple(data["scene_ids"]))
        raise KeyError(object_id)

    def export_inspection(self, asset_id: str, destination: Path | str) -> Path:
        collected = self._collect_closure(asset_id)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps({"asset_id": asset_id, "objects": collected}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination

    def _collect_closure(self, asset_id: str) -> dict[str, object]:
        asset = self.load_object(asset_id)
        assert isinstance(asset, Asset)
        collected: dict[str, object] = {}

        def visit(object_id: str) -> None:
            if object_id in collected:
                return
            obj = self.load_object(object_id)
            collected[object_id] = obj.data()
            if isinstance(obj, Image):
                return
            if isinstance(obj, Texture):
                visit(obj.image_id)
            if isinstance(obj, Mesh):
                visit(obj.geometry_id)
                for material_id, _, _ in obj.material_bindings:
                    visit(material_id)
            elif isinstance(obj, Material):
                for texture_id in obj.texture_slots().values():
                    if texture_id:
                        visit(texture_id)
            elif isinstance(obj, Node):
                if obj.mesh_id:
                    visit(obj.mesh_id)
                for child_id in obj.children:
                    visit(child_id)
                if obj.skin_id:
                    visit(obj.skin_id)
            elif isinstance(obj, Skin):
                for joint_id in obj.joint_node_ids:
                    visit(joint_id)
            elif isinstance(obj, Scene):
                for node_id in obj.root_node_ids:
                    visit(node_id)
            elif isinstance(obj, Asset):
                for scene_id in obj.scene_ids:
                    visit(scene_id)

        visit(asset_id)
        return collected

    def material_update_report(self, old_asset_id: str, new_asset_id: str) -> dict[str, object]:
        """Compare two asset versions and report material-local updates.

        Only materials whose content changed are reported.  Each entry lists
        the texture usages that changed (added/removed/replaced binding, and
        the underlying shared image reference) and the mesh/node objects that
        use the material in each version.
        """
        old_objects = self._collect_closure(old_asset_id)
        new_objects = self._collect_closure(new_asset_id)
        old_materials = {value["name"]: (key, value) for key, value in old_objects.items() if "base_color_factor" in value}
        new_materials = {value["name"]: (key, value) for key, value in new_objects.items() if "base_color_factor" in value}

        def image_of(objects: dict[str, object], texture_id: str | None) -> str | None:
            if not texture_id:
                return None
            texture = objects.get(texture_id)
            return texture["image_id"] if texture else None  # type: ignore[index]

        def users(objects: dict[str, object], material_id: str) -> dict[str, list[str]]:
            mesh_names = sorted(
                value["name"]
                for value in objects.values()
                if "material_bindings" in value
                and any(binding["material_id"] == material_id for binding in value["material_bindings"])
            )
            mesh_ids = {
                key
                for key, value in objects.items()
                if "material_bindings" in value
                and any(binding["material_id"] == material_id for binding in value["material_bindings"])
            }
            node_names = sorted(
                value["name"]
                for value in objects.values()
                if value.get("mesh_id") in mesh_ids
            )
            return {"meshes": mesh_names, "nodes": node_names}

        changed: list[dict[str, object]] = []
        for name in sorted(set(old_materials) & set(new_materials)):
            old_id, old_material = old_materials[name]
            new_id, new_material = new_materials[name]
            if old_id == new_id:
                continue
            usage_changes: list[dict[str, object]] = []
            for usage in TEXTURE_USAGES:
                slot = f"{usage}_texture_id"
                old_texture = old_material.get(slot)
                new_texture = new_material.get(slot)
                old_image = image_of(old_objects, old_texture)
                new_image = image_of(new_objects, new_texture)
                if old_texture == new_texture and old_image == new_image:
                    continue
                if not old_texture and new_texture:
                    change = "added"
                elif old_texture and not new_texture:
                    change = "removed"
                else:
                    change = "replaced"
                usage_changes.append({
                    "usage": usage,
                    "change": change,
                    "old_texture_id": old_texture,
                    "new_texture_id": new_texture,
                    "old_image_id": old_image,
                    "new_image_id": new_image,
                })
            factor_changes: dict[str, dict[str, object]] = {}
            for factor in ("base_color_factor", "metallic_factor", "roughness_factor", "emissive_factor", "alpha_mode", "double_sided"):
                if old_material.get(factor) != new_material.get(factor):
                    factor_changes[factor] = {"old": old_material.get(factor), "new": new_material.get(factor)}
            changed.append({
                "name": name,
                "old_material_id": old_id,
                "new_material_id": new_id,
                "usage_changes": usage_changes,
                "factor_changes": factor_changes,
                "users_before": users(old_objects, old_id),
                "users_after": users(new_objects, new_id),
            })
        added = sorted(name for name in new_materials if name not in old_materials)
        removed = sorted(name for name in old_materials if name not in new_materials)
        return {
            "old_asset_id": old_asset_id,
            "new_asset_id": new_asset_id,
            "changed_materials": changed,
            "added_materials": added,
            "removed_materials": removed,
        }

    def _record(self, value: object) -> dict[str, str]:
        object_id = value.id()  # type: ignore[attr-defined]
        kind = object_id.split("_", 1)[0]
        return {"kind": kind, "path": f"{kind}/{object_id}.json", "sha256": ""}

    def _commit(self, objects: dict[str, object], index: dict[str, object]) -> None:
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        for object_id, value in objects.items():
            kind = object_id.split("_", 1)[0]
            target = self.objects_dir / kind / f"{object_id}.json"
            if target.exists():
                continue
            payload = canonical_json(value.data())  # type: ignore[attr-defined]
            target.parent.mkdir(parents=True, exist_ok=True)
            self._atomic_write(target, payload + b"\n")
            index["objects"][object_id]["sha256"] = sha256_bytes(payload)
        for object_id, record in index["objects"].items():
            path = self.objects_dir / record["path"]
            if path.exists() and not record.get("sha256"):
                record["sha256"] = sha256_bytes(path.read_bytes().rstrip())
        self.root.mkdir(parents=True, exist_ok=True)
        self._atomic_write(self.index_path, canonical_json(index) + b"\n")

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    def _load_index(self) -> dict[str, object]:
        if not self.index_path.exists():
            return {"version": 1, "assets": {}, "objects": {}}
        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        if set(data) != {"version", "assets", "objects"}:
            raise ValueError("invalid library index")
        return data
