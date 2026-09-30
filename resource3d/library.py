from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .errors import ImportErrorBundle
from .gltf import GltfImporter
from .hashing import canonical_json, sha256_bytes
from .models import Asset, Geometry, Image, Material, Mesh, Node, Scene, Skin, Texture

TEXTURE_USAGES = (
    ("color", "color_texture_id"),
    ("data", "data_texture_id"),
    ("normal", "normal_texture_id"),
    ("occlusion", "occlusion_texture_id"),
    ("emissive", "emissive_texture_id"),
)


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
                data["name"],
                tuple(data["base_color_factor"]),
                data["metallic_factor"],
                data["roughness_factor"],
                data["alpha_mode"],
                data["double_sided"],
                data.get("color_texture_id"),
                data.get("data_texture_id"),
                data.get("normal_texture_id"),
                data.get("occlusion_texture_id"),
                data.get("emissive_texture_id"),
                tuple(data.get("emissive_factor", (0.0, 0.0, 0.0))),
                data.get("normal_scale", 1.0),
                data.get("occlusion_strength", 1.0),
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
        collected = {object_id: obj.data() for object_id, obj in self._closure(asset_id).items()}
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps({"asset_id": asset_id, "objects": collected}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination

    def _closure(self, asset_id: str) -> dict[str, object]:
        asset = self.load_object(asset_id)
        assert isinstance(asset, Asset)
        collected: dict[str, object] = {}

        def visit(object_id: str) -> None:
            if object_id in collected:
                return
            obj = self.load_object(object_id)
            collected[object_id] = obj
            if isinstance(obj, Texture):
                visit(obj.image_id)
            elif isinstance(obj, Mesh):
                visit(obj.geometry_id)
                for material_id, _, _ in obj.material_bindings:
                    visit(material_id)
            elif isinstance(obj, Material):
                for _, field_name in TEXTURE_USAGES:
                    texture_id = getattr(obj, field_name)
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

    def compare_assets(self, old_asset_id: str, new_asset_id: str) -> dict[str, object]:
        """Return a material-local diff between two imported assets.

        Materials are matched by name. The report covers texture usage slot
        changes (including the backing image), plus the meshes/nodes that use
        each affected material, and added/removed texture and image objects.
        """
        old_objects = self._closure(old_asset_id)
        new_objects = self._closure(new_asset_id)

        def by_type(objects: dict[str, object], kind: type) -> dict[str, object]:
            return {oid: obj for oid, obj in objects.items() if isinstance(obj, kind)}

        old_materials = by_type(old_objects, Material)
        new_materials = by_type(new_objects, Material)
        old_materials_by_name = {mat.name: (mid, mat) for mid, mat in old_materials.items()}
        new_materials_by_name = {mat.name: (mid, mat) for mid, mat in new_materials.items()}

        def texture_summary(objects: dict[str, object], texture_id: str | None) -> dict[str, object] | None:
            if not texture_id:
                return None
            texture = objects.get(texture_id)
            if not isinstance(texture, Texture):
                return {"texture_id": texture_id, "name": None, "usage": None, "image_id": None}
            return {
                "texture_id": texture_id,
                "name": texture.name,
                "usage": texture.usage,
                "image_id": texture.image_id,
            }

        def material_users(objects: dict[str, object], material_id: str) -> list[dict[str, object]]:
            meshes = by_type(objects, Mesh)
            nodes = by_type(objects, Node)
            users: list[dict[str, object]] = []
            for mesh_id, mesh in meshes.items():
                if not any(bound_material == material_id for bound_material, _, _ in mesh.material_bindings):
                    continue
                mesh_nodes = [
                    {"node_id": node_id, "name": node.name}
                    for node_id, node in nodes.items()
                    if node.mesh_id == mesh_id
                ]
                users.append({"mesh_id": mesh_id, "mesh_name": mesh.name, "nodes": mesh_nodes})
            users.sort(key=lambda entry: entry["mesh_id"])
            return users

        def material_entry(objects: dict[str, object], material_id: str, material: Material) -> dict[str, object]:
            usages = {}
            for usage, field_name in TEXTURE_USAGES:
                summary = texture_summary(objects, getattr(material, field_name))
                if summary is not None:
                    usages[usage] = summary
            return {
                "material_id": material_id,
                "name": material.name,
                "usages": usages,
                "used_by": material_users(objects, material_id),
            }

        changed: list[dict[str, object]] = []
        for name in sorted(old_materials_by_name.keys() & new_materials_by_name.keys()):
            old_id, old_material = old_materials_by_name[name]
            new_id, new_material = new_materials_by_name[name]
            if old_id == new_id:
                continue
            added_usages: list[dict[str, object]] = []
            removed_usages: list[dict[str, object]] = []
            changed_usages: list[dict[str, object]] = []
            for usage, field_name in TEXTURE_USAGES:
                old_texture_id = getattr(old_material, field_name)
                new_texture_id = getattr(new_material, field_name)
                if old_texture_id == new_texture_id:
                    continue
                if old_texture_id is None:
                    added_usages.append({"usage": usage, "new": texture_summary(new_objects, new_texture_id)})
                elif new_texture_id is None:
                    removed_usages.append({"usage": usage, "old": texture_summary(old_objects, old_texture_id)})
                else:
                    old_texture = old_objects.get(old_texture_id)
                    new_texture = new_objects.get(new_texture_id)
                    changed_usages.append({
                        "usage": usage,
                        "old": texture_summary(old_objects, old_texture_id),
                        "new": texture_summary(new_objects, new_texture_id),
                        "old_image_id": old_texture.image_id if isinstance(old_texture, Texture) else None,
                        "new_image_id": new_texture.image_id if isinstance(new_texture, Texture) else None,
                    })
            user_groups: dict[str, dict[str, object]] = {}
            for source, material_key, mesh_entries in (
                ("old", old_id, material_users(old_objects, old_id)),
                ("new", new_id, material_users(new_objects, new_id)),
            ):
                for entry in mesh_entries:
                    group = user_groups.setdefault(
                        entry["mesh_name"],
                        {"mesh_name": entry["mesh_name"], "old_mesh_id": None, "new_mesh_id": None, "nodes": {}},
                    )
                    group[f"{source}_mesh_id"] = entry["mesh_id"]
                    for node in entry["nodes"]:
                        group["nodes"][node["node_id"]] = node
            changed.append({
                "name": name,
                "old_material_id": old_id,
                "new_material_id": new_id,
                "usages": {
                    "added": added_usages,
                    "removed": removed_usages,
                    "changed": changed_usages,
                },
                "used_by": [
                    {
                        "mesh_name": group["mesh_name"],
                        "old_mesh_id": group["old_mesh_id"],
                        "new_mesh_id": group["new_mesh_id"],
                        "nodes": [group["nodes"][key] for key in sorted(group["nodes"])],
                    }
                    for key, group in sorted(user_groups.items())
                ],
            })

        removed_names = sorted(old_materials_by_name.keys() - new_materials_by_name.keys())
        added_names = sorted(new_materials_by_name.keys() - old_materials_by_name.keys())

        def object_summaries(objects: dict[str, object], ids: set[str]) -> list[dict[str, object]]:
            summaries = []
            for object_id in sorted(ids):
                obj = objects[object_id]
                if isinstance(obj, Texture):
                    summaries.append({
                        "texture_id": object_id,
                        "name": obj.name,
                        "usage": obj.usage,
                        "image_id": obj.image_id,
                    })
                elif isinstance(obj, Image):
                    summaries.append({"image_id": object_id, "name": obj.name, "mime_type": obj.mime_type})
            return summaries

        old_texture_ids = set(by_type(old_objects, Texture))
        new_texture_ids = set(by_type(new_objects, Texture))
        old_image_ids = set(by_type(old_objects, Image))
        new_image_ids = set(by_type(new_objects, Image))
        return {
            "old_asset_id": old_asset_id,
            "new_asset_id": new_asset_id,
            "materials": {
                "added": [
                    material_entry(new_objects, *new_materials_by_name[name]) for name in added_names
                ],
                "removed": [
                    material_entry(old_objects, *old_materials_by_name[name]) for name in removed_names
                ],
                "changed": changed,
            },
            "textures": {
                "added": object_summaries(new_objects, new_texture_ids - old_texture_ids),
                "removed": object_summaries(old_objects, old_texture_ids - new_texture_ids),
            },
            "images": {
                "added": object_summaries(new_objects, new_image_ids - old_image_ids),
                "removed": object_summaries(old_objects, old_image_ids - new_image_ids),
            },
        }
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps({"asset_id": asset_id, "objects": collected}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination

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
