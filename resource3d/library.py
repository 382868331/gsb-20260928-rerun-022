from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

from .errors import ImportErrorBundle
from .gltf import GltfImporter
from .hashing import canonical_json, sha256_bytes
from .models import Asset, Geometry, Image, Material, Mesh, Node, Scene, Skin, TextureTile, TiledTexture, Texture
from .retarget import image_dimensions

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
        if kind == "ttx":
            tiles = tuple(
                TextureTile(
                    tile["tile_id"], tile["x"], tile["y"], tile["width"], tile["height"],
                    tile["image_id"], tile.get("source_tile_id"), tile.get("source_image_id"),
                    tile.get("source_texture_id"),
                )
                for tile in data["tiles"]
            )
            return TiledTexture(
                data["name"], data["usage"], tiles,
                data.get("atlas_width", 0), data.get("atlas_height", 0), data.get("tex_coord", 0),
            )
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
                dict(data.get("tiled_texture_ids", {})),
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

    def plan_retarget(self, asset_id: str, plan_path: Path | str) -> dict[str, object]:
        report, _ = self._prepare_retarget(asset_id, plan_path)
        return report

    def apply_retarget(self, asset_id: str, plan_path: Path | str) -> tuple[str, dict[str, object]]:
        report, prepared = self._prepare_retarget(asset_id, plan_path)
        if report["errors"]:
            raise ImportErrorBundle(list(report["errors"]))
        assert prepared is not None
        new_asset, new_objects, record = prepared
        new_asset_id = new_asset.id()
        existing = self._load_index()
        merged = {**existing.get("objects", {}), **{key: self._record(value) for key, value in new_objects.items()}}
        assets = {**existing.get("assets", {}), new_asset_id: record["plan"]}
        retargets = {**existing.get("retargets", {})}
        retargets[new_asset_id] = {
            "base_asset_id": asset_id,
            **{key: value for key, value in record.items() if key != "plan"},
        }
        self._commit(new_objects, {"version": 1, "assets": assets, "objects": merged, "retargets": retargets})
        report["new_asset_id"] = new_asset_id
        return new_asset_id, report

    def retarget_source(self, asset_id: str) -> str:
        record = self._load_index().get("retargets", {}).get(asset_id)
        if not record:
            raise KeyError(f"{asset_id} is not a retargeted asset")
        return record["base_asset_id"]

    def _prepare_retarget(
        self, asset_id: str, plan_path: Path | str
    ) -> tuple[dict[str, object], tuple[Asset, dict[str, object], dict[str, object]] | None]:
        plan_path = Path(plan_path)
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ImportErrorBundle([f"{plan_path}: unreadable or invalid JSON: {exc}"]) from exc
        if not isinstance(plan, dict):
            raise ImportErrorBundle([f"{plan_path}: plan must be an object"])
        if "tile_updates" in plan:
            if not isinstance(plan["tile_updates"], list) or "replacements" in plan:
                raise ImportErrorBundle([f"{plan_path}: tile_updates must be a list and cannot be combined with replacements"])
            return self._prepare_tile_update(asset_id, plan_path, plan)
        if not isinstance(plan.get("replacements"), list):
            raise ImportErrorBundle([f"{plan_path}: plan must contain replacements or tile_updates"])

        closure = self._closure(asset_id)
        images = {oid: obj for oid, obj in closure.items() if isinstance(obj, Image)}
        textures = {oid: obj for oid, obj in closure.items() if isinstance(obj, Texture)}
        materials = {oid: obj for oid, obj in closure.items() if isinstance(obj, Material)}
        meshes = {oid: obj for oid, obj in closure.items() if isinstance(obj, Mesh)}
        nodes = {oid: obj for oid, obj in closure.items() if isinstance(obj, Node)}
        scenes = {oid: obj for oid, obj in closure.items() if isinstance(obj, Scene)}
        base_asset = closure[asset_id]
        assert isinstance(base_asset, Asset)

        errors: list[str] = []
        # usage -> list of (source_image_id, new_image, new_texture, target_node_ids)
        resolved: list[dict[str, object]] = []
        for position, entry in enumerate(plan["replacements"]):
            label = f"replacements[{position}]"
            if not isinstance(entry, dict):
                errors.append(f"{label}: entry must be an object")
                continue
            source_ref = entry.get("image")
            usage = entry.get("usage")
            file_name = entry.get("file")
            source_image_id = None
            if isinstance(source_ref, str):
                if source_ref in images:
                    source_image_id = source_ref
                else:
                    matches = [oid for oid, img in images.items() if img.name == source_ref]
                    if len(matches) == 1:
                        source_image_id = matches[0]
            if source_image_id is None:
                errors.append(f"{label}: 图像缺失: 资产中找不到图像 {source_ref!r}")
                continue
            source_image = images[source_image_id]
            if usage not in dict(TEXTURE_USAGES):
                errors.append(f"{label}: 用途不兼容: 未知用途 {usage!r}")
                continue
            usage_textures = [
                tex for tex in textures.values()
                if tex.image_id == source_image_id and tex.usage == usage
            ]
            if not usage_textures:
                errors.append(
                    f"{label}: 用途不兼容: 图像 {source_ref!r} 在资产中没有 {usage} 用途的贴图"
                )
                continue
            if not isinstance(file_name, str) or not file_name:
                errors.append(f"{label}: 图像缺失: 未提供替换文件")
                continue
            replacement_path = (plan_path.parent / file_name).resolve()
            try:
                payload = replacement_path.read_bytes()
            except OSError:
                errors.append(f"{label}: 图像缺失: 无法读取 {file_name}")
                continue
            suffix = replacement_path.suffix.lower()
            mime_type = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(suffix, "")
            if mime_type not in ("image/png", "image/jpeg"):
                errors.append(f"{label}: 用途不兼容: 不支持的替换图像类型 {suffix!r}")
                continue
            if mime_type != source_image.mime_type:
                errors.append(
                    f"{label}: 用途不兼容: MIME 从 {source_image.mime_type} 变为 {mime_type}"
                )
                continue
            try:
                old_width, old_height = image_dimensions(source_image.mime_type, source_image.payload)
                new_width, new_height = image_dimensions(mime_type, payload)
            except ValueError as exc:
                errors.append(f"{label}: 尺寸关系不合法: {exc}")
                continue
            if (
                new_width > old_width
                or new_height > old_height
                or old_width * new_height != old_height * new_width
                or old_width % new_width != 0
                or old_height % new_height != 0
            ):
                errors.append(
                    f"{label}: 尺寸关系不合法: {new_width}x{new_height} 不是 "
                    f"{old_width}x{old_height} 的整数倍缩小"
                )
                continue
            node_filter = entry.get("nodes")
            target_node_ids: set[str] | None = None
            if node_filter is not None:
                if not isinstance(node_filter, list):
                    errors.append(f"{label}: nodes 必须是节点名称列表")
                    continue
                target_node_ids = set()
                for wanted in node_filter:
                    matches = [oid for oid, node in nodes.items() if node.name == wanted or oid == wanted]
                    if not matches:
                        errors.append(f"{label}: 找不到节点 {wanted!r}")
                    target_node_ids.update(matches)
            new_image = Image(f"{source_image.name}@{new_width}x{new_height}", mime_type, payload)
            resolved.append({
                "label": label,
                "source_image_id": source_image_id,
                "usage": usage,
                "new_image": new_image,
                "target_node_ids": target_node_ids,
                "old_size": (old_width, old_height),
                "new_size": (new_width, new_height),
            })
        if errors:
            return {"asset_id": asset_id, "valid": False, "errors": errors, "changes": []}, None

        # Build replacement textures/materials per usage, keeping each usage
        # record independent even when usages share one backing image.
        new_objects: dict[str, object] = {}
        texture_replacements: dict[str, str] = {}  # old tex id -> new tex id
        replacement_summaries: list[dict[str, object]] = []
        for item in resolved:
            new_image = item["new_image"]
            assert isinstance(new_image, Image)
            new_objects[new_image.id()] = new_image
            for texture in textures.values():
                if texture.image_id != item["source_image_id"] or texture.usage != item["usage"]:
                    continue
                width, height = item["new_size"]
                new_texture = Texture(f"{texture.name}@{width}x{height}", new_image.id(), texture.usage, texture.tex_coord)
                new_objects[new_texture.id()] = new_texture
                texture_replacements[texture.id()] = new_texture.id()
            replacement_summaries.append({
                "source_image_id": item["source_image_id"],
                "usage": item["usage"],
                "replacement_image_id": new_image.id(),
                "old_size": list(item["old_size"]),
                "new_size": list(item["new_size"]),
                "nodes": sorted(item["target_node_ids"]) if item["target_node_ids"] is not None else None,
            })

        usage_fields = dict(TEXTURE_USAGES)
        material_cache: dict[tuple[str, tuple[tuple[str, str], ...]], str] = {}

        def retarget_material(material_id: str, local_textures: dict[str, str]) -> str:
            key = (material_id, tuple(sorted(local_textures.items())))
            if key in material_cache:
                return material_cache[key]
            material = materials[material_id]
            updated = material
            for usage, field_name in TEXTURE_USAGES:
                texture_id = getattr(material, field_name)
                if texture_id and texture_id in local_textures:
                    updated = replace(updated, **{field_name: local_textures[texture_id]})
            if updated.id() != material_id:
                new_objects[updated.id()] = updated
            material_cache[key] = updated.id()
            return updated.id()

        # A node is affected only when every replacement that touches its
        # materials either targets it explicitly or has no node filter.
        def node_selected(node_id: str) -> list[dict[str, object]]:
            node = nodes[node_id]
            if not node.mesh_id or node.mesh_id not in meshes:
                return []
            mesh = meshes[node.mesh_id]
            picked = []
            for item in resolved:
                for material_id, _, _ in mesh.material_bindings:
                    material = materials.get(material_id)
                    if material is None:
                        continue
                    texture_id = getattr(material, usage_fields[item["usage"]])
                    texture = textures.get(texture_id) if texture_id else None
                    if (
                        texture is not None
                        and texture.image_id == item["source_image_id"]
                        and texture.usage == item["usage"]
                    ):
                        if item["target_node_ids"] is None or node_id in item["target_node_ids"]:
                            picked.append(item)
                        break
            return picked

        changes: list[dict[str, object]] = []
        node_mesh_override: dict[str, str] = {}
        for node_id, node in nodes.items():
            picked = node_selected(node_id)
            if not picked or not node.mesh_id:
                continue
            mesh = meshes[node.mesh_id]
            local_textures: dict[str, str] = {}
            for item in picked:
                for old_texture_id, new_texture_id in texture_replacements.items():
                    old_texture = textures[old_texture_id]
                    if (
                        old_texture.image_id == item["source_image_id"]
                        and old_texture.usage == item["usage"]
                    ):
                        local_textures[old_texture_id] = new_texture_id
            new_bindings = tuple(
                (retarget_material(material_id, local_textures), offset, count)
                for material_id, offset, count in mesh.material_bindings
            )
            if new_bindings == mesh.material_bindings:
                continue
            new_mesh = Mesh(mesh.name, mesh.geometry_id, new_bindings)
            new_objects[new_mesh.id()] = new_mesh
            node_mesh_override[node_id] = new_mesh.id()
            for (material_id, _, _), (new_material_id, _, _) in zip(mesh.material_bindings, new_bindings):
                if new_material_id == material_id:
                    continue
                material = materials[material_id]
                for usage, field_name in TEXTURE_USAGES:
                    old_tex = getattr(material, field_name)
                    if old_tex and old_tex in local_textures:
                        new_material = new_objects[new_material_id]
                        assert isinstance(new_material, Material)
                        changes.append({
                            "node_id": node_id,
                            "node_name": node.name,
                            "mesh_id": node.mesh_id,
                            "new_mesh_id": new_mesh.id(),
                            "material_name": material.name,
                            "old_material_id": material_id,
                            "new_material_id": new_material.id(),
                            "usage": usage,
                            "old_texture_id": old_tex,
                            "new_texture_id": getattr(new_material, field_name),
                            "old_image_id": textures[old_tex].image_id,
                            "new_image_id": new_objects[local_textures[old_tex]].image_id,  # type: ignore[attr-defined]
                        })

        # Rebuild the node hierarchy bottom-up; untouched subtrees keep their
        # original content-addressed ids so unselected objects stay shared.
        rebuilt_nodes: dict[str, str] = {}

        def rebuild_node(node_id: str) -> str:
            if node_id in rebuilt_nodes:
                return rebuilt_nodes[node_id]
            node = nodes[node_id]
            new_children = tuple(rebuild_node(child) for child in node.children if child in nodes)
            new_mesh_id = node_mesh_override.get(node_id, node.mesh_id)
            if new_children == node.children and new_mesh_id == node.mesh_id:
                rebuilt_nodes[node_id] = node_id
                return node_id
            new_node = Node(node.name, node.transform, new_mesh_id, new_children, node.skin_id)
            new_objects[new_node.id()] = new_node
            rebuilt_nodes[node_id] = new_node.id()
            return new_node.id()

        scene_replacements: dict[str, str] = {}
        for scene_id, scene in scenes.items():
            new_roots = tuple(rebuild_node(root) for root in scene.root_node_ids if root in nodes)
            if new_roots != scene.root_node_ids:
                new_scene = Scene(scene.name, new_roots)
                new_objects[new_scene.id()] = new_scene
                scene_replacements[scene_id] = new_scene.id()

        new_scene_ids = tuple(scene_replacements.get(sid, sid) for sid in base_asset.scene_ids)
        new_asset = Asset(base_asset.source_format, new_scene_ids)
        new_objects[new_asset.id()] = new_asset

        record = {
            "plan": f"retarget:{Path(plan_path).name}",
            "replacements": replacement_summaries,
        }
        report = {
            "asset_id": asset_id,
            "valid": True,
            "errors": [],
            "changes": sorted(changes, key=lambda item: (item["node_id"], item["usage"])),
            "new_asset_id": new_asset.id(),
        }
        return report, (new_asset, new_objects, record)

    def _prepare_tile_update(
        self, asset_id: str, plan_path: Path, plan: dict[str, object]
    ) -> tuple[dict[str, object], tuple[Asset, dict[str, object], dict[str, object]] | None]:
        closure = self._closure(asset_id)
        images = {oid: obj for oid, obj in closure.items() if isinstance(obj, Image)}
        tiled = {oid: obj for oid, obj in closure.items() if isinstance(obj, TiledTexture)}
        materials = {oid: obj for oid, obj in closure.items() if isinstance(obj, Material)}
        meshes = {oid: obj for oid, obj in closure.items() if isinstance(obj, Mesh)}
        nodes = {oid: obj for oid, obj in closure.items() if isinstance(obj, Node)}
        scenes = {oid: obj for oid, obj in closure.items() if isinstance(obj, Scene)}
        base_asset = closure[asset_id]
        assert isinstance(base_asset, Asset)

        def tiled_ref(ref: object) -> str | None:
            if isinstance(ref, str) and ref in tiled:
                return ref
            if isinstance(ref, str):
                matches = [oid for oid, value in tiled.items() if value.name == ref]
                if len(matches) == 1:
                    return matches[0]
            return None

        errors: list[str] = []
        requests: list[dict[str, object]] = []
        raw_updates = plan["tile_updates"]
        assert isinstance(raw_updates, list)
        for position, entry in enumerate(raw_updates):
            label = f"tile_updates[{position}]"
            if not isinstance(entry, dict):
                errors.append(f"{label}: entry must be an object")
                continue
            texture_id = tiled_ref(entry.get("tiled_texture"))
            if texture_id is None:
                errors.append(f"{label}: 图像缺失: 资产中找不到分片贴图 {entry.get('tiled_texture')!r}")
                continue
            priority = entry.get("priority", 0)
            if isinstance(priority, bool) or not isinstance(priority, int):
                errors.append(f"{label}: priority 必须是整数")
                continue
            if entry.get("withdraw") is True:
                node_filter = entry.get("nodes")
                if not isinstance(node_filter, list) or not node_filter:
                    errors.append(f"{label}: 撤回采用必须明确选择至少一个节点")
                    continue
                target_nodes = set()
                for wanted in node_filter:
                    matches = [oid for oid, node in nodes.items() if node.name == wanted or oid == wanted]
                    if not matches:
                        errors.append(f"{label}: 找不到节点 {wanted!r}")
                    target_nodes.update(matches)
                requests.append({
                    "label": label,
                    "position": position,
                    "priority": priority,
                    "old_tiled": tiled[texture_id],
                    "requested": {},
                    "target_nodes": target_nodes,
                    "withdraw": True,
                })
                continue
            old_tiled = tiled[texture_id]
            usage = entry.get("usage", old_tiled.usage)
            if usage != old_tiled.usage:
                errors.append(f"{label}: 用途不兼容: 不能把 {old_tiled.usage} 分片当 {usage} 使用")
                continue
            old_by_id = {tile.tile_id: tile for tile in old_tiled.tiles}
            raw_tiles = entry.get("tiles", {})
            if not isinstance(raw_tiles, dict):
                errors.append(f"{label}: tiles 必须是以分片 ID 为键的对象")
                continue
            requested: dict[str, TextureTile] = {}
            explicit_tiles = 0
            for tile_id, tile_spec in raw_tiles.items():
                tile_label = f"{label}.tiles[{tile_id}]"
                if tile_id not in old_by_id:
                    errors.append(f"{tile_label}: 用途不兼容: 未知分片位置 {tile_id!r}")
                    continue
                old_tile = old_by_id[tile_id]
                if tile_spec is None:
                    explicit_tiles += 1
                    requested[tile_id] = old_tile
                    continue
                if not isinstance(tile_spec, dict) or not isinstance(tile_spec.get("file"), str):
                    errors.append(f"{tile_label}: 图像缺失: 未提供替换文件")
                    continue
                path = (plan_path.parent / tile_spec["file"]).resolve()
                try:
                    payload = path.read_bytes()
                except OSError:
                    errors.append(f"{tile_label}: 图像缺失: 无法读取 {tile_spec['file']}")
                    continue
                mime_type = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(path.suffix.lower(), "")
                old_image = images[old_tile.image_id]
                if mime_type not in ("image/png", "image/jpeg") or mime_type != old_image.mime_type:
                    errors.append(f"{tile_label}: 用途不兼容: 替换图像必须保持 {old_image.mime_type}")
                    continue
                try:
                    new_width, new_height = image_dimensions(mime_type, payload)
                except ValueError as exc:
                    errors.append(f"{tile_label}: 尺寸关系不合法: {exc}")
                    continue
                old_width, old_height = old_tile.width, old_tile.height
                if (
                    new_width > old_width or new_height > old_height
                    or old_width * new_height != old_height * new_width
                    or old_width % new_width or old_height % new_height
                ):
                    errors.append(
                        f"{tile_label}: 尺寸关系不合法: {new_width}x{new_height} 不是 {old_width}x{old_height} 的等比整数缩小"
                    )
                    continue
                image = Image(f"{old_image.name}#{tile_id}@{new_width}x{new_height}", mime_type, payload)
                explicit_tiles += 1
                requested[tile_id] = TextureTile(
                    tile_id, old_tile.x, old_tile.y, old_tile.width, old_tile.height,
                    image.id(), old_tile.tile_id, old_tile.image_id, old_tile.source_texture_id or old_tiled.id(),
                )
                requested[tile_id + "__image"] = image  # type: ignore[assignment]
            node_filter = entry.get("nodes")
            target_nodes: set[str] | None = None
            if node_filter is not None:
                if not isinstance(node_filter, list):
                    errors.append(f"{label}: nodes 必须是节点名称或 ID 列表")
                    continue
                target_nodes = set()
                for wanted in node_filter:
                    matches = [oid for oid, node in nodes.items() if node.name == wanted or oid == wanted]
                    if not matches:
                        errors.append(f"{label}: 找不到节点 {wanted!r}")
                    target_nodes.update(matches)
            requests.append({
                "label": label,
                "position": position,
                "priority": priority,
                "old_tiled": old_tiled,
                "requested": requested,
                "target_nodes": target_nodes,
                "explicit_tiles": explicit_tiles,
            })
        if errors:
            return {"asset_id": asset_id, "valid": False, "errors": errors, "changes": [], "missing_areas": []}, None

        new_objects: dict[str, object] = {}
        updates_by_old_texture: dict[str, dict[str, object]] = {}
        summaries: list[dict[str, object]] = []
        requests_by_texture: dict[str, list[dict[str, object]]] = {}
        for request in requests:
            grouped = request["old_tiled"]
            assert isinstance(grouped, TiledTexture)
            requests_by_texture.setdefault(grouped.id(), []).append(request)

        for old_texture_id, group in requests_by_texture.items():
            old_tiled = tiled[old_texture_id]
            withdrawals = [req for req in group if req.get("withdraw")]
            candidates = [req for req in group if not req.get("withdraw")]
            if withdrawals and candidates:
                errors.append(f"{group[0]['label']}: 撤回候选不能与更新候选混合登记同一分片贴图")
                continue
            if len(withdrawals) > 1:
                errors.append(f"{withdrawals[1]['label']}: 同一分片贴图只能登记一个撤回候选")
                continue
            if withdrawals:
                request = withdrawals[0]
                source_refs = {tile.source_texture_id for tile in old_tiled.tiles if tile.source_texture_id}
                if len(source_refs) != 1 or not source_refs.intersection(tiled):
                    errors.append(f"{request['label']}: 无法从来源记录恢复原始分片贴图")
                    continue
                original_id = next(iter(source_refs))
                updates_by_old_texture[old_texture_id] = {
                    "candidates": [request],
                    "combos": {
                        frozenset([request["position"]]): {
                            "texture": tiled[original_id],
                            "tiles": [
                                {
                                    "tile_id": tile.tile_id,
                                    "x": tile.x,
                                    "y": tile.y,
                                    "width": tile.width,
                                    "height": tile.height,
                                    "image_id": tile.image_id,
                                    "state": "withdrawn",
                                }
                                for tile in old_tiled.tiles
                            ],
                            "retained": [],
                        }
                    },
                }
                summaries.append({
                    "old_tiled_texture_id": old_tiled.id(),
                    "new_tiled_texture_id": original_id,
                    "usage": old_tiled.usage,
                    "atlas_size": [old_tiled.atlas_width, old_tiled.atlas_height],
                    "nodes": sorted(request["target_nodes"]),
                    "affected_nodes": sorted(request["target_nodes"]),
                    "tiles": [
                        {
                            "tile_id": tile.tile_id,
                            "x": tile.x,
                            "y": tile.y,
                            "width": tile.width,
                            "height": tile.height,
                            "image_id": tile.image_id,
                            "state": "withdrawn",
                            "source_tile_id": tile.source_tile_id,
                            "source_image_id": tile.source_image_id,
                            "source_texture_id": tile.source_texture_id,
                        }
                        for tile in old_tiled.tiles
                    ],
                    "retained_tile_ids": [],
                    "edges": self._tile_edges(old_tiled),
                    "action": "withdraw",
                })
                continue
            for req in candidates:
                if req.get("explicit_tiles", 0) == 0:
                    errors.append(f"{req['label']}: 没有任何分片发生变化")
            if errors:
                continue

            def applicable_key(node_id: str, pool: list[dict[str, object]] = candidates) -> frozenset:
                return frozenset(
                    req["position"] for req in pool
                    if req["target_nodes"] is None or node_id in req["target_nodes"]
                )

            combo_map: dict[frozenset, dict[str, object] | None] = {}
            overlap_reported: set[tuple[str, tuple[int, ...]]] = set()
            candidate_keys = {applicable_key(node_id) for node_id in nodes}
            candidate_keys.discard(frozenset())
            for key in sorted(candidate_keys, key=lambda item: sorted(item)):
                active = [req for req in candidates if req["position"] in key]
                combined_tiles: list[TextureTile] = []
                tile_summary: list[dict[str, object]] = []
                retained: list[str] = []
                for old_tile in old_tiled.tiles:
                    providers = [req for req in active if old_tile.tile_id in req["requested"]]
                    winner = None
                    if providers:
                        best_priority = max(req["priority"] for req in providers)
                        winners = [req for req in providers if req["priority"] == best_priority]
                        if len(winners) > 1:
                            marker = (old_tile.tile_id, tuple(sorted(req["position"] for req in winners)))
                            if marker not in overlap_reported:
                                overlap_reported.add(marker)
                                errors.append(
                                    f"分片 {old_tile.tile_id!r} 的同一位置候选重叠: "
                                    + "、".join(req["label"] for req in winners)
                                    + " 优先级相同, 须通过 priority 明确解决重叠"
                                )
                        else:
                            winner = winners[0]
                    if winner is None:
                        tile = TextureTile(
                            old_tile.tile_id, old_tile.x, old_tile.y, old_tile.width, old_tile.height,
                            old_tile.image_id,
                            old_tile.source_tile_id or old_tile.tile_id,
                            old_tile.source_image_id or old_tile.image_id,
                            old_tile.source_texture_id or old_tiled.id(),
                        )
                        retained.append(old_tile.tile_id)
                        tile_summary.append({
                            "tile_id": tile.tile_id,
                            "x": tile.x,
                            "y": tile.y,
                            "width": tile.width,
                            "height": tile.height,
                            "image_id": old_tile.image_id,
                            "state": "retained",
                        })
                        combined_tiles.append(tile)
                        continue
                    requested = winner["requested"]
                    assert isinstance(requested, dict)
                    tile = requested[old_tile.tile_id]
                    image = requested.get(old_tile.tile_id + "__image")
                    if image is None:
                        retained.append(old_tile.tile_id)
                        tile_summary.append({
                            "tile_id": tile.tile_id,
                            "x": tile.x,
                            "y": tile.y,
                            "width": tile.width,
                            "height": tile.height,
                            "image_id": tile.image_id,
                            "state": "retained",
                            "source_entry": winner["label"],
                        })
                        combined_tiles.append(tile)
                        continue
                    assert isinstance(image, Image)
                    new_objects[image.id()] = image
                    new_width, new_height = image_dimensions(image.mime_type, image.payload)
                    tile_summary.append({
                        "tile_id": tile.tile_id,
                        "x": tile.x,
                        "y": tile.y,
                        "width": tile.width,
                        "height": tile.height,
                        "old_image_id": old_tile.image_id,
                        "new_image_id": tile.image_id,
                        "old_size": [tile.width, tile.height],
                        "new_size": [new_width, new_height],
                        "state": "updated",
                        "source_entry": winner["label"],
                    })
                    combined_tiles.append(tile)
                if all(
                    tile.image_id == old_tiled.tiles[index].image_id
                    for index, tile in enumerate(combined_tiles)
                ):
                    combo_map[key] = None
                    continue
                combined = TiledTexture(
                    f"{old_tiled.name}#partial", old_tiled.usage, tuple(combined_tiles),
                    old_tiled.atlas_width, old_tiled.atlas_height, old_tiled.tex_coord,
                )
                new_objects[combined.id()] = combined
                combo_map[key] = {"texture": combined, "tiles": tile_summary, "retained": retained}
            if errors:
                continue
            adopted = [(key, combo) for key, combo in combo_map.items() if combo]
            if not adopted:
                errors.append(f"{candidates[0]['label']}: 没有任何分片发生变化")
                continue
            updates_by_old_texture[old_texture_id] = {"candidates": candidates, "combos": combo_map}
            affected_nodes = sorted(
                node_id for node_id in nodes if combo_map.get(applicable_key(node_id))
            )
            combinations = [
                {
                    "nodes": sorted(
                        node_id for node_id in nodes
                        if applicable_key(node_id) == key and combo_map.get(key)
                    ),
                    "new_tiled_texture_id": combo["texture"].id(),
                    "tiles": combo["tiles"],
                    "retained_tile_ids": combo["retained"],
                }
                for key, combo in adopted
            ]
            merged_tiles: dict[str, dict[str, object]] = {}
            for _, combo in adopted:
                for tile_entry in combo["tiles"]:
                    previous = merged_tiles.get(tile_entry["tile_id"])
                    if previous is None or previous["state"] != "updated":
                        merged_tiles[tile_entry["tile_id"]] = tile_entry
            retained_union = sorted({
                tile_id for _, combo in adopted for tile_id in combo["retained"]
            })
            summary: dict[str, object] = {
                "old_tiled_texture_id": old_texture_id,
                "usage": old_tiled.usage,
                "atlas_size": [old_tiled.atlas_width, old_tiled.atlas_height],
                "action": "adopt",
                "candidates": [
                    {
                        "entry": req["label"],
                        "priority": req["priority"],
                        "nodes": sorted(req["target_nodes"]) if req["target_nodes"] is not None else None,
                        "tile_ids": sorted(
                            tile_id for tile_id in req["requested"] if not tile_id.endswith("__image")
                        ),
                    }
                    for req in candidates
                ],
                "combinations": combinations,
                "affected_nodes": affected_nodes,
                "nodes": (
                    sorted(candidates[0]["target_nodes"])
                    if len(candidates) == 1 and candidates[0]["target_nodes"] is not None
                    else (None if len(candidates) == 1 else affected_nodes)
                ),
                "tiles": list(merged_tiles.values()),
                "retained_tile_ids": retained_union,
                "edges": self._tile_edges(adopted[0][1]["texture"]),
            }
            if len(adopted) == 1:
                summary["new_tiled_texture_id"] = adopted[0][1]["texture"].id()
            summaries.append(summary)
        if errors:
            return {"asset_id": asset_id, "valid": False, "errors": errors, "changes": [], "missing_areas": []}, None

        material_cache: dict[tuple[str, tuple[tuple[str, str], ...]], str] = {}

        def material_for(material_id: str, adopted: tuple[tuple[str, str], ...]) -> str:
            key = (material_id, adopted)
            if key in material_cache:
                return material_cache[key]
            material = materials[material_id]
            adopted_map = dict(adopted)
            tiled_maps = dict(material.tiled_texture_ids)
            for usage, tiled_id in list(tiled_maps.items()):
                if tiled_id in adopted_map:
                    tiled_maps[usage] = adopted_map[tiled_id]
            updated = replace(material, tiled_texture_ids=tiled_maps)
            if updated.id() != material_id:
                new_objects[updated.id()] = updated
            material_cache[key] = updated.id()
            return updated.id()

        changes: list[dict[str, object]] = []
        node_mesh_override: dict[str, str] = {}
        for node_id, node in nodes.items():
            if not node.mesh_id or node.mesh_id not in meshes:
                continue
            mesh = meshes[node.mesh_id]
            applicable: dict[str, dict[str, object]] = {}
            for material_id, _, _ in mesh.material_bindings:
                material = materials.get(material_id)
                if material is None:
                    continue
                for usage, old_tiled_id in material.tiled_texture_ids.items():
                    update = updates_by_old_texture.get(old_tiled_id)
                    if update is None:
                        continue
                    key = frozenset(
                        req["position"] for req in update["candidates"]
                        if req["target_nodes"] is None or node_id in req["target_nodes"]
                    )
                    combo = update["combos"].get(key)
                    if combo is not None:
                        applicable[old_tiled_id] = combo
            if not applicable:
                continue
            new_bindings = []
            for material_id, offset, count in mesh.material_bindings:
                material = materials[material_id]
                chosen = material_id
                material_applicable = tuple(sorted(
                    (old_tiled_id, applicable[old_tiled_id]["texture"].id())
                    for old_tiled_id in material.tiled_texture_ids.values()
                    if old_tiled_id in applicable
                ))
                if material_applicable:
                    chosen = material_for(material_id, material_applicable)
                    for usage, old_tex_id in material.tiled_texture_ids.items():
                        combo = applicable.get(old_tex_id)
                        if combo is None:
                            continue
                        changes.append({
                            "node_id": node_id,
                            "node_name": node.name,
                            "mesh_id": mesh.id(),
                            "material_name": material.name,
                            "usage": usage,
                            "old_tiled_texture_id": old_tex_id,
                            "new_tiled_texture_id": combo["texture"].id(),
                            "updated_tile_ids": [
                                tile["tile_id"] for tile in combo["tiles"]
                                if tile["state"] == "updated"
                            ],
                            "retained_tile_ids": [
                                tile["tile_id"] for tile in combo["tiles"]
                                if tile["state"] == "retained"
                            ],
                            "atlas_size": [
                                tiled[old_tex_id].atlas_width,
                                tiled[old_tex_id].atlas_height,
                            ],
                        })
                new_bindings.append((chosen, offset, count))
            new_bindings_tuple = tuple(new_bindings)
            if new_bindings_tuple != mesh.material_bindings:
                new_mesh = Mesh(mesh.name, mesh.geometry_id, new_bindings_tuple)
                new_objects[new_mesh.id()] = new_mesh
                node_mesh_override[node_id] = new_mesh.id()

        rebuilt_nodes: dict[str, str] = {}

        def rebuild_node(node_id: str) -> str:
            if node_id in rebuilt_nodes:
                return rebuilt_nodes[node_id]
            node = nodes[node_id]
            child_ids = tuple(rebuild_node(child) for child in node.children if child in nodes)
            mesh_id = node_mesh_override.get(node_id, node.mesh_id)
            if child_ids == node.children and mesh_id == node.mesh_id:
                rebuilt_nodes[node_id] = node_id
                return node_id
            rebuilt = Node(node.name, node.transform, mesh_id, child_ids, node.skin_id)
            new_objects[rebuilt.id()] = rebuilt
            rebuilt_nodes[node_id] = rebuilt.id()
            return rebuilt.id()

        scene_map = {}
        for scene_id, scene in scenes.items():
            roots = tuple(rebuild_node(root) for root in scene.root_node_ids if root in nodes)
            if roots != scene.root_node_ids:
                rebuilt_scene = Scene(scene.name, roots)
                new_objects[rebuilt_scene.id()] = rebuilt_scene
                scene_map[scene_id] = rebuilt_scene.id()
        new_asset = Asset(base_asset.source_format, tuple(scene_map.get(sid, sid) for sid in base_asset.scene_ids))
        new_objects[new_asset.id()] = new_asset
        missing_areas = [
            {
                "tiled_texture_id": item["old_tiled_texture_id"],
                "usage": item["usage"],
                "missing_tile_ids": item["retained_tile_ids"],
                "policy": "retain-old-image",
            }
            for item in summaries if item["retained_tile_ids"]
        ]
        report = {
            "asset_id": asset_id,
            "base_asset_id": asset_id,
            "valid": True,
            "errors": [],
            "changes": sorted(changes, key=lambda item: (item["node_id"], item["usage"])),
            "missing_areas": missing_areas,
            "tile_updates": summaries,
            "new_asset_id": new_asset.id(),
        }
        record = {
            "plan": f"tile-retarget:{plan_path.name}",
            "tile_updates": summaries,
        }
        return report, (new_asset, new_objects, record)

    @staticmethod
    def _tile_edges(texture: TiledTexture) -> list[dict[str, object]]:
        edges: list[dict[str, object]] = []
        tiles = list(texture.tiles)
        for index, first in enumerate(tiles):
            for second in tiles[index + 1:]:
                if first.x + first.width == second.x:
                    overlap_start = max(first.y, second.y)
                    overlap_end = min(first.y + first.height, second.y + second.height)
                    if overlap_start < overlap_end:
                        edges.append({
                            "a": first.tile_id, "b": second.tile_id,
                            "side": "x", "coordinate": second.x,
                            "start": overlap_start, "end": overlap_end,
                        })
                if second.x + second.width == first.x:
                    overlap_start = max(first.y, second.y)
                    overlap_end = min(first.y + first.height, second.y + second.height)
                    if overlap_start < overlap_end:
                        edges.append({
                            "a": second.tile_id, "b": first.tile_id,
                            "side": "x", "coordinate": first.x,
                            "start": overlap_start, "end": overlap_end,
                        })
                if first.y + first.height == second.y:
                    overlap_start = max(first.x, second.x)
                    overlap_end = min(first.x + first.width, second.x + second.width)
                    if overlap_start < overlap_end:
                        edges.append({
                            "a": first.tile_id, "b": second.tile_id,
                            "side": "y", "coordinate": second.y,
                            "start": overlap_start, "end": overlap_end,
                        })
                if second.y + second.height == first.y:
                    overlap_start = max(first.x, second.x)
                    overlap_end = min(first.x + first.width, second.x + second.width)
                    if overlap_start < overlap_end:
                        edges.append({
                            "a": second.tile_id, "b": first.tile_id,
                            "side": "y", "coordinate": first.y,
                            "start": overlap_start, "end": overlap_end,
                        })
        return sorted(edges, key=lambda edge: (edge["side"], edge["coordinate"], edge["start"], edge["a"], edge["b"]))

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
            elif isinstance(obj, TiledTexture):
                for tile in obj.tiles:
                    visit(tile.image_id)
            elif isinstance(obj, Mesh):
                visit(obj.geometry_id)
                for material_id, _, _ in obj.material_bindings:
                    visit(material_id)
            elif isinstance(obj, Material):
                for _, field_name in TEXTURE_USAGES:
                    texture_id = getattr(obj, field_name)
                    if texture_id:
                        visit(texture_id)
                for tiled_texture_id in obj.tiled_texture_ids.values():
                    visit(tiled_texture_id)
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
        if not {"version", "assets", "objects"} <= set(data) <= {"version", "assets", "objects", "retargets"}:
            raise ValueError("invalid library index")
        return data
