from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

from .errors import ImportErrorBundle
from .gltf import GltfImporter
from .hashing import canonical_json, sha256_bytes
from .models import Asset, Geometry, Image, Material, Mesh, Node, Scene, Skin, Texture
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
            tile = data.get("tile")
            position = tuple(tile["position"]) if tile else None
            layout = tuple(tile["layout"]) if tile else None
            edges = tuple(tuple(edge) for edge in tile["edges"]) if tile and tile.get("edges") else None
            return Texture(
                data["name"], data["image_id"], data["usage"], data.get("tex_coord", 0),
                position, layout, edges,
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
                tuple(
                    (entry["usage"], tuple(entry["texture_ids"]))
                    for entry in data.get("tile_textures", [])
                ),
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
        if self._plan_kind(plan_path) == "tiles":
            return self._tile_report(asset_id, plan_path)
        report, _ = self._prepare_retarget(asset_id, plan_path)
        return report

    def apply_retarget(self, asset_id: str, plan_path: Path | str) -> tuple[str, dict[str, object]]:
        if self._plan_kind(plan_path) == "tiles":
            return self._apply_tile_version(asset_id, plan_path)
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
            "replacements": record["replacements"],
        }
        self._commit(new_objects, {"version": 1, "assets": assets, "objects": merged, "retargets": retargets})
        report["new_asset_id"] = new_asset_id
        return new_asset_id, report

    def retarget_source(self, asset_id: str) -> str:
        index = self._load_index()
        record = index.get("retargets", {}).get(asset_id) or index.get("tile_versions", {}).get(asset_id)
        if not record:
            raise KeyError(f"{asset_id} is not a retargeted asset")
        return record["base_asset_id"]

    # ------------------------------------------------------------------
    # Tiled texture versions
    # ------------------------------------------------------------------

    def _tile_report(self, asset_id: str, plan_path: Path | str) -> dict[str, object]:
        report, _ = self._prepare_tile_version(asset_id, plan_path)
        return report

    def _apply_tile_version(self, asset_id: str, plan_path: Path | str) -> tuple[str, dict[str, object]]:
        report, prepared = self._prepare_tile_version(asset_id, plan_path)
        if report["errors"]:
            raise ImportErrorBundle(list(report["errors"]))
        assert prepared is not None
        new_asset, new_objects, record = prepared
        new_asset_id = new_asset.id()
        existing = self._load_index()
        merged = {**existing.get("objects", {}), **{key: self._record(value) for key, value in new_objects.items()}}
        assets = {**existing.get("assets", {}), new_asset_id: record["plan"]}
        tile_versions = {**existing.get("tile_versions", {})}
        tile_versions[new_asset_id] = record["provenance"]
        self._commit(
            new_objects,
            {
                "version": 1,
                "assets": assets,
                "objects": merged,
                "tile_versions": tile_versions,
                **({"retargets": existing["retargets"]} if "retargets" in existing else {}),
            },
        )
        report["new_asset_id"] = new_asset_id
        return new_asset_id, report

    def _prepare_tile_version(
        self, asset_id: str, plan_path: Path | str
    ) -> tuple[dict[str, object], tuple[Asset, dict[str, object], dict[str, object]] | None]:
        plan_path = Path(plan_path)
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ImportErrorBundle([f"{plan_path}: unreadable or invalid JSON: {exc}"]) from exc

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
        revert_targets = plan.get("revert")
        if revert_targets is not None and ("tiles" in plan or "nodes" in plan):
            errors.append("revert cannot be combined with tile updates")
        if revert_targets is not None:
            if not isinstance(revert_targets, list) or not all(isinstance(item, str) for item in revert_targets):
                errors.append("revert must be a list of node names or node ids")
                revert_targets = []
            index = self._load_index()
            root_asset_id = self._tile_base_chain(asset_id, index.get("tile_versions", {}), errors)
            if errors:
                return self._invalid_tile_report(asset_id, errors), None
            requested_nodes = self._resolve_node_names(nodes, revert_targets, "revert", errors)
            if errors:
                return self._invalid_tile_report(asset_id, errors), None
            return self._build_tile_version(
                asset_id=asset_id,
                closure=closure,
                root_asset_id=root_asset_id,
                parsed=[],
                textures=textures,
                images=images,
                materials=materials,
                meshes=meshes,
                nodes=nodes,
                scenes=scenes,
                base_asset=base_asset,
                plan_path=plan_path,
                revert_nodes=requested_nodes,
            )

        tile_entries = plan.get("tiles", [])
        if not isinstance(tile_entries, list):
            errors.append("tiles must be a list")
            tile_entries = []

        # Material version currently used at each node (this asset may already
        # be a tiled version with per-node material variants).
        node_material: dict[str, set[str]] = {}
        for node_id, node in nodes.items():
            if node.mesh_id and node.mesh_id in meshes:
                node_material[node_id] = {
                    material_id for material_id, _, _ in meshes[node.mesh_id].material_bindings
                }
        named_materials: dict[str, set[str]] = {}
        for material_id, material in materials.items():
            named_materials.setdefault(material.name, set()).add(material_id)

        # Tile layout per material name / usage, taken from any tiled variant.
        layouts: dict[tuple[str, str], tuple[int, int]] = {}
        for material in materials.values():
            for usage, texture_ids in material.tile_textures:
                tile_texture = textures[texture_ids[0]]
                assert tile_texture.tile_layout is not None
                layouts[(material.name, usage)] = tile_texture.tile_layout

        parsed: list[dict[str, object]] = []
        seen_keys: set[tuple[str, str, tuple[int, int]]] = set()
        for position_index, entry in enumerate(tile_entries):
            label = f"tiles[{position_index}]"
            if not isinstance(entry, dict):
                errors.append(f"{label}: entry must be an object")
                continue
            material_name = entry.get("material")
            usage = entry.get("usage")
            pos_raw = entry.get("position")
            file_name = entry.get("file")
            if not isinstance(material_name, str) or material_name not in named_materials:
                errors.append(f"{label}: 材质缺失: 找不到分片材质 {material_name!r}")
                continue
            if usage not in dict(TEXTURE_USAGES):
                errors.append(f"{label}: 用途不兼容: 未知用途 {usage!r}")
                continue
            layout = layouts.get((material_name, usage))
            if layout is None:
                errors.append(f"{label}: 用途不兼容: 材质 {material_name!r} 的 {usage} 用途没有分片布局")
                continue
            try:
                if not isinstance(pos_raw, list) or len(pos_raw) != 2:
                    raise ValueError
                column, row = int(pos_raw[0]), int(pos_raw[1])
            except (TypeError, ValueError):
                errors.append(f"{label}: 分片位置不合法: position 必须是 [column, row]")
                continue
            if not (0 <= column < layout[0] and 0 <= row < layout[1]):
                errors.append(
                    f"{label}: 分片位置不合法: {column},{row} 超出 {layout[0]}x{layout[1]} 布局"
                )
                continue
            key = (material_name, usage, (column, row))
            if key in seen_keys:
                errors.append(f"{label}: 同一分片 {material_name!r}/{usage}/{column},{row} 被重复更新")
                continue
            seen_keys.add(key)
            if not isinstance(file_name, str) or not file_name:
                errors.append(f"{label}: 图像缺失: 未提供分片文件")
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
                errors.append(f"{label}: 用途不兼容: 不支持的图像类型 {suffix!r}")
                continue
            # The new tile replaces the currently used tile in every material
            # variant (per-node versions can already differ). Each variant
            # keeps its own old/new source identity.
            variant_sources: dict[str, dict[str, object]] = {}
            entry_errors: list[str] = []
            for material_id in named_materials[material_name]:
                material = materials[material_id]
                slots = dict(material.tile_textures)
                if usage not in slots:
                    continue
                tile_id = self._tile_at(slots[usage], textures, column, row)
                if tile_id is None:
                    entry_errors.append(f"{label}: 分片位置不合法: 材质版本缺少 {column},{row} 分片")
                    continue
                tile = textures[tile_id]
                old_image = images.get(tile.image_id)
                if old_image is None:
                    entry_errors.append(f"{label}: 图像缺失: 找不到当前分片图像")
                    continue
                if mime_type != old_image.mime_type:
                    entry_errors.append(
                        f"{label}: 用途不兼容: MIME 从 {old_image.mime_type} 变为 {mime_type}"
                    )
                    continue
                try:
                    old_width, old_height = image_dimensions(old_image.mime_type, old_image.payload)
                    new_width, new_height = image_dimensions(mime_type, payload)
                except ValueError as exc:
                    entry_errors.append(f"{label}: 尺寸关系不合法: {exc}")
                    continue
                variant_sources[material_id] = {
                    "old_texture_id": tile_id,
                    "old_image_id": old_image.id(),
                    "old_size": (old_width, old_height),
                    "new_size": (new_width, new_height),
                    "edges": tile.tile_edges,
                }
            if entry_errors:
                errors.extend(dict.fromkeys(entry_errors))
                continue
            supplied_edges = entry.get("edges")
            supplied: tuple[tuple[str, ...], ...] | None = None
            if supplied_edges is not None:
                if not isinstance(supplied_edges, list) or len(supplied_edges) != 4 or not all(
                    isinstance(side, list) and all(isinstance(tag, str) for tag in side)
                    for side in supplied_edges
                ):
                    errors.append(f"{label}: 边缘对应不合法: edges 需要四组标签 (left, right, top, bottom)")
                    continue
                supplied = tuple(tuple(side) for side in supplied_edges)
            for material_id, source in variant_sources.items():
                old_tile = textures[source["old_texture_id"]]
                if supplied is not None and old_tile.tile_edges is not None and supplied != old_tile.tile_edges:
                    errors.append(
                        f"{label}: 边缘对应不合法: 更新分片边缘标签与原分片 {material_name!r} 不一致"
                    )
                if supplied is not None and old_tile.tile_edges is None:
                    errors.append(f"{label}: 边缘对应不合法: 原分片没有边缘标签")
            parsed.append({
                "label": label,
                "material_name": material_name,
                "usage": usage,
                "position": (column, row),
                "file": file_name,
                "mime_type": mime_type,
                "payload": payload,
                "variant_sources": variant_sources,
                "supplied_edges": supplied,
            })

        # Cross-entry checks: the update must be a uniform integer downscale
        # within each (material, usage) batch, across every current variant.
        ratio_groups: dict[tuple[str, str], tuple[int, int]] = {}
        for item in parsed:
            group_key = (item["material_name"], item["usage"])
            scale: tuple[int, int] | None = None
            for source in item["variant_sources"].values():
                old_width, old_height = source["old_size"]
                new_width, new_height = source["new_size"]
                if (
                    new_width > old_width or new_height > old_height
                    or old_width * new_height != old_height * new_width
                    or old_width % new_width != 0 or old_height % new_height != 0
                ):
                    errors.append(
                        f"{item['label']}: 尺寸关系不合法: {new_width}x{new_height} 不是 "
                        f"{old_width}x{old_height} 的整数倍缩小"
                    )
                    continue
                candidate = (old_width // new_width, old_height // new_height)
                if scale is None:
                    scale = candidate
                elif scale != candidate:
                    errors.append(
                        f"{item['label']}: 分辨率比例不一致: 同一更新批次中比例必须统一"
                    )
            if scale is not None:
                previous = ratio_groups.get(group_key)
                if previous is not None and previous != scale:
                    errors.append(
                        f"{item['label']}: 分辨率比例不一致: {group_key[1]} 用途已有 1:{previous[0]},"
                        f" 本分片为 1:{scale[0]}"
                    )
                else:
                    ratio_groups[group_key] = scale

        node_filter = plan.get("nodes")
        selected_nodes: set[str] | None = None
        if node_filter is not None:
            if not isinstance(node_filter, list) or not all(isinstance(item, str) for item in node_filter):
                errors.append("nodes 必须是节点名称或节点 id 列表")
            else:
                selected_nodes = self._resolve_node_names(nodes, node_filter, "nodes", errors)
        excluded = plan.get("exclude_nodes", [])
        excluded_nodes: set[str] = set()
        if not isinstance(excluded, list) or not all(isinstance(item, str) for item in excluded):
            errors.append("exclude_nodes 必须是节点名称或节点 id 列表")
        else:
            excluded_nodes = self._resolve_node_names(nodes, excluded, "exclude_nodes", errors)
        if selected_nodes is not None:
            selected_nodes -= excluded_nodes
        if errors:
            return self._invalid_tile_report(asset_id, errors), None

        root_asset_id = self._tile_base_chain(asset_id, self._load_index().get("tile_versions", {}), [])
        return self._build_tile_version(
            asset_id=asset_id,
            closure=closure,
            root_asset_id=root_asset_id,
            parsed=parsed,
            textures=textures,
            images=images,
            materials=materials,
            meshes=meshes,
            nodes=nodes,
            scenes=scenes,
            base_asset=base_asset,
            plan_path=plan_path,
            selected_nodes=selected_nodes,
        )

    @staticmethod
    def _plan_kind(plan_path: Path | str) -> str:
        try:
            plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ImportErrorBundle([f"{plan_path}: unreadable or invalid JSON: {exc}"]) from exc
        if not isinstance(plan, dict):
            raise ImportErrorBundle([f"{plan_path}: plan must be a JSON object"])
        if "tiles" in plan or "revert" in plan:
            return "tiles"
        return "retarget"

    @staticmethod
    def _invalid_tile_report(asset_id: str, errors: list[str]) -> dict[str, object]:
        return {"asset_id": asset_id, "valid": False, "errors": errors, "updates": [], "regions": [], "kept_tiles": []}

    @staticmethod
    def _resolve_node_names(
        nodes: dict[str, Node], wanted: list[str], label: str, errors: list[str]
    ) -> set[str]:
        resolved: set[str] = set()
        for name in wanted:
            if name in nodes:
                resolved.add(name)
                continue
            matches = {oid for oid, node in nodes.items() if node.name == name}
            if not matches:
                errors.append(f"{label}: 找不到节点 {name!r}")
                continue
            resolved.update(matches)
        return resolved

    @staticmethod
    def _tile_at(
        tile_texture_ids: tuple[str, ...], textures: dict[str, Texture], column: int, row: int
    ) -> str | None:
        for texture_id in tile_texture_ids:
            texture = textures.get(texture_id)
            if texture is not None and texture.tile_position == (column, row):
                return texture_id
        return None

    def _tile_base_chain(
        self, asset_id: str, tile_versions: dict[str, object], errors: list[str]
    ) -> str:
        current = asset_id
        seen: set[str] = set()
        while current in tile_versions and current not in seen:
            seen.add(current)
            record = tile_versions[current]
            if not isinstance(record, dict) or "root_asset_id" not in record:
                errors.append(f"{current}: 分片版本溯源记录损坏")
                break
            current = record["root_asset_id"]
        return current

    def _build_tile_version(
        self,
        *,
        asset_id: str,
        closure: dict[str, object],
        root_asset_id: str,
        parsed: list[dict[str, object]],
        textures: dict[str, Texture],
        images: dict[str, Image],
        materials: dict[str, Material],
        meshes: dict[str, Mesh],
        nodes: dict[str, Node],
        scenes: dict[str, Scene],
        base_asset: Asset,
        plan_path: Path,
        selected_nodes: set[str] | None = None,
        revert_nodes: set[str] | None = None,
    ) -> tuple[dict[str, object], tuple[Asset, dict[str, object], dict[str, object]] | None]:
        new_objects: dict[str, object] = {}
        material_cache: dict[tuple[str, tuple[tuple[str, str, int, int], ...]], str] = {}
        root_materials: dict[str, Material] = {}
        if revert_nodes:
            root_closure = self._closure(root_asset_id)
            root_materials = {oid: obj for oid, obj in root_closure.items() if isinstance(obj, Material)}

        def updated_material(material_id: str, items: list[dict[str, object]]) -> str:
            signature = tuple(
                (item["material_name"], item["usage"], item["position"][0], item["position"][1])
                for item in items
            )
            cache_key = (material_id, signature)
            if cache_key in material_cache:
                return material_cache[cache_key]
            material = materials[material_id]
            slots = list(material.tile_textures)
            for item in items:
                source = item["variant_sources"][material_id]
                old_texture = textures[source["old_texture_id"]]
                width, height = source["new_size"]
                new_image = Image(
                    Path(str(item["file"])).name,
                    item["mime_type"],
                    item["payload"],
                )
                new_objects[new_image.id()] = new_image
                new_texture = Texture(
                    old_texture.name,
                    new_image.id(),
                    old_texture.usage,
                    old_texture.tex_coord,
                    old_texture.tile_position,
                    old_texture.tile_layout,
                    old_texture.tile_edges,
                )
                new_objects[new_texture.id()] = new_texture
                source["new_texture_id"] = new_texture.id()
                source["new_image_id"] = new_image.id()
                slot_index = next(
                    index for index, (usage, _) in enumerate(slots) if usage == item["usage"]
                )
                tile_ids = list(slots[slot_index][1])
                tile_index = tile_ids.index(source["old_texture_id"])
                tile_ids[tile_index] = new_texture.id()
                slots[slot_index] = (item["usage"], tuple(tile_ids))
            updated = replace(material, tile_textures=tuple(slots))
            if updated.id() != material_id:
                new_objects[updated.id()] = updated
            material_cache[cache_key] = updated.id()
            return updated.id()

        affected_materials: dict[str, list[dict[str, object]]] = {}
        for item in parsed:
            for material_id in item["variant_sources"]:
                affected_materials.setdefault(material_id, []).append(item)

        def node_is_selected(node_id: str) -> bool:
            return selected_nodes is None or node_id in selected_nodes

        # Surface regions per updated tile, for the preview and provenance.
        regions: list[dict[str, object]] = []
        node_mesh_override: dict[str, str] = {}
        changed_node_ids: set[str] = set()
        revert_nodes = revert_nodes or set()
        for node_id, node in nodes.items():
            if not node.mesh_id or node.mesh_id not in meshes:
                continue
            mesh = meshes[node.mesh_id]
            new_bindings: list[tuple[str, int, int]] = []
            changed = False
            reverted = node_id in revert_nodes
            for material_id, face_offset, face_count in mesh.material_bindings:
                target_id = material_id
                material = materials.get(material_id)
                if reverted and material is not None:
                    root_match = next(
                        (oid for oid, candidate in root_materials.items() if candidate.name == material.name),
                        None,
                    )
                    if root_match is not None and root_match != material_id:
                        target_id = root_match
                        changed = True
                elif not reverted and material_id in affected_materials and node_is_selected(node_id):
                    items = affected_materials[material_id]
                    candidate = updated_material(material_id, items)
                    if candidate != material_id:
                        target_id = candidate
                        changed = True
                        changed_node_ids.add(node_id)
                new_bindings.append((target_id, face_offset, face_count))
                for item in affected_materials.get(material_id, []):
                    if reverted or not node_is_selected(node_id):
                        continue
                    source = item["variant_sources"][material_id]
                    regions.append({
                        "node_id": node_id,
                        "node_name": node.name,
                        "mesh_id": node.mesh_id,
                        "mesh_name": mesh.name,
                        "material_id": material_id,
                        "material_name": item["material_name"],
                        "face_offset": face_offset,
                        "face_count": face_count,
                        "usage": item["usage"],
                        "position": list(item["position"]),
                        "old_texture_id": source["old_texture_id"],
                        "old_image_id": source["old_image_id"],
                    })
            if changed:
                new_mesh = Mesh(mesh.name, mesh.geometry_id, tuple(new_bindings))
                new_objects[new_mesh.id()] = new_mesh
                node_mesh_override[node_id] = new_mesh.id()

        update_summaries: list[dict[str, object]] = []
        for item in parsed:
            first_source = next(iter(item["variant_sources"].values()))
            old_texture = textures[first_source["old_texture_id"]]
            neighbor_tags: dict[str, list[str]] = {}
            if old_texture.tile_edges is not None:
                column, row = item["position"]
                layout = old_texture.tile_layout
                assert layout is not None
                for side, dcolumn, drow in ((1, 1, 0), (0, -1, 0), (2, 0, -1), (3, 0, 1)):
                    neighbor = (column + dcolumn, row + drow)
                    if 0 <= neighbor[0] < layout[0] and 0 <= neighbor[1] < layout[1]:
                        neighbor_tags[f"{neighbor[0]},{neighbor[1]}"] = list(old_texture.tile_edges[side])
            variant_identities = [
                {
                    "material_id": material_id,
                    "new_texture_id": source.get("new_texture_id"),
                    "new_image_id": source.get("new_image_id"),
                }
                for material_id, source in item["variant_sources"].items()
            ]
            summary = {
                "material": item["material_name"],
                "usage": item["usage"],
                "position": list(item["position"]),
                "file": item["file"],
                "old_size": list(first_source["old_size"]),
                "new_size": list(first_source["new_size"]),
                "mime_type": item["mime_type"],
                "edge_tags": [list(edge) for edge in old_texture.tile_edges] if old_texture.tile_edges else None,
                "neighbor_edges": neighbor_tags,
                "affected_nodes": sorted(
                    node_id for node_id in changed_node_ids
                    if any(
                        bound_material in item["variant_sources"]
                        for bound_material, _, _ in meshes[nodes[node_id].mesh_id].material_bindings
                    )
                ),
            }
            summary["variant_identities"] = variant_identities
            update_summaries.append(summary)

        # Tiles that remain on their old source: explicitly enumerated so a
        # partial update can never be read as blank replacement images.
        kept_tiles: list[dict[str, object]] = []
        updated_keys = {
            (item["material_name"], item["usage"], item["position"]) for item in parsed
        }
        adopted_materials: set[str] = set()
        for node_id in changed_node_ids:
            override_id = node_mesh_override[node_id]
            new_mesh = new_objects[override_id]
            assert isinstance(new_mesh, Mesh)
            adopted_materials.update(material_id for material_id, _, _ in new_mesh.material_bindings)
        material_namespace: dict[str, set[str]] = {}
        for material_id, material in materials.items():
            material_namespace.setdefault(material.name, set()).add(material_id)
        touched_names = {item["material_name"] for item in parsed}
        for material_id, material in materials.items():
            if material.name in touched_names or revert_nodes:
                pass
            elif selected_nodes is None:
                continue
            elif not (material_namespace[material.name] & adopted_materials):
                continue
            for usage, tile_ids in material.tile_textures:
                for tile_id in tile_ids:
                    tile = textures[tile_id]
                    key = (material.name, usage, tile.tile_position)
                    if material_id in adopted_materials and key in updated_keys and material_id in affected_materials:
                        continue
                    kept_tiles.append({
                        "material": material.name,
                        "material_id": material_id,
                        "usage": usage,
                        "position": list(tile.tile_position) if tile.tile_position else None,
                        "texture_id": tile_id,
                        "image_id": tile.image_id,
                    })

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

        tile_entries_record: list[dict[str, object]] = []
        for item in parsed:
            variants = []
            for material_id, source in item["variant_sources"].items():
                variants.append({
                    "material_id": material_id,
                    "old_texture_id": source["old_texture_id"],
                    "old_image_id": source["old_image_id"],
                })
            tile_entries_record.append({
                "material": item["material_name"],
                "usage": item["usage"],
                "position": list(item["position"]),
                "file": item["file"],
                "variants": variants,
            })
        provenance = {
            "kind": "tile-version",
            "base_asset_id": asset_id,
            "root_asset_id": root_asset_id,
            "nodes": sorted(selected_nodes) if selected_nodes is not None else None,
            "revert_nodes": sorted(revert_nodes),
            "tiles": tile_entries_record,
            "kept_tiles": kept_tiles,
        }
        record = {"plan": f"tile-version:{plan_path.name}", "provenance": provenance}
        node_name_by_id = {node_id: node.name for node_id, node in nodes.items()}
        for region in regions:
            region["node_name"] = node_name_by_id[region["node_id"]]
        report = {
            "asset_id": asset_id,
            "valid": True,
            "errors": [],
            "updates": update_summaries,
            "regions": sorted(regions, key=lambda entry: (entry["node_id"], entry["usage"], entry["position"])),
            "kept_tiles": kept_tiles,
            "reverted_nodes": sorted(node_name_by_id[node_id] for node_id in revert_nodes),
            "new_asset_id": new_asset.id(),
        }
        return report, (new_asset, new_objects, record)

    def _prepare_retarget(
        self, asset_id: str, plan_path: Path | str
    ) -> tuple[dict[str, object], tuple[Asset, dict[str, object], dict[str, object]] | None]:
        plan_path = Path(plan_path)
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ImportErrorBundle([f"{plan_path}: unreadable or invalid JSON: {exc}"]) from exc
        if not isinstance(plan, dict) or not isinstance(plan.get("replacements"), list):
            raise ImportErrorBundle([f"{plan_path}: plan must be an object with a replacements list"])

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
                for _usage, texture_ids in obj.tile_textures:
                    for texture_id in texture_ids:
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
        if not {"version", "assets", "objects"} <= set(data) <= {
            "version", "assets", "objects", "retargets", "tile_versions",
        }:
            raise ValueError("invalid library index")
        return data
