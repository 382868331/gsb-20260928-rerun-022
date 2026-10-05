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
from .review import DECISION_STATES, candidate_summary, find_overlaps, scan_tile_candidates
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

    def review_tile_plan(
        self,
        asset_id: str,
        plan_path: Path | str,
        review_path: Path | str | None,
        decisions: dict[str, object] | None = None,
    ) -> tuple[dict[str, object], dict[str, object]]:
        """Open/update an independent review record for a tile update plan.

        The review record lives next to (but separate from) the plan.  It
        preserves unresolved overlaps, selected candidates and explicitly
        deferred scopes so the operator can stop and resume reviewing.  When
        a candidate file changes after a decision was saved, only the
        decisions that depend on that file are marked ``needs_recheck``;
        every other decision and pending item is retained.
        """

        plan_path = Path(plan_path)
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ImportErrorBundle([f"{plan_path}: unreadable or invalid JSON: {exc}"]) from exc
        if not isinstance(plan, dict) or not isinstance(plan.get("tile_updates"), list):
            raise ImportErrorBundle([f"{plan_path}: plan must contain a tile_updates list"])
        plan_sha = sha256_bytes(canonical_json(plan))
        closure = self._closure(asset_id)
        candidates, errors = scan_tile_candidates(asset_id, closure, plan_path, plan)
        # File-level candidate problems are recorded per tile so the review
        # record can still be saved; they still make the report invalid and
        # block publication.
        seen_errors: set[str] = set()
        for candidate in candidates:
            for tile in candidate.tiles:
                if tile.get("error"):
                    message = (
                        f"{candidate.label}.tiles[{tile['tile_id']}] @ {candidate.node_name}: {tile['error']}"
                    )
                    if message not in seen_errors:
                        seen_errors.add(message)
                        errors.append(message)

        saved: dict[str, object] = {}
        if review_path is not None and Path(review_path).exists():
            try:
                saved = json.loads(Path(review_path).read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ImportErrorBundle([f"{review_path}: 无法读取审阅记录: {exc}"]) from exc
            if not isinstance(saved, dict) or saved.get("asset_id") != asset_id:
                raise ImportErrorBundle([f"{review_path}: 审阅记录不属于资产 {asset_id}"])
        saved_decisions = {
            item["candidate_id"]: item
            for item in saved.get("decisions", [])
            if isinstance(item, dict) and isinstance(item.get("candidate_id"), str)
        } if isinstance(saved.get("decisions"), list) else {}

        current_file_shas: dict[str, str] = {}
        invalid_candidate_ids: set[str] = set()
        for candidate in candidates:
            for tile in candidate.tiles:
                if tile["error"] is not None:
                    invalid_candidate_ids.add(candidate.candidate_id)
                if tile["file"] is not None and tile["file_sha256"] is not None:
                    current_file_shas[str(tile["file"])] = str(tile["file_sha256"])

        merged: dict[str, dict[str, object]] = {}
        # Carry over every previously saved decision, keyed by the content
        # identity captured when it was saved.
        for candidate_id, prior in saved_decisions.items():
            merged[candidate_id] = dict(prior)
        incoming: dict[str, str] = {}
        if decisions:
            raw = decisions.get("decisions")
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict) and isinstance(item.get("candidate_id"), str):
                        state = item.get("state")
                        if state not in DECISION_STATES:
                            errors.append(f"review: 未知审阅状态 {state!r}")
                            continue
                        incoming[str(item["candidate_id"])] = str(state)
        by_id = {candidate.candidate_id: candidate for candidate in candidates}

        for candidate in candidates:
            prior = merged.get(candidate.candidate_id)
            current_scope = {
                "tiled_texture_id": candidate.tiled_texture_id,
                "tiled_texture_name": candidate.tiled_texture_name,
                "node_id": candidate.node_id,
                "node_name": candidate.node_name,
                "action": "withdraw" if candidate.withdraw else "adopt",
                "entry_index": candidate.position,
                "priority": candidate.priority,
                "tiles": [
                    {
                        "tile_id": tile["tile_id"],
                        "file": tile["file"],
                        "file_sha256": tile["file_sha256"],
                        "image_id": tile["image_id"],
                    }
                    for tile in candidate.tiles
                ],
            }
            entry: dict[str, object] = {
                "candidate_id": candidate.candidate_id,
                "state": "undecided",
                "needs_recheck": False,
                "note": None,
                "scope": current_scope,
            }
            if prior is not None:
                entry["state"] = prior.get("state", "undecided")
                entry["note"] = prior.get("note")
                # The recorded scope keeps the identity the operator last
                # confirmed; the current scan identity stays separate so a
                # later reopen keeps flagging the decision until the operator
                # explicitly reconfirms it.  Other entries and pending items
                # are untouched.
                prior_scope = prior.get("scope")
                if isinstance(prior_scope, dict):
                    entry["scope"] = prior_scope
                entry["current_scope"] = current_scope
                changed_files = []
                for tile in candidate.tiles:
                    if tile["file"] is None:
                        continue
                    prior_sha = next(
                        (
                            saved_tile.get("file_sha256")
                            for saved_tile in (prior_scope.get("tiles", []) if isinstance(prior_scope, dict) else [])
                            if isinstance(saved_tile, dict) and saved_tile.get("tile_id") == tile["tile_id"]
                        ),
                        None,
                    )
                    if prior_sha is not None and prior_sha != tile["file_sha256"]:
                        changed_files.append(str(tile["file"]))
                if changed_files and entry["state"] != "undecided":
                    entry["needs_recheck"] = True
                    entry["changed_files"] = changed_files
                elif candidate.candidate_id in invalid_candidate_ids and entry["state"] != "undecided":
                    entry["needs_recheck"] = True
                elif prior.get("needs_recheck") and entry["state"] != "undecided":
                    entry["needs_recheck"] = True
                    entry["changed_files"] = prior.get("changed_files")
            if candidate.candidate_id in incoming:
                entry["state"] = incoming[candidate.candidate_id]
                # A fresh operator decision re-confirms the current file
                # identity and clears the re-required-confirmation flag.
                entry["needs_recheck"] = False
                entry.pop("changed_files", None)
                entry["scope"] = current_scope
                entry.pop("current_scope", None)
            merged[candidate.candidate_id] = entry

        # Decisions whose candidate vanished from the plan are retained but
        # flagged, never silently dropped.
        for candidate_id, prior in list(merged.items()):
            if candidate_id not in by_id:
                prior["missing_from_plan"] = True

        ordered = sorted(
            merged.values(),
            key=lambda item: (
                item["scope"].get("entry_index", 1 << 30) if isinstance(item.get("scope"), dict) else 1 << 30,
                str(item.get("scope", {}).get("node_id", "")) if isinstance(item.get("scope"), dict) else "",
                str(item.get("candidate_id", "")),
            ),
        )
        overlap_groups = find_overlaps(candidates)
        decided_candidate_ids = {
            str(item["candidate_id"])
            for item in ordered
            if item.get("state") in ("selected", "deferred")
        }
        unresolved_overlaps = [
            group for group in overlap_groups
            if not set(group["candidate_ids"]).issubset(decided_candidate_ids)
        ]
        review_record = {
            "version": 1,
            "kind": "tile-update-review",
            "asset_id": asset_id,
            "plan_file": plan_path.name,
            "plan_sha256": plan_sha,
            "decisions": ordered,
            "unresolved_overlaps": unresolved_overlaps,
        }
        report = {
            "asset_id": asset_id,
            "plan_file": plan_path.name,
            "plan_sha256": plan_sha,
            "candidates": [candidate_summary(candidate) for candidate in candidates],
            "decisions": ordered,
            "overlaps": overlap_groups,
            "unresolved_overlaps": unresolved_overlaps,
            "needs_recheck": [
                str(item["candidate_id"]) for item in ordered if item.get("needs_recheck")
            ],
            "errors": errors,
            "valid": not errors,
        }
        return report, review_record

    def save_tile_review(
        self,
        asset_id: str,
        plan_path: Path | str,
        review_path: Path | str,
        decisions: dict[str, object] | None = None,
        save_as: Path | str | None = None,
    ) -> dict[str, object]:
        """Refresh and persist a review record without publishing assets."""

        report, review_record = self.review_tile_plan(asset_id, plan_path, review_path, decisions)
        if save_as is not None:
            # Save-as produces an independent copy: the incoming decisions
            # belong to the copy only, and the original record (including
            # its own human decisions) is left untouched on disk.
            copy_path = Path(save_as)
            self._atomic_write(copy_path, canonical_json(review_record) + b"\n")
            report["review_file"] = str(copy_path)
            report["saved_as"] = str(copy_path)
            return report
        target = Path(review_path)
        payload = canonical_json(review_record) + b"\n"
        self._atomic_write(target, payload)
        report["review_file"] = str(target)
        return report

    def apply_reviewed_tile_plan(
        self,
        asset_id: str,
        plan_path: Path | str,
        review_path: Path | str,
        commit: bool = False,
    ) -> tuple[dict[str, object], tuple[str, dict[str, object]] | None]:
        """Build the effective tile update from review selections.

        Only ``selected`` candidates are published; ``deferred`` scopes stay
        on their old images and ``undecided`` scopes cannot be published while
        overlaps remain.  The review record itself never publishes assets.
        """

        report, review_record = self.review_tile_plan(asset_id, plan_path, review_path)
        errors: list[str] = list(report["errors"])
        needs_recheck = [
            item for item in review_record["decisions"]
            if item.get("needs_recheck") and item.get("state") == "selected"
        ]
        for item in needs_recheck:
            errors.append(
                f"review: 选择 {item['candidate_id']} 依赖的候选文件已变化，需要重新确认后才能发布"
            )
        if report["unresolved_overlaps"]:
            for group in report["unresolved_overlaps"]:
                errors.append(
                    f"{group['node_name']}: 分片 {group['tile_id']!r} 仍有未解决的候选重叠"
                )
        if errors:
            return {**report, "valid": False, "errors": errors}, None

        plan_path = Path(plan_path)
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        selected = [
            item for item in review_record["decisions"]
            if item.get("state") == "selected" and not item.get("missing_from_plan")
        ]
        # Group selections back by plan entry index and selected node set.
        selections_by_entry: dict[int, list[dict[str, object]]] = {}
        for item in selected:
            scope = item["scope"]
            selections_by_entry.setdefault(int(scope["entry_index"]), []).append(scope)
        effective_entries: list[dict[str, object]] = []
        chosen_positions: dict[tuple[str, str], int] = {}
        for position, scopes in sorted(selections_by_entry.items()):
            source_entry = plan["tile_updates"][position]
            node_ids = sorted({str(scope["node_id"]) for scope in scopes})
            effective_entries.append({**source_entry, "nodes": node_ids})
            for scope in scopes:
                for tile in scope["tiles"]:
                    if scope["action"] == "withdraw":
                        continue
                    key = (str(scope["node_id"]), str(tile["tile_id"]))
                    if key in chosen_positions and chosen_positions[key] != position:
                        errors.append(
                            f"review: 对象 {scope['node_name']} 的分片 {tile['tile_id']!r} "
                            "被多个已选候选同时采用"
                        )
                    chosen_positions[key] = position
        if not effective_entries:
            errors.append("review: 没有任何已选择的候选可发布")
        if errors:
            return {**report, "valid": False, "errors": errors}, None

        effective_plan = {
            **{key: value for key, value in plan.items() if key != "tile_updates"},
            "tile_updates": effective_entries,
        }
        fd, temporary_name = tempfile.mkstemp(prefix=".reviewed-plan-", suffix=".json", dir=plan_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(effective_plan, ensure_ascii=False))
            effective_path = Path(temporary_name)
            try:
                if commit:
                    new_asset_id, apply_report = self.apply_retarget(asset_id, effective_path)
                    apply_report["review_file"] = str(review_path)
                    return apply_report, (new_asset_id, apply_report)
                preview = self.plan_retarget(asset_id, effective_path)
                preview["review_file"] = str(review_path)
                return preview, None
            finally:
                try:
                    effective_path.unlink()
                except FileNotFoundError:
                    pass
        except BaseException:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
            raise

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
            if not isinstance(priority, int) or isinstance(priority, bool):
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
                    "old_tiled": tiled[texture_id],
                    "requested": {},
                    "target_nodes": target_nodes,
                    "withdraw": True,
                    "priority": priority,
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
                "old_tiled": old_tiled,
                "requested": requested,
                "target_nodes": target_nodes,
                "explicit_tiles": explicit_tiles,
                "priority": priority,
            })
        if errors:
            return {"asset_id": asset_id, "valid": False, "errors": errors, "changes": [], "missing_areas": []}, None

        new_objects: dict[str, object] = {}
        # (node_id, old_tiled_texture_id) -> replacement tiled texture id
        node_texture_override: dict[tuple[str, str], str] = {}
        summaries: list[dict[str, object]] = []
        candidate_reports: list[dict[str, object]] = []
        requests_by_texture: dict[str, list[dict[str, object]]] = {}
        for request in requests:
            grouped_old = request["old_tiled"]
            assert isinstance(grouped_old, TiledTexture)
            requests_by_texture.setdefault(grouped_old.id(), []).append(request)

        for old_texture_id, group in requests_by_texture.items():
            old_tiled = tiled[old_texture_id]
            withdraws = [request for request in group if request.get("withdraw")]
            if withdraws:
                if len(group) > 1:
                    errors.append(f"{withdraws[0]['label']}: 撤回不能与其他候选同时作用于同一分片贴图")
                    continue
                request = withdraws[0]
                source_refs = {tile.source_texture_id for tile in old_tiled.tiles if tile.source_texture_id}
                if len(source_refs) != 1:
                    errors.append(f"{request['label']}: 无法从来源记录恢复原始分片贴图")
                    continue
                original_id = next(iter(source_refs))
                try:
                    source_texture = self.load_object(original_id)
                except KeyError:
                    source_texture = None
                if not isinstance(source_texture, TiledTexture):
                    errors.append(f"{request['label']}: 无法从来源记录恢复原始分片贴图")
                    continue
                for node_id in sorted(request["target_nodes"]):
                    node_texture_override[(node_id, old_texture_id)] = original_id
                summaries.append({
                    "old_tiled_texture_id": old_tiled.id(),
                    "new_tiled_texture_id": original_id,
                    "usage": old_tiled.usage,
                    "atlas_size": [old_tiled.atlas_width, old_tiled.atlas_height],
                    "nodes": sorted(request["target_nodes"]),
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
                candidate_reports.append({
                    "label": request["label"],
                    "tiled_texture_id": old_tiled.id(),
                    "action": "withdraw",
                    "priority": request["priority"],
                    "nodes": sorted(request["target_nodes"]),
                    "tiles": [],
                })
                continue
            for request in group:
                if request.get("explicit_tiles", 0) == 0:
                    errors.append(f"{request['label']}: 没有任何分片发生变化")
            if errors:
                continue
            consumer_nodes = sorted(
                node_id
                for node_id, node in nodes.items()
                if node.mesh_id in meshes
                and any(
                    old_texture_id in materials[material_id].tiled_texture_ids.values()
                    for material_id, _, _ in meshes[node.mesh_id].material_bindings
                    if material_id in materials
                )
            )
            # 并行候选：每个候选声明自己影响的对象集合，缺省为所有引用该贴图的对象
            effective: list[tuple[dict[str, object], set[str]]] = []
            for request in group:
                targets = request["target_nodes"]
                assert targets is None or isinstance(targets, set)
                effective.append((request, set(consumer_nodes) if targets is None else set(targets) & set(consumer_nodes)))
            adopted_by_candidate: dict[int, dict[str, list[str]]] = {}
            node_choice: dict[str, dict[str, tuple[TextureTile, Image | None]]] = {}
            for node_id in consumer_nodes:
                applicable = [
                    (index, request)
                    for index, (request, target_set) in enumerate(effective)
                    if node_id in target_set
                ]
                if not applicable:
                    continue
                chosen: dict[str, tuple[TextureTile, Image | None]] = {}
                overlapped = False
                for old_tile in old_tiled.tiles:
                    offers = [
                        (index, request) for index, request in applicable
                        if old_tile.tile_id in request["requested"]
                    ]
                    if not offers:
                        continue
                    distinct_images = {
                        request["requested"][old_tile.tile_id].image_id for _, request in offers
                    }
                    if len(distinct_images) > 1:
                        best = max(request["priority"] for _, request in offers)
                        winners = [(index, request) for index, request in offers if request["priority"] == best]
                        if len(winners) != 1:
                            errors.append(
                                f"{old_tiled.name}: 分片 {old_tile.tile_id!r} 在同一位置有多个候选重叠，"
                                "需要用明确的 priority 解决"
                            )
                            overlapped = True
                            break
                        offers = winners
                    index, request = offers[0]
                    requested = request["requested"]
                    assert isinstance(requested, dict)
                    tile = requested[old_tile.tile_id]
                    assert isinstance(tile, TextureTile)
                    image = requested.get(old_tile.tile_id + "__image")
                    chosen[old_tile.tile_id] = (tile, image if isinstance(image, Image) else None)
                    adopted_by_candidate.setdefault(index, {}).setdefault(old_tile.tile_id, []).append(node_id)
                if overlapped:
                    break
                if chosen:
                    node_choice[node_id] = chosen
            if errors:
                continue
            # 组合采用：结果相同的对象共享同一个新分片贴图身份，不同对象可分别保持旧引用
            combos: dict[tuple[tuple[str, str], ...], dict[str, object]] = {}
            for node_id, chosen in node_choice.items():
                key = tuple(sorted((tile_id, tile.image_id) for tile_id, (tile, _) in chosen.items()))
                combo = combos.setdefault(key, {"chosen": chosen, "nodes": []})
                combo["nodes"].append(node_id)
            produced = 0
            for combo in combos.values():
                chosen = combo["chosen"]
                assert isinstance(chosen, dict)
                new_tiles = []
                tile_summary = []
                retained = []
                changed = False
                for old_tile in old_tiled.tiles:
                    if old_tile.tile_id in chosen:
                        tile, image = chosen[old_tile.tile_id]
                        if image is not None:
                            new_objects[image.id()] = image
                        if tile.image_id != old_tile.image_id:
                            changed = True
                        tile_summary.append({
                            "tile_id": tile.tile_id,
                            "x": tile.x,
                            "y": tile.y,
                            "width": tile.width,
                            "height": tile.height,
                            "old_image_id": old_tile.image_id,
                            "new_image_id": tile.image_id,
                            "old_size": [old_tile.width, old_tile.height],
                            "new_size": list(image_dimensions(image.mime_type, image.payload)) if image is not None else [tile.width, tile.height],
                            "state": "updated",
                        })
                    else:
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
                    new_tiles.append(tile)
                if not changed:
                    continue
                new_tiled = TiledTexture(
                    f"{old_tiled.name}#partial", old_tiled.usage, tuple(new_tiles),
                    old_tiled.atlas_width, old_tiled.atlas_height, old_tiled.tex_coord,
                )
                new_objects[new_tiled.id()] = new_tiled
                for node_id in combo["nodes"]:
                    node_texture_override[(node_id, old_texture_id)] = new_tiled.id()
                summaries.append({
                    "old_tiled_texture_id": old_tiled.id(),
                    "new_tiled_texture_id": new_tiled.id(),
                    "usage": old_tiled.usage,
                    "atlas_size": [old_tiled.atlas_width, old_tiled.atlas_height],
                    "nodes": sorted(combo["nodes"]),
                    "tiles": tile_summary,
                    "retained_tile_ids": retained,
                    "edges": self._tile_edges(new_tiled),
                    "action": "adopt",
                })
                produced += 1
            if produced == 0:
                errors.append(f"{group[0]['label']}: 没有任何分片发生变化")
                continue
            for index, (request, target_set) in enumerate(effective):
                requested = request["requested"]
                assert isinstance(requested, dict)
                adopted = adopted_by_candidate.get(index, {})
                candidate_reports.append({
                    "label": request["label"],
                    "tiled_texture_id": old_tiled.id(),
                    "action": "adopt",
                    "priority": request["priority"],
                    "nodes": sorted(target_set),
                    "tiles": [
                        {
                            "tile_id": tile_id,
                            "image_id": requested[tile_id].image_id,
                            "state": "adopted" if tile_id in adopted else "superseded",
                            "adopted_nodes": sorted(adopted.get(tile_id, [])),
                        }
                        for tile_id in sorted(key for key in requested if not key.endswith("__image"))
                    ],
                })
        if errors:
            return {"asset_id": asset_id, "valid": False, "errors": errors, "changes": [], "missing_areas": []}, None

        material_cache: dict[tuple[str, tuple[tuple[str, str], ...]], str] = {}

        def material_for(material_id: str, mapping: tuple[tuple[str, str], ...]) -> str:
            key = (material_id, mapping)
            if key in material_cache:
                return material_cache[key]
            material = materials[material_id]
            tiled_maps = dict(material.tiled_texture_ids)
            replacements = dict(mapping)
            for usage, tiled_id in list(tiled_maps.items()):
                if tiled_id in replacements:
                    tiled_maps[usage] = replacements[tiled_id]
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
            applicable: dict[str, str] = {}
            for material_id, _, _ in mesh.material_bindings:
                material = materials.get(material_id)
                if material is None:
                    continue
                for usage, old_tiled_id in material.tiled_texture_ids.items():
                    new_tiled_id = node_texture_override.get((node_id, old_tiled_id))
                    if new_tiled_id is not None:
                        applicable[old_tiled_id] = new_tiled_id
            if not applicable:
                continue
            new_bindings = []
            for material_id, offset, count in mesh.material_bindings:
                material = materials[material_id]
                chosen = material_id
                material_mapping = tuple(sorted(
                    (old_tiled_id, applicable[old_tiled_id])
                    for old_tiled_id in material.tiled_texture_ids.values()
                    if old_tiled_id in applicable
                ))
                if material_mapping:
                    chosen = material_for(material_id, material_mapping)
                    chosen_material = materials.get(chosen) or new_objects[chosen]
                    assert isinstance(chosen_material, Material)
                    for usage, old_tex_id in material.tiled_texture_ids.items():
                        new_tex_id = chosen_material.tiled_texture_ids.get(usage)
                        if old_tex_id in applicable and new_tex_id:
                            update_summary = next(
                                item for item in summaries
                                if item["old_tiled_texture_id"] == old_tex_id
                                and item["new_tiled_texture_id"] == new_tex_id
                            )
                            changes.append({
                                "node_id": node_id,
                                "node_name": node.name,
                                "mesh_id": mesh.id(),
                                "material_name": material.name,
                                "usage": usage,
                                "old_tiled_texture_id": old_tex_id,
                                "new_tiled_texture_id": new_tex_id,
                                "updated_tile_ids": [
                                    tile["tile_id"] for tile in update_summary["tiles"]
                                    if tile["state"] == "updated"
                                ],
                                "retained_tile_ids": [
                                    tile["tile_id"] for tile in update_summary["tiles"]
                                    if tile["state"] == "retained"
                                ],
                                "atlas_size": update_summary["atlas_size"],
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
            "affected_nodes": sorted({change["node_id"] for change in changes}),
            "missing_areas": missing_areas,
            "tile_updates": summaries,
            "candidates": candidate_reports,
            "new_asset_id": new_asset.id(),
        }
        record = {
            "plan": f"tile-retarget:{plan_path.name}",
            "tile_updates": summaries,
            "candidates": candidate_reports,
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
