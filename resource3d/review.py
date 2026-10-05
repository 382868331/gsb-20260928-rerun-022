from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .errors import ImportErrorBundle
from .hashing import canonical_json, sha256_bytes
from .models import Image, Material, Mesh, Node, TiledTexture
from .retarget import image_dimensions

MIME_BY_SUFFIX = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}

# Decision states stored in the review record:
#   selected: the operator adopted this candidate for its scope
#   deferred: the operator explicitly leaves this scope on the old image
#   undecided: pending (default when a review is (re)opened)
DECISION_STATES = ("undecided", "selected", "deferred")


@dataclass(frozen=True)
class TileCandidate:
    """One reviewable candidate: a tile update entry acting on one object.

    The same plan entry may target many objects; review decisions are made
    per object ("片区"/object scope), so each (entry, node) pair becomes its
    own candidate.
    """

    candidate_id: str
    position: int
    label: str
    tiled_texture_id: str
    tiled_texture_name: str
    node_id: str
    node_name: str
    tiles: tuple[dict[str, object], ...]
    priority: int
    withdraw: bool


def _tile_candidate_id(
    position: int,
    tiled_texture_id: str,
    node_id: str,
    tile_ids: tuple[str, ...],
) -> str:
    digest = sha256_bytes(
        canonical_json({
            "position": position,
            "tiled_texture_id": tiled_texture_id,
            "node_id": node_id,
            "tiles": tile_ids,
        })
    )
    return f"rc_{digest[:24]}"


def scan_tile_candidates(
    asset_id: str,
    closure: dict[str, object],
    plan_path: Path,
    plan: dict[str, object],
) -> tuple[list[TileCandidate], list[str]]:
    """Resolve a tile_updates plan into per-object review candidates.

    Returns the candidates plus structural/semantic errors.  File-level
    problems (missing file, bad size/MIME) do not abort the scan: the
    offending candidate is recorded with an ``error`` field so the review
    record can still be saved and reopened later.
    """

    tiled = {oid: obj for oid, obj in closure.items() if isinstance(obj, TiledTexture)}
    materials = {oid: obj for oid, obj in closure.items() if isinstance(obj, Material)}
    meshes = {oid: obj for oid, obj in closure.items() if isinstance(obj, Mesh)}
    nodes = {oid: obj for oid, obj in closure.items() if isinstance(obj, Node)}
    images = {oid: obj for oid, obj in closure.items() if isinstance(obj, Image)}

    def tiled_ref(ref: object) -> str | None:
        if isinstance(ref, str) and ref in tiled:
            return ref
        if isinstance(ref, str):
            matches = [oid for oid, value in tiled.items() if value.name == ref]
            if len(matches) == 1:
                return matches[0]
        return None

    errors: list[str] = []
    candidates: list[TileCandidate] = []
    raw_updates = plan.get("tile_updates")
    if not isinstance(raw_updates, list):
        raise ImportErrorBundle([f"{plan_path}: plan must contain a tile_updates list"])

    for position, entry in enumerate(raw_updates):
        label = f"tile_updates[{position}]"
        if not isinstance(entry, dict):
            errors.append(f"{label}: entry must be an object")
            continue
        texture_id = tiled_ref(entry.get("tiled_texture"))
        if texture_id is None:
            errors.append(f"{label}: 图像缺失: 资产中找不到分片贴图 {entry.get('tiled_texture')!r}")
            continue
        old_tiled = tiled[texture_id]
        priority = entry.get("priority", 0)
        if not isinstance(priority, int) or isinstance(priority, bool):
            errors.append(f"{label}: priority 必须是整数")
            continue
        node_filter = entry.get("nodes")
        if node_filter is not None and not isinstance(node_filter, list):
            errors.append(f"{label}: nodes 必须是节点名称或 ID 列表")
            continue

        consumer_nodes = sorted(
            node_id
            for node_id, node in nodes.items()
            if node.mesh_id in meshes
            and any(
                texture_id in materials[material_id].tiled_texture_ids.values()
                for material_id, _, _ in meshes[node.mesh_id].material_bindings
                if material_id in materials
            )
        )
        filter_matches: set[str] = set()
        if node_filter is not None:
            for wanted in node_filter:
                matches = [oid for oid, node in nodes.items() if node.name == wanted or oid == wanted]
                if not matches:
                    errors.append(f"{label}: 找不到节点 {wanted!r}")
                filter_matches.update(matches)
        target_nodes = consumer_nodes if node_filter is None else sorted(filter_matches & set(consumer_nodes))

        usage = entry.get("usage", old_tiled.usage)
        usage_error = None
        if usage != old_tiled.usage:
            usage_error = f"用途不兼容: 不能把 {old_tiled.usage} 分片当 {usage} 使用"

        if entry.get("withdraw") is True:
            if not node_filter or not target_nodes:
                errors.append(f"{label}: 撤回采用必须明确选择至少一个节点")
                continue
            for node_id in target_nodes:
                candidate_id = _tile_candidate_id(position, texture_id, node_id, ())
                candidates.append(TileCandidate(
                    candidate_id=candidate_id,
                    position=position,
                    label=label,
                    tiled_texture_id=texture_id,
                    tiled_texture_name=old_tiled.name,
                    node_id=node_id,
                    node_name=nodes[node_id].name,
                    tiles=(),
                    priority=priority,
                    withdraw=True,
                ))
            continue

        raw_tiles = entry.get("tiles", {})
        if not isinstance(raw_tiles, dict):
            errors.append(f"{label}: tiles 必须是以分片 ID 为键的对象")
            continue
        old_by_id = {tile.tile_id: tile for tile in old_tiled.tiles}
        tile_entries: list[dict[str, object]] = []
        for tile_id, tile_spec in raw_tiles.items():
            tile_label = f"{label}.tiles[{tile_id}]"
            if tile_id not in old_by_id:
                errors.append(f"{tile_label}: 用途不兼容: 未知分片位置 {tile_id!r}")
                continue
            old_tile = old_by_id[tile_id]
            if tile_spec is None:
                tile_entries.append({
                    "tile_id": tile_id,
                    "file": None,
                    "file_sha256": None,
                    "image_id": old_tile.image_id,
                    "width": old_tile.width,
                    "height": old_tile.height,
                    "error": None,
                })
                continue
            file_error: str | None = None
            file_name = tile_spec.get("file") if isinstance(tile_spec, dict) else None
            if not isinstance(file_name, str):
                errors.append(f"{tile_label}: 图像缺失: 未提供替换文件")
                continue
            replacement_path = (plan_path.parent / file_name).resolve()
            try:
                payload = replacement_path.read_bytes()
            except OSError:
                file_error = f"图像缺失: 无法读取 {file_name}"
                payload = b""
            mime_type = MIME_BY_SUFFIX.get(replacement_path.suffix.lower(), "")
            old_image = images[old_tile.image_id]
            image_id = None
            new_width = new_height = None
            if file_error is None:
                if mime_type not in ("image/png", "image/jpeg") or mime_type != old_image.mime_type:
                    file_error = f"用途不兼容: 替换图像必须保持 {old_image.mime_type}"
                else:
                    try:
                        new_width, new_height = image_dimensions(mime_type, payload)
                    except ValueError as exc:
                        file_error = f"尺寸关系不合法: {exc}"
                    else:
                        if (
                            new_width > old_tile.width or new_height > old_tile.height
                            or old_tile.width * new_height != old_tile.height * new_width
                            or old_tile.width % new_width or old_tile.height % new_height
                        ):
                            file_error = (
                                f"尺寸关系不合法: {new_width}x{new_height} 不是 "
                                f"{old_tile.width}x{old_tile.height} 的等比整数缩小"
                            )
                        else:
                            image = Image(
                                f"{old_image.name}#{tile_id}@{new_width}x{new_height}",
                                mime_type, payload,
                            )
                            image_id = image.id()
            if usage_error is not None:
                file_error = usage_error
            tile_entries.append({
                "tile_id": tile_id,
                "file": file_name,
                "file_sha256": sha256_bytes(payload) if payload else None,
                "image_id": image_id,
                "width": new_width,
                "height": new_height,
                "error": file_error,
            })
        if not tile_entries and usage_error is None:
            errors.append(f"{label}: 没有任何分片发生变化")
            continue
        tile_ids = tuple(str(item["tile_id"]) for item in sorted(tile_entries, key=lambda item: str(item["tile_id"])))
        for node_id in target_nodes:
            candidate_id = _tile_candidate_id(position, texture_id, node_id, tile_ids)
            candidates.append(TileCandidate(
                candidate_id=candidate_id,
                position=position,
                label=label,
                tiled_texture_id=texture_id,
                tiled_texture_name=old_tiled.name,
                node_id=node_id,
                node_name=nodes[node_id].name,
                tiles=tuple(tile_entries),
                priority=priority,
                withdraw=False,
            ))
    return candidates, errors


def candidate_summary(candidate: TileCandidate) -> dict[str, object]:
    return {
        "candidate_id": candidate.candidate_id,
        "entry": candidate.label,
        "entry_index": candidate.position,
        "tiled_texture_id": candidate.tiled_texture_id,
        "tiled_texture_name": candidate.tiled_texture_name,
        "node_id": candidate.node_id,
        "node_name": candidate.node_name,
        "priority": candidate.priority,
        "action": "withdraw" if candidate.withdraw else "adopt",
        "tiles": [dict(tile) for tile in candidate.tiles],
    }


def find_overlaps(candidates: list[TileCandidate]) -> list[dict[str, object]]:
    """Group candidates that offer different content for one object/tile.

    Same-content offers are not conflicts.  Groups already resolved by a
    unique maximal priority are not reported either (they need no review).
    """

    offers: dict[tuple[str, str], list[TileCandidate]] = {}
    for candidate in candidates:
        if candidate.withdraw:
            continue
        for tile in candidate.tiles:
            offers.setdefault((candidate.node_id, str(tile["tile_id"])), []).append(candidate)
    overlaps: list[dict[str, object]] = []
    for (node_id, tile_id), group in sorted(offers.items()):
        distinct = {
            str(next(t["image_id"] for t in candidate.tiles if t["tile_id"] == tile_id))
            for candidate in group
        }
        if len(distinct) <= 1:
            continue
        best = max(candidate.priority for candidate in group)
        winners = [candidate for candidate in group if candidate.priority == best]
        resolved = len(winners) == 1 and all(
            not tile.get("error")
            for candidate in group for tile in candidate.tiles if tile["tile_id"] == tile_id
        )
        if resolved:
            continue
        overlaps.append({
            "node_id": node_id,
            "node_name": group[0].node_name,
            "tiled_texture_id": group[0].tiled_texture_id,
            "tile_id": tile_id,
            "candidate_ids": sorted(candidate.candidate_id for candidate in group),
            "priorities": sorted({candidate.priority for candidate in group}),
        })
    return overlaps
