from __future__ import annotations

import json
import shutil
import struct
import sys
import unittest
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resource3d.errors import ImportErrorBundle
from resource3d.library import ResourceLibrary


def png(width: int, height: int, seed: int) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    rows = b"".join(b"\0" + bytes([seed % 256]) * width * 3 for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def write_batch(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    positions = [0, 0, 0, 1, 0, 0, 0, 1, 0]
    blob = struct.pack("<9f6H", *positions, 0, 1, 2, 3, 4, 5)
    (directory / "mesh.bin").write_bytes(blob)
    # The same two backing images are used at different tile positions and by
    # different usages. Content remains deduplicated while tile identity is not.
    (directory / "a.png").write_bytes(png(2, 2, 1))
    (directory / "b.png").write_bytes(png(2, 2, 2))

    def tile(tile_id: str, x: int, y: int, source: int) -> dict:
        return {"id": tile_id, "x": x, "y": y, "width": 2, "height": 2, "source": source}

    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"uri": "mesh.bin", "byteLength": len(blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 36, "byteLength": 6},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "images": [{"name": "a", "uri": "a.png"}, {"name": "b", "uri": "b.png"}],
        "tileTextures": [
            {
                "name": "atlas-color",
                "usage": "color",
                "atlas_width": 4,
                "atlas_height": 2,
                "tiles": [tile("left", 0, 0, 0), tile("right", 2, 0, 1)],
            },
            {
                "name": "atlas-normal",
                "usage": "normal",
                "atlas_width": 4,
                "atlas_height": 2,
                "tiles": [tile("left", 0, 0, 1), tile("right", 2, 0, 0)],
            },
        ],
        "materials": [{
            "name": "tiled-mat",
            "pbrMetallicRoughness": {"baseColorTileTexture": {"index": 0}},
            "normalTileTexture": {"index": 1},
        }],
        "meshes": [{
            "name": "surface",
            "primitives": [
                {"attributes": {"POSITION": 0}, "indices": 1, "material": 0}
            ],
        }],
        "nodes": [{"name": "object-a", "mesh": 0}, {"name": "object-b", "mesh": 0}],
        "scenes": [{"name": "scene", "nodes": [0, 1]}],
        "scene": 0,
    }
    manifest = directory / "model.gltf"
    manifest.write_text(json.dumps(document))
    return manifest


class TiledTextureRetargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = ROOT / ".tmp-tiled-retarget-tests"
        shutil.rmtree(self.temp, ignore_errors=True)
        self.batch = self.temp / "batch"
        self.manifest = write_batch(self.batch)
        (self.batch / "new.png").write_bytes(png(1, 1, 3))
        (self.batch / "bad.png").write_bytes(png(1, 2, 4))
        self.library = ResourceLibrary(self.temp / "library")
        self.asset_id = self.library.import_batch(self.manifest)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp, ignore_errors=True)

    def objects(self, asset_id: str) -> dict:
        path = self.temp / f"{asset_id[:10]}.json"
        self.library.export_inspection(asset_id, path)
        return json.loads(path.read_text())["objects"]

    def test_import_preserves_tile_positions_and_usage_identity(self) -> None:
        objects = self.objects(self.asset_id)
        tiled = [value for value in objects.values() if value.get("atlas_width")]
        self.assertEqual({item["usage"] for item in tiled}, {"color", "normal"})
        by_usage = {item["usage"]: item for item in tiled}
        color_left = next(tile for tile in by_usage["color"]["tiles"] if tile["tile_id"] == "left")
        normal_right = next(tile for tile in by_usage["normal"]["tiles"] if tile["tile_id"] == "right")
        self.assertEqual(color_left["image_id"], normal_right["image_id"])
        self.assertEqual(color_left["x"], 0)
        self.assertEqual(normal_right["x"], 2)
        self.assertNotEqual(by_usage["color"].pop("name"), by_usage["normal"].pop("name"))
        self.assertNotEqual(by_usage["color"], by_usage["normal"])
        self.assertEqual(len([value for value in objects.values() if value.get("mime_type")]), 2)

    def test_partial_tile_and_usage_adoption_keeps_unselected_object_and_old_tiles(self) -> None:
        plan = self.batch / "plan.json"
        plan.write_text(json.dumps({"tile_updates": [
            {
                "tiled_texture": "atlas-color",
                "usage": "color",
                "tiles": {"left": {"file": "new.png"}},
                "nodes": ["object-a"],
            },
            {
                "tiled_texture": "atlas-normal",
                "usage": "normal",
                "tiles": {"right": {"file": "new.png"}},
                "nodes": ["object-a"],
            },
        ]}))
        preview = self.library.plan_retarget(self.asset_id, plan)
        self.assertTrue(preview["valid"], preview["errors"])
        self.assertEqual({change["node_name"] for change in preview["changes"]}, {"object-a"})
        self.assertEqual(
            {area["usage"]: area["missing_tile_ids"] for area in preview["missing_areas"]},
            {"color": ["right"], "normal": ["left"]},
        )
        self.assertNotIn(preview["new_asset_id"], json.loads(self.library.index_path.read_text())["assets"])

        new_asset, report = self.library.apply_retarget(self.asset_id, plan)
        old_objects = self.objects(self.asset_id)
        new_objects = self.objects(new_asset)
        old_nodes = {item["name"]: oid for oid, item in old_objects.items() if "children" in item}
        new_nodes = {item["name"]: (oid, item) for oid, item in new_objects.items() if "children" in item}
        self.assertEqual(old_nodes["object-b"], new_nodes["object-b"][0])
        self.assertNotEqual(old_nodes["object-a"], new_nodes["object-a"][0])

        old_material_id = next(oid for oid, item in old_objects.items() if item.get("name") == "tiled-mat")
        new_materials = [item for item in new_objects.values() if item.get("name") == "tiled-mat"]
        self.assertEqual(len(new_materials), 2)
        new_material = next(item for item in new_materials if item not in old_objects.values())
        self.assertNotEqual(old_material_id, next(oid for oid in new_objects if new_objects[oid] == new_material))
        new_color = new_objects[new_material["tiled_texture_ids"]["color"]]
        new_normal = new_objects[new_material["tiled_texture_ids"]["normal"]]
        color_tiles = {tile["tile_id"]: tile for tile in new_color["tiles"]}
        normal_tiles = {tile["tile_id"]: tile for tile in new_normal["tiles"]}
        self.assertEqual(color_tiles["left"]["source_tile_id"], "left")
        self.assertEqual(normal_tiles["right"]["source_tile_id"], "right")
        self.assertNotEqual(color_tiles["left"]["image_id"], color_tiles["right"]["image_id"])
        self.assertNotEqual(normal_tiles["left"]["image_id"], normal_tiles["right"]["image_id"])
        self.assertEqual(new_color["usage"], "color")
        self.assertEqual(new_normal["usage"], "normal")
        self.assertEqual(color_tiles["left"]["x"], 0)
        self.assertEqual(normal_tiles["right"]["x"], 2)
        self.assertEqual(len({tile["image_id"] for ttx in (new_color, new_normal) for tile in ttx["tiles"]}), 3)

        record = json.loads(self.library.index_path.read_text())["retargets"][new_asset]
        self.assertEqual(record["base_asset_id"], self.asset_id)
        self.assertEqual(len(record["tile_updates"]), 2)
        states = {
            (item["usage"], tile["tile_id"], tile["state"])
            for item in record["tile_updates"] for tile in item["tiles"]
        }
        self.assertIn(("color", "left", "updated"), states)
        self.assertIn(("color", "right", "retained"), states)
        self.assertIn(("normal", "right", "updated"), states)
        self.assertIn(("normal", "left", "retained"), states)
        self.assertEqual(report["new_asset_id"], new_asset)

    def test_invalid_tiles_fail_entire_publication(self) -> None:
        plan = self.batch / "bad-plan.json"
        plan.write_text(json.dumps({"tile_updates": [
            {
                "tiled_texture": "atlas-color",
                "tiles": {
                    "left": {"file": "missing.png"},
                    "right": {"file": "bad.png"},
                    "unknown": {"file": "new.png"},
                },
            },
            {
                "tiled_texture": "atlas-color",
                "usage": "normal",
                "tiles": {"left": {"file": "new.png"}},
            },
        ]}))
        report = self.library.plan_retarget(self.asset_id, plan)
        self.assertFalse(report["valid"])
        self.assertTrue(any("图像缺失" in error for error in report["errors"]))
        self.assertTrue(any("尺寸关系不合法" in error for error in report["errors"]))
        self.assertTrue(any("未知分片位置" in error for error in report["errors"]))
        self.assertTrue(any("用途不兼容" in error for error in report["errors"]))
        before = self.library.index_path.read_text()
        with self.assertRaises(ImportErrorBundle):
            self.library.apply_retarget(self.asset_id, plan)
        self.assertEqual(before, self.library.index_path.read_text())

    def test_withdraw_selected_object_uses_recorded_source_version(self) -> None:
        update_plan = self.batch / "update.json"
        update_plan.write_text(json.dumps({"tile_updates": [{
            "tiled_texture": "atlas-color",
            "tiles": {
                "left": {"file": "new.png"},
                "right": {"file": "new.png"},
            },
            "nodes": ["object-a"],
        }]}))
        updated_asset, _ = self.library.apply_retarget(self.asset_id, update_plan)
        updated_objects = self.objects(updated_asset)
        updated_material = next(
            item for item in updated_objects.values()
            if item.get("name") == "tiled-mat" and item.get("tiled_texture_ids", {}).get("color")
        )
        updated_texture_id = updated_material["tiled_texture_ids"]["color"]

        withdraw_plan = self.batch / "withdraw.json"
        withdraw_plan.write_text(json.dumps({"tile_updates": [{
            "withdraw": True,
            "tiled_texture": updated_texture_id,
            "nodes": ["object-a"],
        }]}))
        withdrawn_asset, report = self.library.apply_retarget(updated_asset, withdraw_plan)
        self.assertTrue(report["valid"], report.get("errors"))
        withdrawn_objects = self.objects(withdrawn_asset)
        nodes_a = [item for item in withdrawn_objects.values() if item.get("name") == "object-a"]
        self.assertEqual(len(nodes_a), 1)
        mesh = withdrawn_objects[nodes_a[0]["mesh_id"]]
        material_id = mesh["material_bindings"][0]["material_id"]
        material = withdrawn_objects[material_id]
        old_objects = self.objects(self.asset_id)
        old_material_id = next(oid for oid, item in old_objects.items() if item.get("name") == "tiled-mat")
        self.assertEqual(material, old_objects[old_material_id])
        self.assertEqual(
            material["tiled_texture_ids"]["color"],
            old_material_id and next(item for item in old_objects.values() if item.get("name") == "atlas-color") and
            next(oid for oid, item in old_objects.items() if item.get("name") == "atlas-color"),
        )


    def write_plan(self, name: str, updates: list) -> Path:
        plan = self.batch / name
        plan.write_text(json.dumps({"tile_updates": updates}))
        return plan

    def color_texture_id(self, objects: dict, node_name: str) -> str:
        node = next(item for item in objects.values() if item.get("name") == node_name)
        mesh = objects[node["mesh_id"]]
        material = objects[mesh["material_bindings"][0]["material_id"]]
        return material["tiled_texture_ids"]["color"]

    def test_parallel_candidates_combine_per_object(self) -> None:
        (self.batch / "c1.png").write_bytes(png(1, 1, 3))
        (self.batch / "c2.png").write_bytes(png(1, 1, 5))
        plan = self.write_plan("parallel.json", [
            {"tiled_texture": "atlas-color", "tiles": {"left": {"file": "c1.png"}}, "nodes": ["object-a"]},
            {"tiled_texture": "atlas-color", "tiles": {"left": {"file": "c2.png"}}, "nodes": ["object-b"]},
        ])
        preview = self.library.plan_retarget(self.asset_id, plan)
        self.assertTrue(preview["valid"], preview["errors"])
        self.assertEqual(preview["base_asset_id"], self.asset_id)
        self.assertEqual({change["node_name"] for change in preview["changes"]}, {"object-a", "object-b"})
        candidates = [item for item in preview["candidates"] if item["action"] == "adopt"]
        self.assertEqual(len(candidates), 2)
        for candidate in candidates:
            self.assertEqual(candidate["tiles"][0]["state"], "adopted")
            self.assertEqual(len(candidate["tiles"][0]["adopted_nodes"]), 1)

        new_asset, _ = self.library.apply_retarget(self.asset_id, plan)
        new_objects = self.objects(new_asset)
        old_objects = self.objects(self.asset_id)
        old_color = next(oid for oid, item in old_objects.items() if item.get("name") == "atlas-color")
        color_a = self.color_texture_id(new_objects, "object-a")
        color_b = self.color_texture_id(new_objects, "object-b")
        self.assertNotEqual(color_a, color_b)
        self.assertNotEqual(color_a, old_color)
        self.assertNotEqual(color_b, old_color)
        # 每个对象只换自己的 left 片，right 片沿用旧图
        for texture_id in (color_a, color_b):
            tiles = {tile["tile_id"]: tile for tile in new_objects[texture_id]["tiles"]}
            old_tiles = {tile["tile_id"]: tile for tile in old_objects[old_color]["tiles"]}
            self.assertEqual(tiles["right"]["image_id"], old_tiles["right"]["image_id"])
            self.assertNotEqual(tiles["left"]["image_id"], old_tiles["left"]["image_id"])
        left_a = next(t["image_id"] for t in new_objects[color_a]["tiles"] if t["tile_id"] == "left")
        left_b = next(t["image_id"] for t in new_objects[color_b]["tiles"] if t["tile_id"] == "left")
        self.assertNotEqual(left_a, left_b)
        # 重复采用得到同一资产，不增加重复资产
        again, _ = self.library.apply_retarget(self.asset_id, plan)
        self.assertEqual(again, new_asset)
        self.assertEqual(
            len(json.loads(self.library.index_path.read_text())["assets"]),
            2,
        )

    def test_overlapping_candidates_need_explicit_resolution(self) -> None:
        (self.batch / "c1.png").write_bytes(png(1, 1, 3))
        (self.batch / "c2.png").write_bytes(png(1, 1, 5))
        conflict = self.write_plan("conflict.json", [
            {"tiled_texture": "atlas-color", "tiles": {"left": {"file": "c1.png"}}, "nodes": ["object-a"]},
            {"tiled_texture": "atlas-color", "tiles": {"left": {"file": "c2.png"}}, "nodes": ["object-a"]},
        ])
        report = self.library.plan_retarget(self.asset_id, conflict)
        self.assertFalse(report["valid"])
        self.assertTrue(any("重叠" in error for error in report["errors"]))
        with self.assertRaises(ImportErrorBundle):
            self.library.apply_retarget(self.asset_id, conflict)

        resolved = self.write_plan("resolved.json", [
            {"tiled_texture": "atlas-color", "tiles": {"left": {"file": "c1.png"}}, "nodes": ["object-a"]},
            {"tiled_texture": "atlas-color", "priority": 1, "tiles": {"left": {"file": "c2.png"}}, "nodes": ["object-a"]},
        ])
        preview = self.library.plan_retarget(self.asset_id, resolved)
        self.assertTrue(preview["valid"], preview["errors"])
        states = {item["priority"]: item["tiles"][0]["state"] for item in preview["candidates"]}
        self.assertEqual(states, {0: "superseded", 1: "adopted"})
        # 采用结果由 priority 决定，与登记顺序无关
        swapped = self.write_plan("swapped.json", [
            {"tiled_texture": "atlas-color", "priority": 1, "tiles": {"left": {"file": "c2.png"}}, "nodes": ["object-a"]},
            {"tiled_texture": "atlas-color", "tiles": {"left": {"file": "c1.png"}}, "nodes": ["object-a"]},
        ])
        first, _ = self.library.apply_retarget(self.asset_id, resolved)
        second, _ = self.library.apply_retarget(self.asset_id, swapped)
        self.assertEqual(first, second)
        # 单独采用 c2 的结果与 priority 胜出的结果一致
        single = self.write_plan("single.json", [
            {"tiled_texture": "atlas-color", "tiles": {"left": {"file": "c2.png"}}, "nodes": ["object-a"]},
        ])
        single_asset, _ = self.library.apply_retarget(self.asset_id, single)
        self.assertEqual(single_asset, first)

    def test_withdraw_one_candidate_input_and_reopen(self) -> None:
        (self.batch / "c1.png").write_bytes(png(1, 1, 3))
        (self.batch / "c2.png").write_bytes(png(1, 1, 5))
        plan = self.write_plan("combined.json", [
            {"tiled_texture": "atlas-color", "tiles": {"left": {"file": "c1.png"}}, "nodes": ["object-a"]},
            {"tiled_texture": "atlas-color", "tiles": {"left": {"file": "c2.png"}}, "nodes": ["object-b"]},
        ])
        combined_asset, _ = self.library.apply_retarget(self.asset_id, plan)
        combined_objects = self.objects(combined_asset)
        combined_color_a = self.color_texture_id(combined_objects, "object-a")
        combined_color_b = self.color_texture_id(combined_objects, "object-b")

        withdraw = self.write_plan("withdraw-one.json", [
            {"withdraw": True, "tiled_texture": combined_color_a, "nodes": ["object-a"]},
        ])
        withdrawn_asset, report = self.library.apply_retarget(combined_asset, withdraw)
        self.assertTrue(report["valid"], report.get("errors"))
        withdrawn_objects = self.objects(withdrawn_asset)
        old_objects = self.objects(self.asset_id)
        old_color = next(oid for oid, item in old_objects.items() if item.get("name") == "atlas-color")
        # 撤回后 object-a 回到原始组合，object-b 保持自己的候选版本
        self.assertEqual(self.color_texture_id(withdrawn_objects, "object-a"), old_color)
        self.assertEqual(self.color_texture_id(withdrawn_objects, "object-b"), combined_color_b)
        # 已发布资产仍可查看
        self.assertEqual(self.color_texture_id(self.objects(combined_asset), "object-b"), combined_color_b)
        # 重开同一组合计划得到同一资产；重复撤回也不增加重复资产
        reopened, _ = self.library.apply_retarget(self.asset_id, plan)
        self.assertEqual(reopened, combined_asset)
        withdrawn_again, _ = self.library.apply_retarget(combined_asset, withdraw)
        self.assertEqual(withdrawn_again, withdrawn_asset)

    def test_tile_review_preserves_pending_selected_and_deferred_ranges(self) -> None:
        (self.batch / "c1.png").write_bytes(png(1, 1, 3))
        (self.batch / "c2.png").write_bytes(png(1, 1, 5))
        plan = self.write_plan("review-plan.json", [
            {"tiled_texture": "atlas-color", "tiles": {"left": {"file": "c1.png"}}, "nodes": ["object-a"]},
            {"tiled_texture": "atlas-color", "tiles": {"left": {"file": "c2.png"}}, "nodes": ["object-a"]},
            {"tiled_texture": "atlas-color", "tiles": {"right": {"file": "c1.png"}}, "nodes": ["object-a"]},
            {"tiled_texture": "atlas-color", "tiles": {"right": {"file": "c2.png"}}, "nodes": ["object-a"]},
            {"tiled_texture": "atlas-color", "tiles": {"left": {"file": "c1.png"}}, "nodes": ["object-b"]},
        ])
        initial = self.library.plan_retarget(self.asset_id, plan)
        self.assertFalse(initial["valid"])
        self.assertEqual(len(initial["review"]["unresolved_overlaps"]), 2)
        self.assertEqual(len(json.loads(self.library.index_path.read_text())["assets"]), 1)

        review = self.batch / "review.json"
        saved = self.library.save_tile_review(self.asset_id, plan, review, [
            {"candidate_id": "candidate-1", "node_id": "object-a", "tile_ids": ["left"], "status": "selected"},
            {"candidate_id": "candidate-4", "node_id": "object-b", "tile_ids": ["left"], "status": "deferred"},
        ])
        saved_data = json.loads(review.read_text())
        self.assertEqual(saved_data["plan_path"], str(plan))
        self.assertNotEqual(review, plan)
        self.assertEqual(len(saved["review"]["unresolved_overlaps"]), 1)
        self.assertEqual(
            {(item["node_name"], item["tile_id"]) for item in saved["review"]["unresolved_overlaps"]},
            {("object-a", "right")},
        )
        selected = next(item for item in saved_data["choices"] if item["status"] == "selected")
        self.assertEqual(selected["node_name"], "object-a")
        self.assertTrue(selected["tiled_texture_id"].startswith("ttx_"))
        self.assertEqual(selected["tile_id"], "left")
        self.assertTrue(selected["file_sha256"])
        deferred = next(item for item in saved_data["choices"] if item["status"] == "deferred")
        self.assertEqual((deferred["node_name"], deferred["tile_id"]), ("object-b", "left"))
        self.assertEqual(len(json.loads(self.library.index_path.read_text())["assets"]), 1)

        reopened = self.library.plan_retarget(self.asset_id, plan, review)
        self.assertFalse(reopened["valid"])
        self.assertTrue(any("重叠" in error for error in reopened["errors"]))
        self.assertEqual(len(reopened["review"]["unresolved_overlaps"]), 1)
        self.assertEqual(
            {(item["candidate_id"], item["node_name"], item["tile_id"], item["status"])
             for item in reopened["review"]["choices"]},
            {("candidate-1", "object-a", "left", "selected"), ("candidate-4", "object-b", "left", "deferred")},
        )

        before_index = self.library.index_path.read_text()
        (self.batch / "c2.png").write_bytes(png(1, 1, 9))
        changed_report = self.library.plan_retarget(self.asset_id, plan, review)
        changed_choices = {
            (item["candidate_id"], item["node_name"], item["tile_id"], item["status"]): item["needs_reconfirmation"]
            for item in changed_report["review"]["choices"]
        }
        self.assertEqual(len(changed_report["review"]["choices"]), 2)
        self.assertEqual(len(changed_report["review"]["unresolved_overlaps"]), 1)
        selected_key = next(key for key in changed_choices if key[3] == "selected")
        deferred_key = next(key for key in changed_choices if key[3] == "deferred")
        self.assertTrue(changed_choices[selected_key])
        self.assertFalse(changed_choices[deferred_key])
        with self.assertRaises(ImportErrorBundle):
            self.library.apply_retarget(self.asset_id, plan, review)
        self.assertEqual(before_index, self.library.index_path.read_text())

        alternate = self.batch / "alternate-review.json"
        self.library.save_tile_review_as(self.asset_id, plan, review, alternate, [
            {"candidate_id": "candidate-0", "node_id": "object-a", "tile_ids": ["left"], "status": "selected"},
            {"candidate_id": "candidate-3", "node_id": "object-a", "tile_ids": ["right"], "status": "selected"},
        ])
        self.assertEqual(self.library.index_path.read_text(), before_index)
        alternate_preview = self.library.plan_retarget(self.asset_id, plan, alternate)
        self.assertTrue(alternate_preview["valid"], alternate_preview["errors"])
        alternate_asset, alternate_report = self.library.apply_retarget(self.asset_id, plan, alternate)
        alternate_objects = self.objects(alternate_asset)
        old_objects = self.objects(self.asset_id)
        old_color = next(oid for oid, item in old_objects.items() if item.get("name") == "atlas-color")
        self.assertEqual(self.color_texture_id(alternate_objects, "object-b"), old_color)
        self.assertEqual(
            {change["node_name"] for change in alternate_report["changes"]},
            {"object-a"},
        )
        alternate_tiles = {
            tile["tile_id"]: tile
            for tile in alternate_objects[self.color_texture_id(alternate_objects, "object-a")]["tiles"]
        }
        old_tiles = {tile["tile_id"]: tile for tile in old_objects[old_color]["tiles"]}
        self.assertNotEqual(alternate_tiles["left"]["image_id"], old_tiles["left"]["image_id"])
        self.assertNotEqual(alternate_tiles["right"]["image_id"], old_tiles["right"]["image_id"])
        alternate_again, _ = self.library.apply_retarget(self.asset_id, plan, alternate)
        self.assertEqual(alternate_again, alternate_asset)
        with self.assertRaises(ImportErrorBundle):
            self.library.apply_retarget(self.asset_id, plan, review)
        self.assertEqual(len(json.loads(self.library.index_path.read_text())["assets"]), 2)

    def test_invalid_candidate_cannot_borrow_same_name_same_size_file(self) -> None:
        (self.batch / "c1.png").write_bytes(png(1, 1, 3))
        alt = self.batch / "alt"
        alt.mkdir()
        (alt / "c1.png").write_bytes(png(1, 1, 7))
        # 同名同尺寸但内容不同的文件不能互相借用身份
        plan_a = self.write_plan("only-a.json", [
            {"tiled_texture": "atlas-color", "tiles": {"left": {"file": "c1.png"}}, "nodes": ["object-a"]},
        ])
        plan_b = self.write_plan("only-b.json", [
            {"tiled_texture": "atlas-color", "tiles": {"left": {"file": "alt/c1.png"}}, "nodes": ["object-a"]},
        ])
        asset_a, _ = self.library.apply_retarget(self.asset_id, plan_a)
        asset_b, _ = self.library.apply_retarget(self.asset_id, plan_b)
        self.assertNotEqual(asset_a, asset_b)
        objects_a = self.objects(asset_a)
        objects_b = self.objects(asset_b)
        left_a = next(t["image_id"] for t in objects_a[self.color_texture_id(objects_a, "object-a")]["tiles"] if t["tile_id"] == "left")
        left_b = next(t["image_id"] for t in objects_b[self.color_texture_id(objects_b, "object-a")]["tiles"] if t["tile_id"] == "left")
        self.assertNotEqual(left_a, left_b)
        # 失效候选整批拒绝，不留下半成品，已发布资产不受影响
        before = self.library.index_path.read_text()
        bad = self.write_plan("bad-mix.json", [
            {"tiled_texture": "atlas-color", "tiles": {"left": {"file": "c1.png"}}, "nodes": ["object-a"]},
            {"tiled_texture": "atlas-color", "tiles": {"left": {"file": "bad.png"}}, "nodes": ["object-b"]},
        ])
        report = self.library.plan_retarget(self.asset_id, bad)
        self.assertFalse(report["valid"])
        self.assertTrue(any("尺寸关系不合法" in error for error in report["errors"]))
        with self.assertRaises(ImportErrorBundle):
            self.library.apply_retarget(self.asset_id, bad)
        self.assertEqual(before, self.library.index_path.read_text())
        self.assertEqual(self.color_texture_id(self.objects(asset_a), "object-a"), self.color_texture_id(objects_a, "object-a"))


if __name__ == "__main__":
    unittest.main()
