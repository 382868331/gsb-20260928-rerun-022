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
from resource3d.models import Image


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
        (self.batch / "alt.png").write_bytes(png(1, 1, 9))
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

    def test_overlapping_candidates_require_explicit_resolution(self) -> None:
        plan = self.batch / "conflict.json"
        plan.write_text(json.dumps({"tile_updates": [
            {"tiled_texture": "atlas-color", "tiles": {"left": {"file": "new.png"}}, "nodes": ["object-a"]},
            {"tiled_texture": "atlas-color", "tiles": {"left": {"file": "alt.png"}}, "nodes": ["object-a"]},
        ]}))
        report = self.library.plan_retarget(self.asset_id, plan)
        self.assertFalse(report["valid"])
        self.assertTrue(any("重叠" in error and "priority" in error for error in report["errors"]))
        before = self.library.index_path.read_text()
        with self.assertRaises(ImportErrorBundle):
            self.library.apply_retarget(self.asset_id, plan)
        self.assertEqual(before, self.library.index_path.read_text())

    def test_parallel_candidates_combine_per_object_with_priority(self) -> None:
        plan = self.batch / "parallel.json"
        plan.write_text(json.dumps({"tile_updates": [
            {
                "tiled_texture": "atlas-color",
                "tiles": {"left": {"file": "new.png"}},
                "nodes": ["object-a"],
                "priority": 1,
            },
            {
                "tiled_texture": "atlas-color",
                "tiles": {"left": {"file": "alt.png"}, "right": {"file": "alt.png"}},
                "nodes": ["object-a", "object-b"],
            },
            {
                "tiled_texture": "atlas-color",
                "tiles": {"right": {"file": "new.png"}},
                "nodes": ["object-b"],
                "priority": 2,
            },
        ]}))
        preview = self.library.plan_retarget(self.asset_id, plan)
        self.assertTrue(preview["valid"], preview["errors"])
        self.assertEqual(preview["base_asset_id"], self.asset_id)
        self.assertEqual(preview["asset_id"], self.asset_id)
        summary = next(item for item in preview["tile_updates"] if item["usage"] == "color")
        self.assertEqual(len(summary["candidates"]), 3)
        self.assertEqual(len(summary["combinations"]), 2)
        old_objects = self.objects(self.asset_id)
        old_nodes = {item["name"]: oid for oid, item in old_objects.items() if "children" in item}
        self.assertEqual(
            set(summary["affected_nodes"]),
            {old_nodes["object-a"], old_nodes["object-b"]},
        )
        self.assertEqual(
            {change["node_name"] for change in preview["changes"]},
            {"object-a", "object-b"},
        )
        self.assertNotIn(preview["new_asset_id"], json.loads(self.library.index_path.read_text())["assets"])

        new_asset, _ = self.library.apply_retarget(self.asset_id, plan)
        new_objects = self.objects(new_asset)
        new_nodes = {item["name"]: item for item in new_objects.values() if "children" in item}

        def color_texture(node_name: str) -> dict:
            mesh = new_objects[new_nodes[node_name]["mesh_id"]]
            material = new_objects[mesh["material_bindings"][0]["material_id"]]
            return new_objects[material["tiled_texture_ids"]["color"]]

        def tile_images(texture: dict) -> dict:
            return {tile["tile_id"]: tile["image_id"] for tile in texture["tiles"]}

        left_new = Image("a#left@1x1", "image/png", png(1, 1, 3)).id()
        left_alt = Image("a#left@1x1", "image/png", png(1, 1, 9)).id()
        right_new = Image("b#right@1x1", "image/png", png(1, 1, 3)).id()
        right_alt = Image("b#right@1x1", "image/png", png(1, 1, 9)).id()
        object_a_tiles = tile_images(color_texture("object-a"))
        object_b_tiles = tile_images(color_texture("object-b"))
        # object-a: left 由高优先级候选提供 new.png, right 组合采用另一候选的 alt.png。
        self.assertEqual(object_a_tiles["left"], left_new)
        self.assertEqual(object_a_tiles["right"], right_alt)
        # object-b: left 沿用低优先级候选的 alt.png, right 由自己的高优先级候选提供。
        self.assertEqual(object_b_tiles["left"], left_alt)
        self.assertEqual(object_b_tiles["right"], right_new)
        # 两个对象得到各自的新组合, 旧资产闭包保持不变。
        self.assertNotEqual(new_nodes["object-a"], new_nodes["object-b"])
        self.assertEqual(len([item for item in old_objects.values() if item.get("atlas_width")]), 2)
        # 重复采用得到同一资产, 不增加重复资源。
        again_asset, _ = self.library.apply_retarget(self.asset_id, plan)
        self.assertEqual(again_asset, new_asset)
        index = json.loads(self.library.index_path.read_text())
        self.assertEqual(list(index["retargets"]).count(new_asset), 1)

    def test_withdraw_then_reopen_keeps_published_assets_viewable(self) -> None:
        update_plan = self.batch / "update.json"
        update_plan.write_text(json.dumps({"tile_updates": [{
            "tiled_texture": "atlas-color",
            "tiles": {"left": {"file": "new.png"}},
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
        # 重复撤回得到同一资产, 不产生重复记录。
        withdrawn_again, _ = self.library.apply_retarget(updated_asset, withdraw_plan)
        self.assertEqual(withdrawn_again, withdrawn_asset)
        # 撤回后重开同一更新, 得到与首次完全相同的资产 ID。
        reopened_asset, _ = self.library.apply_retarget(withdrawn_asset, update_plan)
        self.assertEqual(reopened_asset, updated_asset)
        # 已发布的各版本资产仍可完整查看。
        for asset in (self.asset_id, updated_asset, withdrawn_asset, reopened_asset):
            objects = self.objects(asset)
            self.assertTrue(any(item.get("atlas_width") for item in objects.values()))
        index = json.loads(self.library.index_path.read_text())
        self.assertEqual(list(index["retargets"]).count(updated_asset), 1)
        self.assertEqual(list(index["retargets"]).count(withdrawn_asset), 1)

    def test_invalid_candidate_cannot_borrow_same_name_same_size_file(self) -> None:
        shadow = self.batch / "shadow"
        shadow.mkdir()
        # 同名同尺寸的损坏文件: 内容不是合法 PNG, 不能借用正式文件的身份。
        (shadow / "new.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        plan = shadow / "plan.json"
        plan.write_text(json.dumps({"tile_updates": [{
            "tiled_texture": "atlas-color",
            "tiles": {"left": {"file": "new.png"}},
            "nodes": ["object-a"],
        }]}))
        report = self.library.plan_retarget(self.asset_id, plan)
        self.assertFalse(report["valid"])
        self.assertTrue(any("尺寸关系不合法" in error for error in report["errors"]))
        before = self.library.index_path.read_text()
        with self.assertRaises(ImportErrorBundle):
            self.library.apply_retarget(self.asset_id, plan)
        self.assertEqual(before, self.library.index_path.read_text())
        # 同名同尺寸但内容不同的合法文件只会得到自己的内容寻址身份。
        (shadow / "new.png").write_bytes(png(1, 1, 9))
        shadow_asset, _ = self.library.apply_retarget(self.asset_id, plan)
        official_plan = self.batch / "official.json"
        official_plan.write_text(json.dumps({"tile_updates": [{
            "tiled_texture": "atlas-color",
            "tiles": {"left": {"file": "new.png"}},
            "nodes": ["object-a"],
        }]}))
        official_asset, _ = self.library.apply_retarget(self.asset_id, official_plan)
        self.assertNotEqual(shadow_asset, official_asset)


if __name__ == "__main__":
    unittest.main()
