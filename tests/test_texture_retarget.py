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

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def make_png(width: int, height: int, seed: int = 0) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = bytes([seed % 256]) * (width * 3)
    raw = b"".join(b"\x00" + row for _ in range(height))
    return (
        PNG_SIGNATURE
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def write_shared_batch(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    blob = struct.pack("<9f3H", *(value for vector in positions for value in vector), 0, 1, 2)
    (destination / "mesh.bin").write_bytes(blob)
    (destination / "shared.png").write_bytes(make_png(4, 4, seed=1))

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
        "images": [{"name": "shared", "uri": "shared.png"}],
        "textures": [{"name": "shared-tex", "source": 0}],
        "materials": [
            {
                "name": "shared-mat",
                "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                "normalTexture": {"index": 0},
                "occlusionTexture": {"index": 0},
            }
        ],
        "meshes": [
            {
                "name": "tri",
                "primitives": [
                    {"attributes": {"POSITION": 0}, "indices": 1, "material": 0}
                ],
            }
        ],
        "nodes": [
            {"name": "node-a", "mesh": 0},
            {"name": "node-b", "mesh": 0},
        ],
        "scenes": [{"name": "scene", "nodes": [0, 1]}],
        "scene": 0,
    }
    manifest = destination / "model.gltf"
    manifest.write_text(json.dumps(document) + "\n")
    return manifest


class TextureRetargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = ROOT / ".tmp-retarget-tests"
        shutil.rmtree(self.temp, ignore_errors=True)
        self.batch = self.temp / "batch"
        self.manifest = write_shared_batch(self.batch)
        (self.batch / "shared-low.png").write_bytes(make_png(2, 2, seed=2))
        self.library = ResourceLibrary(self.temp / "library")
        self.asset_id = self.library.import_batch(self.manifest)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp, ignore_errors=True)

    def _objects(self, asset_id: str) -> dict:
        output = self.temp / f"inspection-{asset_id[4:12]}.json"
        self.library.export_inspection(asset_id, output)
        return json.loads(output.read_text())["objects"]

    def _write_plan(self, replacements: list[dict]) -> Path:
        plan = self.batch / "plan.json"
        plan.write_text(json.dumps({"replacements": replacements}))
        return plan

    def _color_plan(self, nodes: list[str] | None = None) -> Path:
        entry = {"image": "shared", "usage": "color", "file": "shared-low.png"}
        if nodes is not None:
            entry["nodes"] = nodes
        return self._write_plan([entry])

    def test_preview_lists_switching_materials_and_objects(self) -> None:
        report = self.library.plan_retarget(self.asset_id, self._color_plan())
        self.assertTrue(report["valid"])
        self.assertEqual(report["errors"], [])
        changed_nodes = {change["node_name"] for change in report["changes"]}
        self.assertEqual(changed_nodes, {"node-a", "node-b"})
        for change in report["changes"]:
            self.assertEqual(change["usage"], "color")
            self.assertEqual(change["material_name"], "shared-mat")
            self.assertNotEqual(change["old_image_id"], change["new_image_id"])
            self.assertNotEqual(change["old_texture_id"], change["new_texture_id"])
        # Preview must not publish anything.
        index = json.loads(self.library.index_path.read_text())
        self.assertNotIn(report["new_asset_id"], index["assets"])

    def test_partial_selection_keeps_unselected_objects_shared(self) -> None:
        plan = self._color_plan(nodes=["node-a"])
        new_asset, report = self.library.apply_retarget(self.asset_id, plan)
        self.assertEqual({c["node_name"] for c in report["changes"]}, {"node-a"})
        old_objects = self._objects(self.asset_id)
        new_objects = self._objects(new_asset)

        def node_by_name(objects: dict, name: str) -> tuple[str, dict]:
            matches = {oid: o for oid, o in objects.items() if o.get("name") == name and "transform" in o}
            self.assertEqual(len(matches), 1)
            return next(iter(matches.items()))

        old_node_a_id, _ = node_by_name(old_objects, "node-a")
        old_node_b_id, old_node_b = node_by_name(old_objects, "node-b")
        new_node_a_id, new_node_a = node_by_name(new_objects, "node-a")
        new_node_b_id, new_node_b = node_by_name(new_objects, "node-b")
        # Unselected node keeps its exact object identity; only node-a moved.
        self.assertEqual(old_node_b_id, new_node_b_id)
        self.assertNotEqual(old_node_a_id, new_node_a_id)
        self.assertEqual(old_node_b["mesh_id"], new_node_b["mesh_id"])
        self.assertNotEqual(new_node_a["mesh_id"], new_node_b["mesh_id"])
        old_mesh = old_objects[old_node_b["mesh_id"]]
        new_mesh_b = new_objects[new_node_b["mesh_id"]]
        self.assertEqual(old_mesh["material_bindings"], new_mesh_b["material_bindings"])
        # Both resolutions remain reachable in the new asset closure.
        images = [o for o in new_objects.values() if o.get("mime_type")]
        self.assertEqual(len(images), 2)

    def test_shared_image_usages_retarget_independently(self) -> None:
        new_asset, _ = self.library.apply_retarget(self.asset_id, self._color_plan())
        old_objects = self._objects(self.asset_id)
        new_objects = self._objects(new_asset)
        old_material = next(o for o in old_objects.values() if o.get("name") == "shared-mat")
        new_material = next(o for o in new_objects.values() if o.get("name") == "shared-mat")
        # Color switched; normal and occlusion records stay on the old image.
        self.assertNotEqual(old_material["color_texture_id"], new_material["color_texture_id"])
        self.assertEqual(old_material["normal_texture_id"], new_material["normal_texture_id"])
        self.assertEqual(old_material["occlusion_texture_id"], new_material["occlusion_texture_id"])
        old_color = old_objects[old_material["color_texture_id"]]
        new_color = new_objects[new_material["color_texture_id"]]
        self.assertEqual(new_color["usage"], "color")
        self.assertNotEqual(old_color["image_id"], new_color["image_id"])
        self.assertEqual(
            old_objects[old_material["normal_texture_id"]]["image_id"],
            old_color["image_id"],
        )

    def test_invalid_plans_report_reasons_and_publish_nothing(self) -> None:
        (self.batch / "odd.png").write_bytes(make_png(3, 3, seed=3))
        (self.batch / "big.png").write_bytes(make_png(8, 8, seed=4))
        plan = self._write_plan([
            {"image": "shared", "usage": "color", "file": "missing.png"},
            {"image": "shared", "usage": "color", "file": "odd.png"},
            {"image": "shared", "usage": "color", "file": "big.png"},
            {"image": "shared", "usage": "data", "file": "shared-low.png"},
            {"image": "ghost", "usage": "color", "file": "shared-low.png"},
        ])
        report = self.library.plan_retarget(self.asset_id, plan)
        self.assertFalse(report["valid"])
        self.assertEqual(len(report["errors"]), 5)
        self.assertTrue(any("图像缺失" in e and "missing.png" in e for e in report["errors"]))
        self.assertTrue(any("尺寸关系不合法" in e and "3x3" in e for e in report["errors"]))
        self.assertTrue(any("尺寸关系不合法" in e and "8x8" in e for e in report["errors"]))
        self.assertTrue(any("用途不兼容" in e and "data" in e for e in report["errors"]))
        self.assertTrue(any("图像缺失" in e and "ghost" in e for e in report["errors"]))

        index_before = self.library.index_path.read_text()
        with self.assertRaises(ImportErrorBundle):
            self.library.apply_retarget(self.asset_id, plan)
        # Failed update publishes no half-written resources.
        self.assertEqual(self.library.index_path.read_text(), index_before)

    def test_repeat_retarget_reuses_objects_and_records_provenance(self) -> None:
        plan = self._color_plan()
        first_asset, _ = self.library.apply_retarget(self.asset_id, plan)
        index_after_first = json.loads(self.library.index_path.read_text())
        second_asset, _ = self.library.apply_retarget(self.asset_id, plan)
        index_after_second = json.loads(self.library.index_path.read_text())
        self.assertEqual(first_asset, second_asset)
        self.assertEqual(index_after_first, index_after_second)
        record = index_after_second["retargets"][first_asset]
        self.assertEqual(record["base_asset_id"], self.asset_id)
        self.assertEqual(len(record["replacements"]), 1)
        self.assertEqual(record["replacements"][0]["usage"], "color")
        self.assertEqual(record["replacements"][0]["new_size"], [2, 2])

    def test_round_trip_export_and_rollback(self) -> None:
        old_export = self.temp / "old.json"
        self.library.export_inspection(self.asset_id, old_export)
        old_bytes = old_export.read_bytes()

        new_asset, _ = self.library.apply_retarget(self.asset_id, self._color_plan(nodes=["node-a"]))
        # Rollback resolves to the original version; its export is byte-stable.
        self.assertEqual(self.library.retarget_source(new_asset), self.asset_id)
        self.library.export_inspection(self.asset_id, old_export)
        self.assertEqual(old_export.read_bytes(), old_bytes)
        with self.assertRaises(KeyError):
            self.library.retarget_source(self.asset_id)

        # Re-exporting the retargeted asset reproduces each node's choice:
        # two material versions coexist, one per resolution.
        new_objects = self._objects(new_asset)
        old_objects = self._objects(self.asset_id)
        old_material_id = next(
            oid for oid, o in old_objects.items() if o.get("name") == "shared-mat"
        )
        new_materials = [o for o in new_objects.values() if o.get("name") == "shared-mat"]
        self.assertEqual(len(new_materials), 2)
        color_textures = {m["color_texture_id"] for m in new_materials}
        self.assertIn(old_objects[old_material_id]["color_texture_id"], color_textures)
        images = {o["image_id"] for o in new_objects.values() if o.get("usage") == "color"}
        self.assertEqual(len(images), 2)


if __name__ == "__main__":
    unittest.main()
