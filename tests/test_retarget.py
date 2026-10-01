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


def make_png(width: int, height: int, seed: int = 0) -> bytes:
    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = bytes([(seed + i) % 256 for i in range(3 * width)])
    raw = b"".join(b"\x00" + row for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def write_shared_batch(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    blob = struct.pack("<9f3H", 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 2)
    (destination / "mesh.bin").write_bytes(blob)
    (destination / "atlas.png").write_bytes(make_png(4, 4, seed=1))
    (destination / "mix.png").write_bytes(make_png(4, 4, seed=2))
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
        "images": [
            {"name": "atlas", "uri": "atlas.png"},
            {"name": "mix", "uri": "mix.png"},
        ],
        "textures": [
            {"name": "atlas-color", "source": 0},
            {"name": "atlas-normal", "source": 0},
            {"name": "atlas-occlusion", "source": 0},
            {"name": "mix-color", "source": 1},
            {"name": "mix-data", "source": 1},
        ],
        "materials": [
            {
                "name": "shared",
                "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                "normalTexture": {"index": 1},
                "occlusionTexture": {"index": 2},
            },
            {
                "name": "dual",
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 3},
                    "metallicRoughnessTexture": {"index": 4},
                },
            },
        ],
        "meshes": [
            {
                "name": "tri-a",
                "primitives": [
                    {"attributes": {"POSITION": 0}, "indices": 1, "material": 0},
                ],
            },
            {
                "name": "tri-b",
                "primitives": [
                    {"attributes": {"POSITION": 0}, "indices": 1, "material": 0},
                ],
            },
            {
                "name": "tri-c",
                "primitives": [
                    {"attributes": {"POSITION": 0}, "indices": 1, "material": 1},
                ],
            },
        ],
        "nodes": [
            {"name": "obj-a", "mesh": 0},
            {"name": "obj-b", "mesh": 1},
            {"name": "obj-c", "mesh": 2},
            {"name": "root", "children": [0, 1, 2]},
        ],
        "scenes": [{"name": "scene", "nodes": [3]}],
        "scene": 0,
    }
    manifest = destination / "model.gltf"
    manifest.write_text(json.dumps(document) + "\n")
    return manifest


class RetargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = ROOT / ".tmp-retarget-tests"
        shutil.rmtree(self.temp, ignore_errors=True)
        self.batch = self.temp / "batch"
        self.manifest = write_shared_batch(self.batch)
        (self.batch / "atlas_1k.png").write_bytes(make_png(2, 2, seed=3))
        (self.batch / "mix_1k.png").write_bytes(make_png(2, 2, seed=4))
        self.library = ResourceLibrary(self.temp / "library")
        self.asset_id = self.library.import_batch(self.manifest)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp, ignore_errors=True)

    def _objects(self, asset_id: str) -> dict:
        output = self.temp / f"export-{asset_id[:12]}.json"
        self.library.export_inspection(asset_id, output)
        return json.loads(output.read_text())["objects"]

    def _spec(self, replacements: list[dict]) -> dict:
        return {"replacements": replacements}

    def test_plan_previews_switches_for_selected_objects_only(self) -> None:
        spec = self._spec([{
            "image": "atlas",
            "file": "atlas_1k.png",
            "usages": ["color", "normal", "occlusion"],
            "objects": ["obj-a"],
        }])
        plan = self.library.plan_retarget(self.asset_id, spec, self.batch)
        self.assertEqual(plan["errors"], [])
        self.assertEqual(len(plan["changes"]), 1)
        change = plan["changes"][0]
        self.assertEqual(change["node"], "obj-a")
        self.assertEqual(change["material"], "shared")
        self.assertEqual(
            [switch["usage"] for switch in change["switches"]],
            ["color", "normal", "occlusion"],
        )
        for switch in change["switches"]:
            self.assertNotEqual(switch["old_image_id"], switch["new_image_id"])

    def test_apply_keeps_unselected_objects_on_original_versions(self) -> None:
        spec = self._spec([{
            "image": "atlas",
            "file": "atlas_1k.png",
            "objects": ["obj-a"],
        }])
        new_asset = self.library.apply_retarget(self.asset_id, spec, self.batch)
        self.assertNotEqual(new_asset, self.asset_id)
        old_objects = self._objects(self.asset_id)
        new_objects = self._objects(new_asset)

        def node_id(objects: dict, name: str) -> str:
            return next(key for key, value in objects.items() if value.get("name") == name and key.startswith("ndo_"))

        # Unselected objects keep their exact node ids; the selected one moved.
        self.assertIn(node_id(old_objects, "obj-b"), new_objects)
        self.assertIn(node_id(old_objects, "obj-c"), new_objects)
        self.assertNotIn(node_id(old_objects, "obj-a"), new_objects)

        def material_for(objects: dict, node_name: str) -> dict:
            node = objects[node_id(objects, node_name)]
            mesh = objects[node["mesh_id"]]
            return objects[mesh["material_bindings"][0]["material_id"]]

        old_mat_a = material_for(old_objects, "obj-a")
        new_mat_a = material_for(new_objects, "obj-a")
        new_mat_b = material_for(new_objects, "obj-b")
        # obj-b still uses the original shared material identity.
        self.assertIn(node_id(old_objects, "obj-b"), new_objects)
        old_mat_b = material_for(old_objects, "obj-b")
        self.assertEqual(
            old_mat_b["color_texture_id"],
            new_mat_b["color_texture_id"],
        )
        # obj-a switched all three usages to textures backed by the new image.
        for slot in ("color_texture_id", "normal_texture_id", "occlusion_texture_id"):
            self.assertNotEqual(old_mat_a[slot], new_mat_a[slot])
            old_tex = old_objects[old_mat_a[slot]]
            new_tex = new_objects[new_mat_a[slot]]
            self.assertEqual(new_tex["usage"], old_tex["usage"])
            self.assertNotEqual(new_tex["image_id"], old_tex["image_id"])
        # Material parameters and usage interpretation are unchanged.
        self.assertEqual(new_mat_a["name"], "shared")
        self.assertEqual(new_mat_a["normal_scale"], old_mat_a["normal_scale"])
        self.assertEqual(new_mat_a["occlusion_strength"], old_mat_a["occlusion_strength"])

    def test_color_and_data_adoptions_stay_independent(self) -> None:
        spec = self._spec([{
            "image": "mix",
            "file": "mix_1k.png",
            "usages": ["color"],
            "objects": ["obj-c"],
        }])
        new_asset = self.library.apply_retarget(self.asset_id, spec, self.batch)
        old_objects = self._objects(self.asset_id)
        new_objects = self._objects(new_asset)
        old_mat = next(v for v in old_objects.values() if v.get("name") == "dual")
        new_mat = next(v for v in new_objects.values() if v.get("name") == "dual")
        # Only the color usage switched; the data usage keeps the original texture.
        self.assertNotEqual(old_mat["color_texture_id"], new_mat["color_texture_id"])
        self.assertEqual(old_mat["data_texture_id"], new_mat["data_texture_id"])
        record = next(v for v in new_objects.values() if "base_asset_id" in v)
        self.assertEqual(record["base_asset_id"], self.asset_id)
        usages = [entry["usage"] for entry in record["entries"]]
        self.assertEqual(usages, ["color"])
        self.assertEqual(record["entries"][0]["node"], "obj-c")

    def test_repeated_apply_is_idempotent(self) -> None:
        spec = self._spec([{
            "image": "atlas",
            "file": "atlas_1k.png",
            "objects": ["obj-a"],
        }])
        first = self.library.apply_retarget(self.asset_id, spec, self.batch)
        index_after_first = json.loads(self.library.index_path.read_text())
        second = self.library.apply_retarget(self.asset_id, spec, self.batch)
        self.assertEqual(first, second)
        index_after_second = json.loads(self.library.index_path.read_text())
        self.assertEqual(index_after_first, index_after_second)

    def test_failed_update_publishes_nothing(self) -> None:
        (self.batch / "atlas_3x3.png").write_bytes(make_png(3, 3, seed=5))
        spec = self._spec([
            {"image": "atlas", "file": "missing.png", "objects": ["obj-a"]},
            {"image": "atlas", "file": "atlas_3x3.png", "objects": ["obj-a"]},
            {"image": "atlas", "file": "atlas_1k.png", "usages": ["emissive"], "objects": ["obj-a"]},
            {"image": "ghost", "file": "atlas_1k.png", "objects": ["obj-a"]},
            {"image": "atlas", "file": "atlas_1k.png", "objects": ["obj-zzz"]},
        ])
        plan = self.library.plan_retarget(self.asset_id, spec, self.batch)
        self.assertEqual(len(plan["errors"]), 5, plan["errors"])
        self.assertTrue(any("missing" in error for error in plan["errors"]))
        self.assertTrue(any("integer multiple" in error for error in plan["errors"]))
        self.assertTrue(any("emissive" in error for error in plan["errors"]))
        self.assertTrue(any("ghost" in error for error in plan["errors"]))
        self.assertTrue(any("obj-zzz" in error for error in plan["errors"]))
        index_before = self.library.index_path.read_text()
        with self.assertRaises(ImportErrorBundle):
            self.library.apply_retarget(self.asset_id, spec, self.batch)
        self.assertEqual(self.library.index_path.read_text(), index_before)

    def test_same_size_replacement_is_rejected(self) -> None:
        (self.batch / "atlas_copy.png").write_bytes(make_png(4, 4, seed=9))
        spec = self._spec([{"image": "atlas", "file": "atlas_copy.png", "objects": ["obj-a"]}])
        plan = self.library.plan_retarget(self.asset_id, spec, self.batch)
        self.assertTrue(any("not smaller" in error for error in plan["errors"]))

    def test_roundtrip_export_and_rollback(self) -> None:
        spec = self._spec([
            {"image": "atlas", "file": "atlas_1k.png", "objects": ["obj-a"]},
            {"image": "mix", "file": "mix_1k.png", "usages": ["color", "data"], "objects": ["obj-c"]},
        ])
        new_asset = self.library.apply_retarget(self.asset_id, spec, self.batch)
        # Re-export reproduces each object's choice deterministically.
        first = self.temp / "roundtrip-1.json"
        second = self.temp / "roundtrip-2.json"
        self.library.export_inspection(new_asset, first)
        self.library.export_inspection(new_asset, second)
        self.assertEqual(first.read_text(), second.read_text())
        objects = json.loads(first.read_text())["objects"]
        record = next(v for v in objects.values() if "base_asset_id" in v)
        adopted = {(entry["node"], entry["usage"]) for entry in record["entries"]}
        self.assertEqual(
            adopted,
            {
                ("obj-a", "color"), ("obj-a", "normal"), ("obj-a", "occlusion"),
                ("obj-c", "color"), ("obj-c", "data"),
            },
        )
        # Rollback resolves the original asset, whose export is untouched.
        self.assertEqual(self.library.rollback_asset(new_asset), self.asset_id)
        self.assertEqual(self.library.rollback_asset(self.asset_id), self.asset_id)
        old_export = self.temp / "rollback.json"
        self.library.export_inspection(self.asset_id, old_export)
        old_objects = json.loads(old_export.read_text())["objects"]
        self.assertFalse(any("base_asset_id" in value for value in old_objects.values()))


if __name__ == "__main__":
    unittest.main()
