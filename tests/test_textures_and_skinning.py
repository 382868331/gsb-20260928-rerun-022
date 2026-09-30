from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resource3d.errors import ImportErrorBundle
from resource3d.library import ResourceLibrary
from resource3d.math3d import multiply, skinned_point, transform_point
from resource3d.models import Material
from scripts.create_skinned_sample import write_sample


class TextureAndSkinningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = ROOT / ".tmp-tests-rig"
        self.temp.mkdir(exist_ok=True)
        self.batch = self.temp / "rig"
        self.manifest = write_sample(self.batch)
        self.library = ResourceLibrary(self.temp / "library")

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.temp, ignore_errors=True)

    def _document(self) -> dict:
        return json.loads(self.manifest.read_text())

    def _write(self, document: dict) -> None:
        self.manifest.write_text(json.dumps(document))

    def _inspect(self, asset_id: str) -> dict:
        output = self.temp / "inspection.json"
        self.library.export_inspection(asset_id, output)
        return json.loads(output.read_text())["objects"]

    def test_duplicate_texture_shares_identity_with_distinct_usages(self) -> None:
        asset_id = self.library.import_batch(self.manifest)
        objects = self._inspect(asset_id)
        textures = [value for value in objects.values() if "media_type" in value]
        self.assertEqual(len(textures), 2)

        self.library.import_batch(self.manifest)
        index = self.library._load_index()["objects"]
        all_materials: list[Material] = [
            self.library.load_object(object_id)
            for object_id, record in index.items()
            if record["kind"] == "mat"
        ]
        material_by_name = {material.name: material for material in all_materials}
        red = material_by_name["red-cloth"]
        blue = material_by_name["blue-cloth"]
        red_usage = {usage: texture_id for texture_id, usage, _ in red.texture_bindings}
        blue_usage = {usage: texture_id for texture_id, usage, _ in blue.texture_bindings}
        self.assertEqual(red_usage["base_color"], blue_usage["base_color"])
        self.assertNotIn("normal", red_usage)
        self.assertNotIn("metallic_roughness", blue_usage)
        self.assertEqual(red_usage["metallic_roughness"], blue_usage["normal"])
        red_usages = [usage for _id, usage, _ in red.texture_bindings]
        blue_usages = [usage for _id, usage, _ in blue.texture_bindings]
        self.assertEqual(set(red_usages), {"base_color", "metallic_roughness"})
        self.assertEqual(set(blue_usages), {"base_color", "normal"})

    def test_missing_image_is_collected_and_publishes_nothing(self) -> None:
        (self.batch / "shared.png").unlink()
        with self.assertRaises(ImportErrorBundle) as captured:
            self.library.import_batch(self.manifest)
        self.assertTrue(any("shared.png" in error for error in captured.exception.errors))
        self.assertFalse(self.library.index_path.exists())

    def test_missing_skin_joint_node_fails_batch(self) -> None:
        document = self._document()
        document["skins"][0]["joints"] = [2, 99]
        self._write(document)
        with self.assertRaises(ImportErrorBundle) as captured:
            self.library.import_batch(self.manifest)
        self.assertTrue(any("joints[1]" in error for error in captured.exception.errors))
        self.assertFalse(self.library.index_path.exists())

    def test_joint_index_out_of_bounds_fails_batch(self) -> None:
        document = self._document()
        document["skins"][0]["joints"] = [2]
        document["accessors"][4]["count"] = 1
        self._write(document)
        with self.assertRaises(ImportErrorBundle) as captured:
            self.library.import_batch(self.manifest)
        messages = captured.exception.errors
        self.assertTrue(any("exceeds joints" in message for message in messages), messages)
        self.assertFalse(self.library.index_path.exists())

    def test_weight_out_of_bounds_fails_batch(self) -> None:
        import struct

        blob = (self.batch / "rig.bin").read_bytes()
        bad = struct.pack("<ffff", 1.5, 0.0, 0.0, 0.0)
        start = 48 + 12 + 16
        (self.batch / "rig.bin").write_bytes(blob[:start] + bad + blob[start + 16:])
        with self.assertRaises(ImportErrorBundle) as captured:
            self.library.import_batch(self.manifest)
        self.assertTrue(any("outside [0, 1]" in error for error in captured.exception.errors))
        self.assertFalse(self.library.index_path.exists())

    def test_skinned_nested_transforms_are_evaluable(self) -> None:
        asset_id = self.library.import_batch(self.manifest)
        objects = self._inspect(asset_id)
        nodes = {value["name"]: value for value in objects.values() if len(value.get("transform", [])) == 16}
        skins = [value for value in objects.values() if "joint_node_ids" in value]
        geometries = [value for value in objects.values() if "triangle_indices" in value]

        root = tuple(nodes["rig-root"]["transform"])
        node_by_id = {key: value for key, value in objects.items() if len(value.get("transform", [])) == 16}

        def world_for(node_id: str) -> tuple[float, ...]:
            node = node_by_id[node_id]
            if node["name"] == "rig-root":
                return tuple(node["transform"])
            return multiply(root, tuple(node["transform"]))

        skin = skins[0]
        joint_world = [world_for(node_id) for node_id in skin["joint_node_ids"]]
        inverse_bind = [tuple(matrix) for matrix in skin["inverse_bind_matrices"]]
        geometry = geometries[0]
        positions = [tuple(geometry["positions"][i:i + 3]) for i in range(0, len(geometry["positions"]), 3)]
        skinned = [
            skinned_point(positions[i], tuple(geometry["joint_indices"][i]), tuple(geometry["weights"][i]), joint_world, inverse_bind)
            for i in range(len(positions))
        ]
        self.assertTrue(all(math.isclose(value, 0.0, abs_tol=1e-9) for point in skinned[:2] for value in (point[0], point[1])))
        self.assertTrue(math.isclose(skinned[0][2], 0.5, abs_tol=1e-9))
        self.assertTrue(math.isclose(skinned[1][2], -0.5, abs_tol=1e-9))
        self.assertTrue(math.isclose(skinned[2][2], 0.5, abs_tol=1e-9))
        self.assertTrue(math.isclose(skinned[3][2], -0.5, abs_tol=1e-9))
        plain = transform_point(root, (0.5, 1.0, 0.0))
        self.assertTrue(math.isclose(plain[2], -0.5, abs_tol=1e-9))

    def test_reimport_keeps_old_version_pristine(self) -> None:
        first = self.library.import_batch(self.manifest)
        before = sorted(path.read_bytes() for path in self.library.objects_dir.rglob("*.json"))
        second = self.library.import_batch(self.manifest)
        after = sorted(path.read_bytes() for path in self.library.objects_dir.rglob("*.json"))
        self.assertEqual(first, second)
        self.assertEqual(before, after)

        document = self._document()
        document["materials"][0]["name"] = "red-cloth-v2"
        self._write(document)
        updated = self.library.import_batch(self.manifest)
        self.assertNotEqual(first, updated)
        first_objects = self._inspect(first)
        self.assertTrue(any(value.get("name") == "red-cloth" for value in first_objects.values()))
        updated_objects = self._inspect(updated)
        self.assertTrue(any(value.get("name") == "red-cloth-v2" for value in updated_objects.values()))


if __name__ == "__main__":
    unittest.main()
