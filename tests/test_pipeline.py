from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.create_sample import write_sample
from resource3d.errors import ImportErrorBundle
from resource3d.library import ResourceLibrary
from resource3d.math3d import multiply, transform_point
from resource3d.models import Mesh, Node


def matmul(matrix_a: list[float], matrix_b: list[float]) -> list[float]:
    return list(multiply(tuple(matrix_a), tuple(matrix_b)))


class Resource3dPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = ROOT / ".tmp-tests"
        self.temp.mkdir(exist_ok=True)
        self.batch = self.temp / "robot"
        self.manifest = write_sample(self.batch)
        self.library = ResourceLibrary(self.temp / "library")

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.temp, ignore_errors=True)

    def test_import_dedup_export_transforms_faces_and_bindings(self) -> None:
        first_asset_id = self.library.import_batch(self.manifest)
        before = sorted((self.library.objects_dir).rglob("*.json"))
        second_asset_id = self.library.import_batch(self.manifest)
        after = sorted((self.library.objects_dir).rglob("*.json"))

        self.assertEqual(first_asset_id, second_asset_id)
        self.assertEqual(before, after)

        output = self.temp / "inspection.json"
        self.library.export_inspection(first_asset_id, output)
        data = json.loads(output.read_text())["objects"]
        nodes = [value for value in data.values() if len(value.get("transform", [])) == 16]
        meshes = [value for value in data.values() if "geometry_id" in value]
        geometries = [value for value in data.values() if "triangle_indices" in value]

        self.assertEqual(len(meshes), 3)
        self.assertEqual(len(geometries), 2)
        body_mesh = next(mesh for mesh in meshes if mesh["name"] == "body")
        head_mesh = next(mesh for mesh in meshes if mesh["name"] == "sensor-head")
        self.assertEqual(body_mesh["geometry_id"], head_mesh["geometry_id"])
        self.assertEqual(len(body_mesh["material_bindings"]), 1)
        self.assertNotEqual(
            body_mesh["material_bindings"][0]["material_id"],
            head_mesh["material_bindings"][0]["material_id"],
        )

        total_faces = sum(len(geometry["triangle_indices"]) // 3 for geometry in geometries)
        self.assertEqual(total_faces, 18)

        by_name = {node["name"]: node for node in nodes}
        world_transforms = {}

        def world(node_name: str) -> list[float]:
            if node_name in world_transforms:
                return world_transforms[node_name]
            node = by_name[node_name]
            parent_names = {"body-node": "world", "head-node": "body-node", "antenna-node": "world"}
            parent = parent_names.get(node_name)
            result = matmul(world(parent), node["transform"]) if parent else node["transform"]
            world_transforms[node_name] = result
            return result

        head_origin = transform_point(tuple(world("head-node")), (0.0, 0.0, 0.0))
        angle = math.pi / 4
        expected_x = 1.2 * math.cos(angle)
        expected_z = -1.2 * math.sin(angle)
        self.assertTrue(math.isclose(head_origin[0], expected_x, abs_tol=1e-9))
        self.assertTrue(math.isclose(head_origin[1], 0.8, abs_tol=1e-9))
        self.assertTrue(math.isclose(head_origin[2], expected_z, abs_tol=1e-9))

    def test_missing_dependency_reports_batch_and_publishes_nothing(self) -> None:
        (self.batch / "robot.bin").unlink()
        with self.assertRaises(ImportErrorBundle) as captured:
            self.library.import_batch(self.manifest)
        self.assertTrue(any("robot.bin" in error for error in captured.exception.errors))
        self.assertFalse(self.library.index_path.exists())

    def test_external_reference_outside_batch_is_rejected(self) -> None:
        outside = self.temp / "evil.bin"
        outside.write_bytes((self.batch / "robot.bin").read_bytes())
        document = json.loads(self.manifest.read_text())
        document["buffers"][0]["uri"] = "../evil.bin"
        self.manifest.write_text(json.dumps(document))
        with self.assertRaises(ImportErrorBundle) as captured:
            self.library.import_batch(self.manifest)
        self.assertTrue(any("batch root" in error for error in captured.exception.errors))
        self.assertFalse(self.library.index_path.exists())

    def test_unsupported_scope_is_rejected(self) -> None:
        document = json.loads(self.manifest.read_text())
        document["animations"] = []
        self.manifest.write_text(json.dumps(document))
        with self.assertRaises(ImportErrorBundle):
            self.library.import_batch(self.manifest)

    def test_corrupt_json_is_bundle_error_without_publication(self) -> None:
        self.manifest.write_text("{not valid json", encoding="utf-8")
        with self.assertRaises(ImportErrorBundle):
            self.library.import_batch(self.manifest)
        self.assertFalse(self.library.index_path.exists())


if __name__ == "__main__":
    unittest.main()
