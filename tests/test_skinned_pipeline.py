from __future__ import annotations

import json
import math
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resource3d.errors import ImportErrorBundle
from resource3d.library import ResourceLibrary
from resource3d.math3d import multiply, skin_point, transform_point
from resource3d.models import Geometry, Node, Skin
from scripts.create_skinned_sample import write_skinned_sample


def load_character(temporary: Path) -> tuple[Path, ResourceLibrary, str, dict]:
    batch = temporary / "character"
    manifest = write_skinned_sample(batch)
    library = ResourceLibrary(temporary / "library")
    asset_id = library.import_batch(manifest)
    output = temporary / "inspection.json"
    library.export_inspection(asset_id, output)
    return manifest, library, asset_id, json.loads(output.read_text())["objects"]


class SkinnedImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = ROOT / ".tmp-skinned-tests"
        self.manifest, self.library, self.asset_id, self.objects = load_character(self.temporary)

    def tearDown(self) -> None:
        shutil.rmtree(self.temporary, ignore_errors=True)

    def _world_transforms(self) -> dict[str, tuple]:
        nodes = {value["name"]: value for value in self.objects.values() if len(value.get("transform", [])) == 16}
        parents = {
            "rig": None,
            "joint-hip": "rig",
            "joint-tip": "joint-hip",
            "bar-node": "rig",
        }
        worlds: dict[str, tuple] = {}

        def world(name: str) -> tuple:
            if name in worlds:
                return worlds[name]
            node = nodes[name]
            parent = parents[name]
            result = multiply(world(parent), tuple(node["transform"])) if parent else tuple(node["transform"])
            worlds[name] = result
            return result

        for name in parents:
            world(name)
        return worlds

    def test_shared_texture_identity_and_distinct_usage(self) -> None:
        # Add a second mesh/primitive that references the second material but
        # shares the same color image; geometry must remain identical too.
        document = json.loads(self.manifest.read_text())
        document["meshes"].append({
            "name": "skinned-bar-trim",
            "primitives": [{
                "attributes": {"POSITION": 0, "JOINTS_0": 2, "WEIGHTS_0": 3},
                "indices": 1,
                "material": 1,
            }],
        })
        document["nodes"].append({"name": "trim-node", "mesh": 1, "skin": 0})
        document["scenes"][0]["nodes"].append(4)
        self.manifest.write_text(json.dumps(document))
        asset_id = self.library.import_batch(self.manifest)
        output = self.temporary / "inspection2.json"
        self.library.export_inspection(asset_id, output)
        objects = json.loads(output.read_text())["objects"]

        textures = [value for value in objects.values() if value.get("usage")]
        color = {value["name"] for value in textures if value["usage"] == "color"}
        data = {value["name"] for value in textures if value["usage"] == "data"}
        self.assertEqual(color, {"paint-srgb"})
        self.assertEqual(data, {"metal-rough-linear"})
        by_name = {value["name"]: value for value in textures}
        pass
        self.assertNotEqual(by_name["paint-srgb"]["image_id"], by_name["metal-rough-linear"]["image_id"])

        materials = [value for value in objects.values() if "base_color_factor" in value]
        armor = next(value for value in materials if value["name"] == "armor")
        trim = next(value for value in materials if value["name"] == "trim")
        self.assertEqual(armor["color_texture_id"], trim["color_texture_id"])
        self.assertNotEqual(armor["color_texture_id"], armor["data_texture_id"])

        meshes = [value for value in objects.values() if "geometry_id" in value]
        geometry_ids = {mesh["geometry_id"] for mesh in meshes}
        self.assertEqual(len(geometry_ids), 1)

    def test_skinned_vertices_evaluate_through_nested_joints(self) -> None:
        skin_data = next(value for value in self.objects.values() if "joint_node_ids" in value)
        skin = Skin(skin_data["name"], tuple(skin_data["joint_node_ids"]), tuple(
            tuple(matrix) for matrix in skin_data["inverse_bind_matrices"]
        ))
        geometry_data = next(value for value in self.objects.values() if "triangle_indices" in value)
        geometry = Geometry(
            tuple(geometry_data["positions"]),
            tuple(geometry_data["triangle_indices"]),
            skin_joints=tuple(geometry_data["skin_joints"]),
            skin_weights=tuple(geometry_data["skin_weights"]),
        )
        self.assertIsNotNone(geometry.skin_joints)
        nodes_by_id = {key: value for key, value in self.objects.items() if len(value.get("transform", [])) == 16}
        joint_world = []
        worlds = self._world_transforms()
        node_names_by_id = {key: value["name"] for key, value in nodes_by_id.items()}
        for joint_id in skin.joint_node_ids:
            joint_world.append(worlds[node_names_by_id[joint_id]])

        def skinned(vertex: int) -> tuple:
            joints = tuple(geometry.skin_joints[vertex * 4:vertex * 4 + 4])
            weights = tuple(geometry.skin_weights[vertex * 4:vertex * 4 + 4])
            return skin_point(
                tuple(geometry.positions[vertex * 3:vertex * 3 + 3]),
                joints,
                weights,
                joint_world,
                list(skin.inverse_bind_matrices),
            )

        left = skinned(0)
        middle = skinned(1)
        right = skinned(2)
        self.assertTrue(all(abs(value) < 1e-9 for value in left))
        self.assertTrue(all(abs(a - b) < 1e-9 for a, b in zip(right, (0.0, 2.0, 0.0))))
        self.assertTrue(all(abs(a - b) < 1e-9 for a, b in zip(middle, (0.0, 1.0, 0.0))))

    def test_missing_joint_is_bundled_and_not_published(self) -> None:
        shutil.rmtree(self.library.root)
        document = json.loads(self.manifest.read_text())
        document["skins"][0]["joints"] = [1, 9]
        self.manifest.write_text(json.dumps(document))
        with self.assertRaises(ImportErrorBundle) as captured:
            self.library.import_batch(self.manifest)
        self.assertTrue(any("missing joint node" in error for error in captured.exception.errors))
        self.assertFalse(self.library.index_path.exists())

    def test_out_of_range_joint_index_is_rejected(self) -> None:
        shutil.rmtree(self.library.root)
        document = json.loads(self.manifest.read_text())
        # JOINTS_0 accessor 2 uses bytes in character.bin; rewrite the first
        # vertex joints to index 5.
        blob = (self.manifest.parent / "character.bin").read_bytes()
        view = document["bufferViews"][2]
        start = view["byteOffset"]
        blob = blob[:start] + bytes([5, 0, 0, 0]) + blob[start + 4:]
        (self.manifest.parent / "character.bin").write_bytes(blob)
        with self.assertRaises(ImportErrorBundle) as captured:
            self.library.import_batch(self.manifest)
        self.assertTrue(any("exceeds skin joints count" in error for error in captured.exception.errors))
        self.assertFalse(self.library.index_path.exists())

    def test_out_of_range_weight_is_rejected(self) -> None:
        shutil.rmtree(self.library.root)
        document = json.loads(self.manifest.read_text())
        blob = (self.manifest.parent / "character.bin").read_bytes()
        view = document["bufferViews"][3]
        start = view["byteOffset"]
        import struct
        patched = struct.pack("<f", 1.5)
        blob = blob[:start] + patched + blob[start + 4:]
        (self.manifest.parent / "character.bin").write_bytes(blob)
        with self.assertRaises(ImportErrorBundle) as captured:
            self.library.import_batch(self.manifest)
        self.assertTrue(any("out-of-range skin weights" in error for error in captured.exception.errors))
        self.assertFalse(self.library.index_path.exists())

    def test_missing_image_dependency_fails_batch(self) -> None:
        shutil.rmtree(self.library.root)
        (self.manifest.parent / "paint.png").unlink()
        with self.assertRaises(ImportErrorBundle) as captured:
            self.library.import_batch(self.manifest)
        self.assertTrue(any("paint.png" in error for error in captured.exception.errors))
        self.assertFalse(self.library.index_path.exists())

    def test_reimport_keeps_old_version_unchanged(self) -> None:
        before = sorted(path.relative_to(self.library.root) for path in self.library.objects_dir.rglob("*.json"))
        reimported = self.library.import_batch(self.manifest)
        after = sorted(path.relative_to(self.library.root) for path in self.library.objects_dir.rglob("*.json"))
        self.assertEqual(reimported, self.asset_id)
        self.assertEqual(before, after)

        old_output = self.temporary / "old.json"
        self.library.export_inspection(self.asset_id, old_output)
        old_objects = json.loads(old_output.read_text())["objects"]

        document = json.loads(self.manifest.read_text())
        document["materials"][0]["name"] = "armor-v2"
        self.manifest.write_text(json.dumps(document))
        new_asset_id = self.library.import_batch(self.manifest)
        self.assertNotEqual(new_asset_id, self.asset_id)

        self.library.export_inspection(self.asset_id, old_output)
        self.assertEqual(json.loads(old_output.read_text())["objects"], old_objects)


if __name__ == "__main__":
    unittest.main()
