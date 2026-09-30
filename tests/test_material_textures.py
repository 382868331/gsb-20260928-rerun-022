from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resource3d.errors import ImportErrorBundle
from resource3d.library import ResourceLibrary

PNG_HEADER = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])


def write_material_batch(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    positions = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    import struct

    blob = struct.pack("<9f3H", *(value for vector in positions for value in vector), 0, 1, 2)
    (destination / "mesh.bin").write_bytes(blob)
    (destination / "paint.png").write_bytes(PNG_HEADER + b"PAINT")
    (destination / "data.png").write_bytes(PNG_HEADER + b"DATA")

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
            {"name": "paint", "uri": "paint.png"},
            {"name": "data", "uri": "data.png"},
        ],
        "textures": [
            {"name": "paint-srgb", "source": 0},
            {"name": "data-linear", "source": 1},
            {"name": "paint-normal", "source": 0},
        ],
        "materials": [
            {
                "name": "full-fat",
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicRoughnessTexture": {"index": 1},
                },
                "normalTexture": {"index": 2, "texCoord": 0, "scale": 1.5},
                "occlusionTexture": {"index": 1, "strength": 0.8},
                "emissiveTexture": {"index": 0},
                "emissiveFactor": [0.2, 0.1, 0.0],
            }
        ],
        "meshes": [
            {
                "name": "tri",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "indices": 1,
                        "material": 0,
                    }
                ],
            }
        ],
        "nodes": [{"name": "tri-node", "mesh": 0}],
        "scenes": [{"name": "scene", "nodes": [0]}],
        "scene": 0,
    }
    manifest = destination / "model.gltf"
    manifest.write_text(json.dumps(document) + "\n")
    return manifest


class MaterialTextureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = ROOT / ".tmp-material-tests"
        self.manifest = write_material_batch(self.temp / "batch")
        self.library = ResourceLibrary(self.temp / "library")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp, ignore_errors=True)

    def _objects(self, asset_id: str) -> dict:
        output = self.temp / "inspection.json"
        self.library.export_inspection(asset_id, output)
        return json.loads(output.read_text())["objects"]

    def test_all_texture_usages_import_and_export(self) -> None:
        asset_id = self.library.import_batch(self.manifest)
        objects = self._objects(asset_id)
        material = next(value for value in objects.values() if value.get("name") == "full-fat")
        for slot in (
            "color_texture_id",
            "data_texture_id",
            "normal_texture_id",
            "occlusion_texture_id",
            "emissive_texture_id",
        ):
            self.assertTrue(material[slot], slot)
        self.assertAlmostEqual(material["normal_scale"], 1.5)
        self.assertAlmostEqual(material["occlusion_strength"], 0.8)
        self.assertEqual(material["emissive_factor"], [0.2, 0.1, 0.0])

        by_id = {key: value for key, value in objects.items()}
        normal = by_id[material["normal_texture_id"]]
        color = by_id[material["color_texture_id"]]
        occlusion = by_id[material["occlusion_texture_id"]]
        data = by_id[material["data_texture_id"]]
        emissive = by_id[material["emissive_texture_id"]]
        self.assertEqual(normal["usage"], "normal")
        self.assertEqual(occlusion["usage"], "occlusion")
        self.assertEqual(emissive["usage"], "emissive")

        # The same backing image keeps a single img_ id but distinct texture
        # interpretations; a data map never becomes a color map.
        self.assertEqual(normal["image_id"], color["image_id"])
        self.assertEqual(normal["image_id"], emissive["image_id"])
        self.assertNotEqual(material["normal_texture_id"], material["color_texture_id"])
        self.assertNotEqual(material["normal_texture_id"], material["emissive_texture_id"])
        self.assertEqual(occlusion["image_id"], data["image_id"])
        self.assertNotEqual(material["occlusion_texture_id"], material["data_texture_id"])
        images = [value for value in objects.values() if value.get("mime_type")]
        self.assertEqual(len(images), 2)

    def test_broken_normal_texture_is_bundled_with_missing_image(self) -> None:
        document = json.loads(self.manifest.read_text())
        (self.manifest.parent / "data.png").unlink()
        document["materials"][0]["normalTexture"] = {"index": 9}
        self.manifest.write_text(json.dumps(document))
        with self.assertRaises(ImportErrorBundle) as captured:
            self.library.import_batch(self.manifest)
        messages = captured.exception.errors
        self.assertTrue(any("data.png" in message for message in messages), messages)
        self.assertTrue(any("normalTexture" in message for message in messages), messages)
        self.assertFalse(self.library.index_path.exists())

    def test_material_update_diff_reports_usages_images_and_users(self) -> None:
        old_asset = self.library.import_batch(self.manifest)
        document = json.loads(self.manifest.read_text())
        # Drop occlusion, switch the normal map to the data image, add a mesh
        # node using the same material so users are reported.
        del document["materials"][0]["occlusionTexture"]
        document["materials"][0]["normalTexture"] = {"index": 1}
        document["nodes"].append({"name": "tri-node-copy", "mesh": 0})
        document["scenes"][0]["nodes"].append(1)
        self.manifest.write_text(json.dumps(document))
        new_asset = self.library.import_batch(self.manifest)
        self.assertNotEqual(old_asset, new_asset)

        report = self.library.compare_assets(old_asset, new_asset)
        changed = report["materials"]["changed"]
        self.assertEqual(len(changed), 1)
        entry = changed[0]
        self.assertEqual(entry["name"], "full-fat")
        removed_usages = {item["usage"] for item in entry["usages"]["removed"]}
        changed_usages = {item["usage"] for item in entry["usages"]["changed"]}
        self.assertEqual(removed_usages, {"occlusion"})
        self.assertEqual(changed_usages, {"normal"})
        normal_change = next(item for item in entry["usages"]["changed"] if item["usage"] == "normal")
        self.assertNotEqual(normal_change["old_image_id"], normal_change["new_image_id"])
        mesh_users = entry["used_by"]
        self.assertEqual(len(mesh_users), 1)
        node_names = {node["name"] for mesh in mesh_users for node in mesh["nodes"]}
        self.assertEqual(node_names, {"tri-node", "tri-node-copy"})
        removed_textures = {item["usage"] for item in report["textures"]["removed"]}
        self.assertIn("occlusion", removed_textures)

        # Old version closure and a shared material stay byte-stable.
        output_old = self.temp / "old.json"
        output_new = self.temp / "new.json"
        self.library.export_inspection(old_asset, output_old)
        old_first = output_old.read_text()
        self.library.import_batch(self.manifest)
        self.library.export_inspection(old_asset, output_old)
        self.assertEqual(output_old.read_text(), old_first)
        self.library.export_inspection(new_asset, output_new)
        old_objects = json.loads(old_first)["objects"]
        new_objects = json.loads(output_new.read_text())["objects"]
        shared = {
            key
            for key in old_objects
            if key.startswith("img_") and key in new_objects
        }
        self.assertTrue(shared)


if __name__ == "__main__":
    unittest.main()
