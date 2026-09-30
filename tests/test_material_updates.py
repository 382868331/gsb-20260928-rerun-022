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
from scripts.create_skinned_sample import write_skinned_sample


class MaterialUsageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = ROOT / ".tmp-usage-tests"
        self.batch = self.temporary / "character"
        self.manifest = write_skinned_sample(self.batch)
        self.library = ResourceLibrary(self.temporary / "library")
        self.asset_id = self.library.import_batch(self.manifest)

    def tearDown(self) -> None:
        shutil.rmtree(self.temporary, ignore_errors=True)

    def _objects(self) -> dict:
        output = self.temporary / "inspection.json"
        self.library.export_inspection(self.asset_id, output)
        return json.loads(output.read_text())["objects"]

    def test_normal_occlusion_emissive_flow_through(self) -> None:
        objects = self._objects()
        textures = [value for value in objects.values() if value.get("usage")]
        usages = {value["name"]: value["usage"] for value in textures}
        self.assertEqual(usages["surface-normal"], "normal")
        self.assertEqual(usages["surface-occlusion"], "occlusion")
        self.assertEqual(usages["glow-emissive"], "emissive")
        self.assertEqual(usages["paint-srgb"], "color")
        self.assertEqual(usages["metal-rough-linear"], "data")

        armor = next(value for value in objects.values() if value.get("name") == "armor")
        self.assertIsNotNone(armor["normal_texture_id"])
        self.assertIsNotNone(armor["occlusion_texture_id"])
        self.assertIsNotNone(armor["emissive_texture_id"])
        self.assertEqual(armor["emissive_factor"], [0.1, 0.0, 0.0])
        by_id = {value["name"]: value for value in textures}
        self.assertEqual(
            by_id["surface-normal"]["image_id"],
            by_id["surface-occlusion"]["image_id"],
        )
        self.assertNotEqual(armor["normal_texture_id"], armor["occlusion_texture_id"])

    def test_data_usage_never_treated_as_color(self) -> None:
        objects = self._objects()
        armor = next(value for value in objects.values() if value.get("name") == "armor")
        data_texture = objects[armor["data_texture_id"]]
        color_texture = objects[armor["color_texture_id"]]
        self.assertEqual(data_texture["usage"], "data")
        self.assertEqual(color_texture["usage"], "color")
        # Shared image across usages still produces distinct tex identities.
        same_image_pairs = [
            (armor["normal_texture_id"], armor["occlusion_texture_id"]),
        ]
        for first, second in same_image_pairs:
            self.assertEqual(objects[first]["image_id"], objects[second]["image_id"])
            self.assertNotEqual(first, second)

    def test_missing_normal_image_fails_whole_batch(self) -> None:
        shutil.rmtree(self.library.root)
        document = json.loads(self.manifest.read_text())
        # Repoint the normal texture at an image whose URI cannot resolve.
        document["images"][2]["uri"] = "missing-surface.png"
        self.manifest.write_text(json.dumps(document))
        with self.assertRaises(ImportErrorBundle) as captured:
            self.library.import_batch(self.manifest)
        self.assertTrue(any("missing-surface.png" in error for error in captured.exception.errors))
        self.assertFalse(self.library.index_path.exists())

    def test_multiple_failures_are_aggregated(self) -> None:
        shutil.rmtree(self.library.root)
        document = json.loads(self.manifest.read_text())
        document["images"][2]["uri"] = "missing-surface.png"
        document["skins"][0]["joints"] = [1, 9]
        self.manifest.write_text(json.dumps(document))
        with self.assertRaises(ImportErrorBundle) as captured:
            self.library.import_batch(self.manifest)
        joined = "\n".join(captured.exception.errors)
        self.assertIn("missing-surface.png", joined)
        self.assertIn("missing joint node", joined)
        self.assertFalse(self.library.index_path.exists())


class MaterialUpdateReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = ROOT / ".tmp-diff-tests"
        self.batch = self.temporary / "character"
        self.manifest = write_skinned_sample(self.batch)
        self.library = ResourceLibrary(self.temporary / "library")
        document = json.loads(self.manifest.read_text())
        document["meshes"].append({
            "name": "trim-bar",
            "primitives": [{
                "attributes": {"POSITION": 0, "JOINTS_0": 2, "WEIGHTS_0": 3},
                "indices": 1,
                "material": 1,
            }],
        })
        document["nodes"].append({"name": "trim-node", "mesh": 1, "skin": 0})
        document["scenes"][0]["nodes"].append(4)
        self.manifest.write_text(json.dumps(document))
        self.old_asset_id = self.library.import_batch(self.manifest)

    def tearDown(self) -> None:
        shutil.rmtree(self.temporary, ignore_errors=True)

    def test_local_material_update_report(self) -> None:
        document = json.loads(self.manifest.read_text())
        armor = document["materials"][0]
        armor["normalTexture"] = {"index": 3}
        armor["emissiveFactor"] = [0.5, 0.0, 0.0]
        self.manifest.write_text(json.dumps(document))
        new_asset_id = self.library.import_batch(self.manifest)
        report = self.library.material_update_report(self.old_asset_id, new_asset_id)

        self.assertEqual(len(report["changed_materials"]), 1)
        change = report["changed_materials"][0]
        self.assertEqual(change["name"], "armor")
        usages = {entry["usage"]: entry for entry in change["usage_changes"]}
        self.assertEqual(usages["normal"]["change"], "replaced")
        self.assertNotEqual(usages["normal"]["old_texture_id"], usages["normal"]["new_texture_id"])
        self.assertEqual(usages["normal"]["new_image_id"], usages["normal"]["old_image_id"])
        self.assertEqual(change["factor_changes"]["emissive_factor"]["new"], [0.5, 0.0, 0.0])
        self.assertIn("skinned-bar", change["users_after"]["meshes"])
        self.assertIn("bar-node", change["users_after"]["nodes"])
        self.assertEqual(report["added_materials"], [])
        self.assertEqual(report["removed_materials"], [])

    def test_update_does_not_change_shared_material(self) -> None:
        # Both armor and trim share the same color image/texture.
        document = json.loads(self.manifest.read_text())
        document["materials"][0]["emissiveFactor"] = [0.2, 0.2, 0.2]
        self.manifest.write_text(json.dumps(document))
        new_asset_id = self.library.import_batch(self.manifest)
        report = self.library.material_update_report(self.old_asset_id, new_asset_id)
        changed_names = {entry["name"] for entry in report["changed_materials"]}
        self.assertEqual(changed_names, {"armor"})

        old_output = self.temporary / "old.json"
        self.library.export_inspection(self.old_asset_id, old_output)
        old_objects = json.loads(old_output.read_text())["objects"]
        new_output = self.temporary / "new.json"
        self.library.export_inspection(new_asset_id, new_output)
        new_objects = json.loads(new_output.read_text())["objects"]
        trim_before = next(value for value in old_objects.values() if value.get("name") == "trim")
        trim_after = next(value for value in new_objects.values() if value.get("name") == "trim")
        self.assertEqual(trim_before, trim_after)
        paint_before = next(value for value in old_objects.values() if value.get("name") == "paint-srgb")
        paint_after = next(value for value in new_objects.values() if value.get("name") == "paint-srgb")
        self.assertEqual(paint_before, paint_after)


if __name__ == "__main__":
    unittest.main()
