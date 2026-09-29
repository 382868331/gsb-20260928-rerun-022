import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.create_sample import create_broken_batch, create_valid_batch
from r3d.inspect import export_scene, verify_closure
from r3d.store import Library


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.library = Library(self.root / "library")
        self.batch = self.root / "sample_batch"
        create_valid_batch(self.batch)

    def tearDown(self):
        self.temp.cleanup()

    def test_import_export_preserves_transforms_faces_and_references(self):
        report = self.library.import_batch(self.batch)
        self.assertTrue(report.committed, report.issues)
        self.assertEqual(len(report.scene_ids), 1)
        scene_id = report.scene_ids[0]

        closure = verify_closure(self.library, scene_id)
        self.assertTrue(closure.ok, closure.issues)
        self.assertEqual(closure.resources["geometry"], 6)
        self.assertEqual(closure.resources["material"], 6)
        self.assertEqual(closure.resources["image"], 3)
        self.assertEqual(closure.resources["node"], 4)

        output = self.root / "export.json"
        bundle = export_scene(self.library, scene_id, output)
        geometry_stat = bundle["geometryStats"][0]
        self.assertEqual(geometry_stat["vertexCount"], 24)
        self.assertEqual(geometry_stat["triangleCount"], 12)
        self.assertEqual(geometry_stat["instances"], 6)
        self.assertEqual(len(geometry_stat["materials"]), 2)

        # Node name -> id, then world matrix checks for nested/shared nodes.
        nodes_by_name = {
            payload["name"]: rid
            for rid, payload in bundle["resources"].items()
            if payload.get("type") == "node"
        }
        matrices = bundle["worldMatrices"]
        left_cap = nodes_by_name["left-cap"]
        right_box = nodes_by_name["right-box"]
        # rig-root translates y+1; left-box translates x-1.5; cap scales 0.5
        # and translates y+2 -> local unit-x becomes world (-1.25, 1, 0).
        cap_x = matrices[left_cap][0]
        cap_origin_x = matrices[left_cap][12]
        cap_origin_y = matrices[left_cap][13]
        self.assertAlmostEqual(cap_x, 0.5, places=6)
        self.assertAlmostEqual(cap_origin_x, -1.5, places=6)
        self.assertAlmostEqual(cap_origin_y, 3.0, places=6)
        self.assertAlmostEqual(matrices[right_box][5], 0.70710678, places=6)
        self.assertAlmostEqual(matrices[right_box][4], -0.70710678, places=6)
        self.assertAlmostEqual(matrices[right_box][0], 0.70710678, places=6)

        on_disk = json.loads(output.read_text("utf-8"))
        self.assertEqual(on_disk["sceneId"], scene_id)
        self.assertEqual(len(on_disk["worldPointSamples"]), 3)

    def test_repeated_import_reuses_every_resource(self):
        first = self.library.import_batch(self.batch)
        self.assertTrue(first.committed)
        resource_files_before = list((self.library.root / "resources").rglob("*.json"))
        blob_files_before = list((self.library.root / "blobs").rglob("*.bin"))
        second = self.library.import_batch(self.batch)
        self.assertTrue(second.committed)
        self.assertEqual(second.new_resources, {})
        self.assertEqual(second.new_blobs, 0)
        self.assertGreater(sum(second.reused_resources.values()), 0)
        resource_files_after = list((self.library.root / "resources").rglob("*.json"))
        blob_files_after = list((self.library.root / "blobs").rglob("*.bin"))
        self.assertEqual(resource_files_before, resource_files_after)
        self.assertEqual(blob_files_before, blob_files_after)
        self.assertEqual(first.scene_ids, second.scene_ids)

    def test_broken_batch_is_reported_and_publishes_nothing(self):
        broken = self.root / "broken_batch"
        create_broken_batch(broken)
        report = self.library.import_batch(broken)
        self.assertFalse(report.committed)
        self.assertEqual(report.scene_ids, [])
        self.assertFalse((self.library.root / "resources").exists())
        self.assertFalse((self.library.root / "blobs").exists())
        self.assertTrue(any("missing dependency" in issue for issue in report.issues))
        self.assertTrue(any("escapes the batch" in issue for issue in report.issues))

    def test_one_bad_file_rejects_entire_multi_file_batch(self):
        good_report = self.library.import_batch(self.batch)
        self.assertTrue(good_report.committed)
        mixed = self.root / "mixed"
        create_valid_batch(mixed)
        create_broken_batch(mixed)
        before = {path.name for path in (self.library.root / "resources").rglob("*.json")}
        report = self.library.import_batch(mixed)
        self.assertFalse(report.committed)
        after = {path.name for path in (self.library.root / "resources").rglob("*.json")}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
