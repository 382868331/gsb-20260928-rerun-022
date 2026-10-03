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
from scripts.create_tiled_sample import write_batch


class TiledTextureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = ROOT / ".tmp-tiled-tests"
        shutil.rmtree(self.temp, ignore_errors=True)
        self.batch = self.temp / "batch"
        self.manifest = write_batch(self.batch)
        self.library = ResourceLibrary(self.temp / "library")
        self.asset_id = self.library.import_batch(self.manifest)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp, ignore_errors=True)

    def _objects(self, asset_id: str) -> dict:
        output = self.temp / f"inspection-{asset_id[4:12]}.json"
        self.library.export_inspection(asset_id, output)
        return json.loads(output.read_text())["objects"]

    def _plan(self, payload: dict) -> Path:
        path = self.batch / "plan.json"
        path.write_text(json.dumps(payload))
        return path

    def test_import_keeps_tile_identities_and_layout(self) -> None:
        objects = self._objects(self.asset_id)
        materials = [o for o in objects.values() if o.get("name") == "tiled-mat"]
        self.assertEqual(len(materials), 1)
        material = materials[0]
        slots = {entry["usage"]: entry["texture_ids"] for entry in material["tile_textures"]}
        self.assertEqual(set(slots), {"color", "normal"})
        for usage, texture_ids in slots.items():
            self.assertEqual(len(texture_ids), 2)
            positions = []
            for texture_id in texture_ids:
                texture = objects[texture_id]
                self.assertEqual(texture["usage"], usage)
                self.assertEqual(texture["tile"]["layout"], [2, 1])
                positions.append(tuple(texture["tile"]["position"]))
            self.assertEqual(sorted(positions), [(0, 0), (1, 0)])
        color_0 = objects[slots["color"][0]]
        normal_0 = objects[slots["normal"][0]]
        # Same backing bytes share one image identity, but usage+position give
        # each tile an independent texture identity.
        self.assertEqual(color_0["image_id"], normal_0["image_id"])
        self.assertNotEqual(slots["color"][0], slots["normal"][0])

    def test_preview_lists_objects_regions_and_kept_tiles(self) -> None:
        report = self.library.plan_retarget(self.asset_id, self._plan(json.loads((self.batch / "tile-plan.json").read_text())))
        self.assertTrue(report["valid"], report["errors"])
        nodes = {region["node_name"] for region in report["regions"]}
        self.assertEqual(nodes, {"node-a", "node-b"})
        positions = {(region["usage"], tuple(region["position"])) for region in report["regions"]}
        self.assertEqual(positions, {("color", (0, 0)), ("normal", (1, 0))})
        kept = {(entry["usage"], tuple(entry["position"])) for entry in report["kept_tiles"]}
        self.assertIn(("color", (1, 0)), kept)
        self.assertIn(("normal", (0, 0)), kept)
        updates = {(u["usage"], tuple(u["position"])): u for u in report["updates"]}
        self.assertEqual(updates[("color", (0, 0))]["old_size"], [4, 4])
        self.assertEqual(updates[("color", (0, 0))]["new_size"], [2, 2])
        self.assertTrue(updates[("color", (0, 0))]["edge_tags"])
        index = json.loads(self.library.index_path.read_text())
        self.assertNotIn(report["new_asset_id"], index["assets"])

    def test_selective_adoption_leaves_other_objects_on_old_tiles(self) -> None:
        plan = self._plan({**json.loads((self.batch / "tile-plan.json").read_text()), "nodes": ["node-a"]})
        new_asset, report = self.library.apply_retarget(self.asset_id, plan)
        self.assertEqual({r["node_name"] for r in report["regions"]}, {"node-a"})
        old_objects = self._objects(self.asset_id)
        new_objects = self._objects(new_asset)

        def node(objects: dict, name: str) -> tuple[str, dict]:
            matches = {oid: o for oid, o in objects.items() if o.get("name") == name and "transform" in o}
            return next(iter(matches.items()))

        old_b_id, old_b = node(old_objects, "node-b")
        new_b_id, new_b = node(new_objects, "node-b")
        self.assertEqual(old_b_id, new_b_id)
        old_a_id, _ = node(old_objects, "node-a")
        new_a_id, new_a = node(new_objects, "node-a")
        self.assertNotEqual(old_a_id, new_a_id)
        self.assertEqual(old_b["mesh_id"], new_b["mesh_id"])

        new_materials = [o for o in new_objects.values() if o.get("name") == "tiled-mat"]
        self.assertEqual(len(new_materials), 2)
        by_node = {}
        for node_id, node_obj in (("a", new_a), ("b", new_b)):
            mesh = new_objects[node_obj["mesh_id"]]
            by_node[node_id] = new_objects[mesh["material_bindings"][0]["material_id"]]
        old_material = old_objects[next(oid for oid, o in old_objects.items() if o.get("name") == "tiled-mat")]
        new_material = by_node["a"]
        old_slots = {e["usage"]: e["texture_ids"] for e in old_material["tile_textures"]}
        new_slots = {e["usage"]: e["texture_ids"] for e in new_material["tile_textures"]}
        # Updated tiles moved; untouched tiles keep the exact texture ids.
        self.assertNotEqual(old_slots["color"][0], new_slots["color"][0])
        self.assertEqual(old_slots["color"][1], new_slots["color"][1])
        self.assertEqual(old_slots["normal"][0], new_slots["normal"][0])
        self.assertNotEqual(old_slots["normal"][1], new_slots["normal"][1])
        # node-b stayed entirely on the old material.
        self.assertEqual(by_node["b"]["tile_textures"], old_material["tile_textures"])
        index = json.loads(self.library.index_path.read_text())
        record = index["tile_versions"][new_asset]
        self.assertEqual(record["root_asset_id"], self.asset_id)
        self.assertEqual(len(record["tiles"]), 2)
        self.assertIn("kept_tiles", record)

    def test_validation_failure_publishes_nothing(self) -> None:
        bad = self._plan({"tiles": [
            {"material": "tiled-mat", "usage": "color", "position": [7, 0], "file": "color-0-low.png"},
            {"material": "ghost", "usage": "color", "position": [0, 0], "file": "color-0-low.png"},
            {"material": "tiled-mat", "usage": "data", "position": [0, 0], "file": "color-0-low.png"},
            {"material": "tiled-mat", "usage": "color", "position": [0, 0], "file": "missing.png"},
        ]})
        report = self.library.plan_retarget(self.asset_id, bad)
        self.assertFalse(report["valid"])
        self.assertEqual(len(report["errors"]), 4)
        before = self.library.index_path.read_text()
        with self.assertRaises(ImportErrorBundle):
            self.library.apply_retarget(self.asset_id, bad)
        self.assertEqual(self.library.index_path.read_text(), before)

    def test_repeat_and_revert_are_reproducible(self) -> None:
        plan_path = self.batch / "tile-plan.json"
        first, _ = self.library.apply_retarget(self.asset_id, plan_path)
        second, _ = self.library.apply_retarget(self.asset_id, plan_path)
        self.assertEqual(first, second)
        self.assertEqual(self.library.retarget_source(first), self.asset_id)
        # Revert node-a only; node-b (never adopted) stays as-is.
        partial = self._plan({**json.loads(plan_path.read_text()), "nodes": ["node-a", "node-b"]})
        adopted, _ = self.library.apply_retarget(self.asset_id, partial)
        revert = self._plan({"revert": ["node-a"]})
        reverted, report = self.library.apply_retarget(adopted, revert)
        self.assertIn("node-a", report["reverted_nodes"])
        new_objects = self._objects(reverted)
        node_a = next(o for o in new_objects.values() if o.get("name") == "node-a" and "transform" in o)
        mesh = new_objects[node_a["mesh_id"]]
        material = new_objects[mesh["material_bindings"][0]["material_id"]]
        old_objects = self._objects(self.asset_id)
        old_material = next(o for o in old_objects.values() if o.get("name") == "tiled-mat")
        self.assertEqual(material["tile_textures"], old_material["tile_textures"])
        record = json.loads(self.library.index_path.read_text())["tile_versions"][reverted]
        self.assertEqual(record["root_asset_id"], self.asset_id)

    def test_edge_tag_mismatch_rejects_update(self) -> None:
        plan = self._plan({"tiles": [
            {
                "material": "tiled-mat", "usage": "color", "position": [0, 0],
                "file": "color-0-low.png",
                "edges": [["wrong"], [], [], []],
            }
        ]})
        report = self.library.plan_retarget(self.asset_id, plan)
        self.assertFalse(report["valid"])
        self.assertTrue(any("边缘对应不合法" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
