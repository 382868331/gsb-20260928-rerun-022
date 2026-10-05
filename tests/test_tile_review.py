from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_tiled_texture_retarget import png, write_batch

from resource3d.library import ResourceLibrary


class TileReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = ROOT / ".tmp-tile-review-tests"
        shutil.rmtree(self.temp, ignore_errors=True)
        self.batch = self.temp / "batch"
        self.manifest = write_batch(self.batch)
        (self.batch / "c1.png").write_bytes(png(1, 1, 3))
        (self.batch / "c2.png").write_bytes(png(1, 1, 5))
        self.library = ResourceLibrary(self.temp / "library")
        self.asset_id = self.library.import_batch(self.manifest)
        self.plan = self.batch / "plan.json"
        self.plan.write_text(json.dumps({"tile_updates": [
            {"tiled_texture": "atlas-color", "tiles": {"left": {"file": "c1.png"}}, "nodes": ["object-a"]},
            {"tiled_texture": "atlas-color", "tiles": {"left": {"file": "c2.png"}}, "nodes": ["object-a"]},
            {"tiled_texture": "atlas-color", "tiles": {"left": {"file": "c1.png"}}, "nodes": ["object-b"]},
        ]}))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp, ignore_errors=True)

    def write_review_decisions(self, review_path: Path, choices: dict[tuple[str, str], str]) -> None:
        report = self.library.review_tile_plan(self.asset_id, self.plan, review_path)[0]
        by_scope = {
            (candidate["node_name"], tile["file"]): candidate["candidate_id"]
            for candidate in report["candidates"]
            for tile in candidate["tiles"]
        }
        self.library.save_tile_review(
            self.asset_id, self.plan, review_path,
            {"decisions": [
                {"candidate_id": by_scope[scope], "state": state}
                for scope, state in choices.items()
            ]},
        )

    def objects(self, asset_id: str) -> dict:
        path = self.temp / f"{asset_id[:10]}.json"
        self.library.export_inspection(asset_id, path)
        return json.loads(path.read_text())["objects"]

    def color_texture_id(self, objects: dict, node_name: str) -> str:
        node = next(item for item in objects.values() if item.get("name") == node_name)
        mesh = objects[node["mesh_id"]]
        material = objects[mesh["material_bindings"][0]["material_id"]]
        return material["tiled_texture_ids"]["color"]

    def test_unresolved_overlaps_and_undecided_scopes_are_saved_and_resumed(self) -> None:
        review = self.batch / "review.json"
        report = self.library.save_tile_review(self.asset_id, self.plan, review)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(len(report["overlaps"]), 1)
        self.assertEqual(len(report["unresolved_overlaps"]), 1)
        node_names = {candidate["node_name"] for candidate in report["candidates"]}
        self.assertEqual(node_names, {"object-a", "object-b"})
        # 重开后未决项与未解决重叠原样保留
        reopened = self.library.review_tile_plan(self.asset_id, self.plan, review)[0]
        self.assertEqual(len(reopened["unresolved_overlaps"]), 1)
        self.assertTrue(all(item["state"] == "undecided" for item in reopened["decisions"]))
        # 未解决重叠时不能发布
        blocked, published = self.library.apply_reviewed_tile_plan(
            self.asset_id, self.plan, review, commit=True,
        )
        self.assertFalse(blocked["valid"])
        self.assertIsNone(published)
        self.assertTrue(any("重叠" in error for error in blocked["errors"]))
        # 审阅记录与计划是两份独立文件
        record = json.loads(review.read_text())
        self.assertEqual(record["kind"], "tile-update-review")
        self.assertNotEqual(review.resolve(), self.plan.resolve())
        self.assertEqual(record["plan_file"], self.plan.name)

    def test_decision_records_object_tile_and_candidate_file_identity(self) -> None:
        review = self.batch / "review.json"
        self.write_review_decisions(review, {
            ("object-a", "c1.png"): "selected",
            ("object-a", "c2.png"): "deferred",
        })
        record = json.loads(review.read_text())
        selected = next(item for item in record["decisions"] if item["state"] == "selected")
        scope = selected["scope"]
        self.assertEqual(scope["node_name"], "object-a")
        tile = scope["tiles"][0]
        self.assertEqual(tile["tile_id"], "left")
        self.assertEqual(tile["file"], "c1.png")
        self.assertEqual(len(tile["file_sha256"]), 64)
        self.assertTrue(tile["image_id"].startswith("img_"))
        self.assertTrue(scope["tiled_texture_id"].startswith("ttx_"))

    def test_selected_candidates_publish_and_other_objects_stay_unchanged(self) -> None:
        review = self.batch / "review.json"
        self.write_review_decisions(review, {
            ("object-a", "c1.png"): "selected",
            ("object-a", "c2.png"): "deferred",
        })
        preview, _ = self.library.apply_reviewed_tile_plan(
            self.asset_id, self.plan, review, commit=False,
        )
        self.assertTrue(preview["valid"], preview["errors"])
        self.assertEqual({c["node_name"] for c in preview["changes"]}, {"object-a"})
        _, (new_asset, _) = self.library.apply_reviewed_tile_plan(
            self.asset_id, self.plan, review, commit=True,
        )
        new_objects = self.objects(new_asset)
        old_objects = self.objects(self.asset_id)
        old_color = next(oid for oid, item in old_objects.items() if item.get("name") == "atlas-color")
        self.assertNotEqual(self.color_texture_id(new_objects, "object-a"), old_color)
        # object-b 没有任何已选候选，分片身份完全不变
        self.assertEqual(self.color_texture_id(new_objects, "object-b"), old_color)
        # 重复操作不增加资产
        before = len(json.loads(self.library.index_path.read_text())["assets"])
        _, (again, _) = self.library.apply_reviewed_tile_plan(
            self.asset_id, self.plan, review, commit=True,
        )
        after = len(json.loads(self.library.index_path.read_text())["assets"])
        self.assertEqual(again, new_asset)
        self.assertEqual(before, after)

    def test_changed_candidate_file_flags_only_dependent_choices(self) -> None:
        review = self.batch / "review.json"
        self.write_review_decisions(review, {
            ("object-a", "c1.png"): "selected",
            ("object-a", "c2.png"): "deferred",
        })
        # 另有一个未决候选（object-b），验证文件变化后未决项也保留
        report = self.library.review_tile_plan(self.asset_id, self.plan, review)[0]
        undecided = [item for item in report["decisions"] if item["state"] == "undecided"]
        self.assertEqual(len(undecided), 1)

        (self.batch / "c1.png").write_bytes(png(1, 1, 9))
        reopened = self.library.review_tile_plan(self.asset_id, self.plan, review)[0]
        object_b_c1 = next(
            candidate["candidate_id"] for candidate in reopened["candidates"]
            if candidate["node_name"] == "object-b" and candidate["tiles"][0]["file"] == "c1.png"
        )
        flagged = {item["candidate_id"] for item in reopened["decisions"] if item["needs_recheck"]}
        self.assertEqual(len(flagged), 1)
        states = {item["candidate_id"]: item["state"] for item in reopened["decisions"]}
        # 其他选择与未决项都保留，整份记录没有被清空
        self.assertEqual(len(states), 3)
        self.assertIn("deferred", states.values())
        self.assertIn("undecided", states.values())
        blocked, _ = self.library.apply_reviewed_tile_plan(
            self.asset_id, self.plan, review, commit=True,
        )
        self.assertFalse(blocked["valid"])
        self.assertTrue(any("重新确认" in error for error in blocked["errors"]))
        # 重新确认后恢复可发布
        flagged_id = next(iter(flagged))
        self.library.save_tile_review(
            self.asset_id, self.plan, review,
            {"decisions": [
                {"candidate_id": flagged_id, "state": "selected"},
                {"candidate_id": object_b_c1, "state": "deferred"},
            ]},
        )
        confirmed = self.library.review_tile_plan(self.asset_id, self.plan, review)[0]
        self.assertEqual(confirmed["needs_recheck"], [])
        _, published = self.library.apply_reviewed_tile_plan(
            self.asset_id, self.plan, review, commit=True,
        )
        self.assertIsNotNone(published)

    def test_save_as_creates_independent_review_without_publishing(self) -> None:
        review = self.batch / "review.json"
        first = self.library.review_tile_plan(self.asset_id, self.plan, review)[0]
        ids = {
            (candidate["node_name"], tile["file"]): candidate["candidate_id"]
            for candidate in first["candidates"]
            for tile in candidate["tiles"]
            if candidate["node_name"] == "object-a" and tile["file"] in {"c1.png", "c2.png"}
        }
        # 原记录先有自己的人工决定
        self.library.save_tile_review(
            self.asset_id, self.plan, review,
            {"decisions": [
                {"candidate_id": ids[("object-a", "c1.png")], "state": "selected"},
                {"candidate_id": ids[("object-a", "c2.png")], "state": "deferred"},
            ]},
        )
        before = len(json.loads(self.library.index_path.read_text())["assets"])
        alternate = self.batch / "review-alt.json"
        self.library.save_tile_review(
            self.asset_id, self.plan, review,
            {"decisions": [
                # 副本最初从原记录分支：这里再改选不会写回原记录
                {"candidate_id": ids[("object-a", "c2.png")], "state": "selected"},
                {"candidate_id": ids[("object-a", "c1.png")], "state": "deferred"},
            ]},
            save_as=alternate,
        )
        # 保存（含另存）本身不发布资产
        self.assertEqual(
            len(json.loads(self.library.index_path.read_text())["assets"]), before,
        )
        selected_of = lambda path: [
            item["scope"]["tiles"][0]["file"]
            for item in json.loads(path.read_text())["decisions"]
            if item["state"] == "selected"
        ]
        self.assertEqual(selected_of(review), ["c1.png"])
        self.assertEqual(selected_of(alternate), ["c2.png"])
        _, (asset_a, _) = self.library.apply_reviewed_tile_plan(
            self.asset_id, self.plan, review, commit=True,
        )
        _, (asset_b, _) = self.library.apply_reviewed_tile_plan(
            self.asset_id, self.plan, alternate, commit=True,
        )
        self.assertNotEqual(asset_a, asset_b)

    def test_changed_file_is_rechecked_per_record_without_cross_copy_edits(self) -> None:
        review = self.batch / "review.json"
        alternate = self.batch / "review-alt.json"
        scan = self.library.review_tile_plan(self.asset_id, self.plan, review)[0]
        ids = {
            (candidate["node_name"], tile["file"]): candidate["candidate_id"]
            for candidate in scan["candidates"]
            for tile in candidate["tiles"]
            if candidate["node_name"] == "object-a"
        }
        # 原记录选择 c1；另存出副本后在副本上改选 c2、暂缓 c1
        self.library.save_tile_review(
            self.asset_id, self.plan, review,
            {"decisions": [
                {"candidate_id": ids[("object-a", "c1.png")], "state": "selected"},
                {"candidate_id": ids[("object-a", "c2.png")], "state": "deferred"},
            ]},
        )
        self.library.save_tile_review(self.asset_id, self.plan, review, None, save_as=alternate)
        self.library.save_tile_review(
            self.asset_id, self.plan, alternate,
            {"decisions": [
                {"candidate_id": ids[("object-a", "c2.png")], "state": "selected"},
                {"candidate_id": ids[("object-a", "c1.png")], "state": "deferred"},
            ]},
        )
        # 修改 c1.png：原记录的 selected(c1) 与副本的 deferred(c1) 各自被标记；
        # 副本的 selected(c2) 与原记录的 deferred(c2) 不受影响
        (self.batch / "c1.png").write_bytes(png(1, 1, 9))
        reopened_original = self.library.review_tile_plan(self.asset_id, self.plan, review)[0]
        reopened_copy = self.library.review_tile_plan(self.asset_id, self.plan, alternate)[0]
        flagged_original = {
            (item["scope"]["tiles"][0]["file"], item["state"])
            for item in reopened_original["decisions"] if item["needs_recheck"]
        }
        flagged_copy = {
            (item["scope"]["tiles"][0]["file"], item["state"])
            for item in reopened_copy["decisions"] if item["needs_recheck"]
        }
        self.assertEqual(flagged_original, {("c1.png", "selected")})
        self.assertEqual(flagged_copy, {("c1.png", "deferred")})
        selected_of = lambda report: [
            item["scope"]["tiles"][0]["file"]
            for item in report["decisions"] if item["state"] == "selected"
        ]
        self.assertEqual(selected_of(reopened_original), ["c1.png"])
        self.assertEqual(selected_of(reopened_copy), ["c2.png"])
        # 只在副本上重新确认暂缓 c1（现文件身份），原记录仍需复核且发布被阻止
        self.library.save_tile_review(
            self.asset_id, self.plan, alternate,
            {"decisions": [
                {"candidate_id": ids[("object-a", "c1.png")], "state": "deferred"},
            ]},
        )
        blocked, _ = self.library.apply_reviewed_tile_plan(
            self.asset_id, self.plan, review, commit=True,
        )
        self.assertFalse(blocked["valid"])
        self.assertTrue(any("重新确认" in error for error in blocked["errors"]))
        # 副本可独立复核后发布；原记录的 c1 选择没有被副本动作改写
        _, (asset_b, first_report) = self.library.apply_reviewed_tile_plan(
            self.asset_id, self.plan, alternate, commit=True,
        )
        self.assertTrue(first_report["valid"], first_report.get("errors"))
        still_original = self.library.review_tile_plan(self.asset_id, self.plan, review)[0]
        self.assertEqual(selected_of(still_original), ["c1.png"])
        self.assertEqual(still_original["needs_recheck"], [ids[("object-a", "c1.png")]])
        # 复核原记录（当前文件内容）后发布，得到另一份资产；重复发布不新增资产
        self.library.save_tile_review(
            self.asset_id, self.plan, review,
            {"decisions": [
                {"candidate_id": ids[("object-a", "c1.png")], "state": "selected"},
            ]},
        )
        _, (asset_a, _) = self.library.apply_reviewed_tile_plan(
            self.asset_id, self.plan, review, commit=True,
        )
        self.assertNotEqual(asset_a, asset_b)
        before = len(json.loads(self.library.index_path.read_text())["assets"])
        _, (asset_a_again, _) = self.library.apply_reviewed_tile_plan(
            self.asset_id, self.plan, review, commit=True,
        )
        _, (asset_b_again, _) = self.library.apply_reviewed_tile_plan(
            self.asset_id, self.plan, alternate, commit=True,
        )
        after = len(json.loads(self.library.index_path.read_text())["assets"])
        self.assertEqual((asset_a_again, asset_b_again), (asset_a, asset_b))
        self.assertEqual(before, after)

    def test_withdraw_choice_remains_valid_through_review(self) -> None:
        adopt = self.batch / "adopt.json"
        adopt.write_text(json.dumps({"tile_updates": [
            {"tiled_texture": "atlas-color",
             "tiles": {"left": {"file": "c1.png"}, "right": {"file": "c1.png"}},
             "nodes": ["object-a"]},
        ]}))
        updated, _ = self.library.apply_retarget(self.asset_id, adopt)
        updated_objects = self.objects(updated)
        updated_ttx = self.color_texture_id(updated_objects, "object-a")
        withdraw_plan = self.batch / "withdraw.json"
        withdraw_plan.write_text(json.dumps({"tile_updates": [
            {"withdraw": True, "tiled_texture": updated_ttx, "nodes": ["object-a"]},
        ]}))
        review = self.batch / "withdraw-review.json"
        scan = self.library.save_tile_review(updated, withdraw_plan, review)
        self.assertEqual(scan["candidates"][0]["action"], "withdraw")
        candidate_id = scan["candidates"][0]["candidate_id"]
        self.library.save_tile_review(
            updated, withdraw_plan, review,
            {"decisions": [{"candidate_id": candidate_id, "state": "selected"}]},
        )
        _, (withdrawn, report) = self.library.apply_reviewed_tile_plan(
            updated, withdraw_plan, review, commit=True,
        )
        self.assertTrue(report["valid"], report.get("errors"))
        self.assertEqual(withdrawn, self.asset_id)
        _, (again, _) = self.library.apply_reviewed_tile_plan(
            updated, withdraw_plan, review, commit=True,
        )
        self.assertEqual(again, withdrawn)

    def test_invalid_candidate_keeps_review_record_but_blocks_publication(self) -> None:
        plan = self.batch / "bad.json"
        plan.write_text(json.dumps({"tile_updates": [
            {"tiled_texture": "atlas-color", "tiles": {"left": {"file": "missing.png"}},
             "nodes": ["object-a"]},
        ]}))
        review = self.batch / "bad-review.json"
        report = self.library.save_tile_review(self.asset_id, plan, review)
        self.assertFalse(report["valid"])
        self.assertTrue(any("图像缺失" in error for error in report["errors"]))
        # 记录仍然保存，重开后还能看到该候选
        reopened = self.library.review_tile_plan(self.asset_id, plan, review)[0]
        self.assertEqual(len(reopened["candidates"]), 1)
        blocked, _ = self.library.apply_reviewed_tile_plan(
            self.asset_id, plan, review, commit=True,
        )
        self.assertFalse(blocked["valid"])


if __name__ == "__main__":
    unittest.main()
