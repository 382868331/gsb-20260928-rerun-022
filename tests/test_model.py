import math
import unittest

from r3d.model import make_geometry, make_material, make_node, make_scene, resource_id, GEOMETRY
from r3d.transform import IDENTITY, compose_trs, transform_point


class ModelTests(unittest.TestCase):
    def test_content_identity_is_stable_and_deduplicates(self):
        positions = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        triangles = [[0, 1, 2]]
        first_id, first = make_geometry(positions=positions, triangles=triangles)
        second_id, second = make_geometry(
            positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            triangles=[[0, 1, 2]],
        )
        self.assertTrue(first_id.startswith("geometry-"))
        self.assertEqual(first_id, second_id)
        self.assertEqual(first, second)

    def test_semantic_changes_change_identity(self):
        triangles = [[0, 1, 2]]
        red_id, _ = make_material(base_color_factor=[1, 0, 0, 1])
        blue_id, _ = make_material(base_color_factor=[0, 0, 1, 1])
        self.assertNotEqual(red_id, blue_id)
        positions = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        moved_id, _ = make_geometry(positions=[[0, 0, 1], [1, 0, 0], [0, 1, 0]], triangles=triangles)
        original_id, _ = make_geometry(positions=positions, triangles=triangles)
        self.assertNotEqual(moved_id, original_id)

    def test_references_must_use_matching_id_kinds(self):
        geometry_id, _ = make_geometry(
            positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], triangles=[[0, 1, 2]]
        )
        with self.assertRaises(ValueError):
            make_node(matrix=IDENTITY, meshes=[{"geometry": geometry_id, "material": geometry_id}])
        with self.assertRaises(ValueError):
            make_scene(roots=[geometry_id])

    def test_trs_compose_matches_expected_point_transform(self):
        matrix = compose_trs(translation=[2, 3, 4], scale=[2, 2, 2])
        self.assertEqual(transform_point(matrix, [1, 0, -1]), [4, 3, 2])
        rotated = compose_trs(rotation=[0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)])
        x, y, _ = transform_point(rotated, [1, 0, 0])
        self.assertAlmostEqual(x, 0, places=6)
        self.assertAlmostEqual(y, 1, places=6)


if __name__ == "__main__":
    unittest.main()
