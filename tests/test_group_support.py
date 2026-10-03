import copy
import unittest

from thyroid_poc.data import group_support


class GroupSupportTests(unittest.TestCase):
    def test_transitive_links_order_and_no_mutation(self):
        rows = [
            dict(group="a", sha256="x", label=0),
            dict(group="a", sha256="y", label=1),
            dict(group="b", sha256="y", label=1),
            dict(group="b", sha256="z", label=0),
            dict(group="c", sha256="z", label=0),
            dict(group="d", sha256="w", label=1),
        ]
        before = copy.deepcopy(rows)
        support = group_support(rows)
        self.assertEqual(6, support["image_count"])
        self.assertEqual(4, support["declared_group_count"])
        self.assertEqual(2, support["duplicate_connected_component_count"])
        self.assertEqual("image", support["metric_unit"])
        self.assertEqual(support, group_support(list(reversed(rows))))
        self.assertEqual(before, rows)
        repeated = group_support(rows + [dict(rows[0])])
        self.assertEqual(7, repeated["image_count"])
        self.assertEqual(2, repeated["duplicate_connected_component_count"])

    def test_empty(self):
        support = group_support([])
        for key in ("image_count", "declared_group_count", "duplicate_connected_component_count"):
            self.assertEqual(0, support[key])

    def test_invalid_identifiers_rejected(self):
        for field in ("group", "sha256"):
            for value in (None, "", " ", 1, [], {}):
                with self.subTest(field=field, value=value):
                    row = dict(group="case", sha256="hash")
                    row[field] = value
                    with self.assertRaises(ValueError):
                        group_support([row])


if __name__ == "__main__":
    unittest.main()
