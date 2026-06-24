import unittest

from semantic_asr.sidecars.wtpsplit_server import segments_to_end_indices


class WtpsplitServerTest(unittest.TestCase):
    def test_maps_thai_segments_back_to_token_end_indices(self):
        self.assertEqual(
            segments_to_end_indices(
                ["วันนี้", "ฝนตก", "เจ้าหน้าที่", "เตือนภัย"],
                ["วันนี้ฝนตก", "เจ้าหน้าที่เตือนภัย"],
            ),
            [1, 3],
        )

    def test_snaps_boundary_inside_coarse_token_to_token_end(self):
        self.assertEqual(
            segments_to_end_indices(
                ["วันนี้เราจะพูดถึงความคืบหน้าของโครงการ", "จากนั้นจะดูแผนงาน"],
                ["วันนี้", "เราจะพูดถึงความคืบหน้าของโครงการ", "จากนั้นจะดูแผนงาน"],
            ),
            [0, 1],
        )


if __name__ == "__main__":
    unittest.main()
