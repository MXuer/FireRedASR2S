import json
import unittest

from semantic_asr.adapters.wtpsplit_boundary import (
    WtpsplitBoundaryConfig,
    WtpsplitBoundaryPunc,
    parse_wtpsplit_end_indices,
)


class WtpsplitBoundaryTest(unittest.TestCase):
    def test_valid_end_indices_create_tagged_sentences_without_rewriting_text(self):
        calls = []

        def requester(body):
            calls.append(body)
            return json.dumps({
                "text": "changed text must be ignored",
                "token_count": 4,
                "end_indices": [1, 3],
            })

        punc = WtpsplitBoundaryPunc(
            WtpsplitBoundaryConfig(base_url="http://127.0.0.1:11001", language="th_th"),
            requester=requester,
        )
        result = punc.process_with_timestamp(
            [[
                ["วันนี้", 0.0, 0.4],
                ["ฝนตก", 0.4, 0.9],
                ["เจ้าหน้าที่", 1.2, 1.7],
                ["เตือนภัย", 1.7, 2.1],
            ]],
            ["utt"],
        )[0]

        self.assertEqual(calls[0]["language"], "th_th")
        self.assertEqual(calls[0]["tokens"], ["วันนี้", "ฝนตก", "เจ้าหน้าที่", "เตือนภัย"])
        self.assertEqual(
            [item["punc_text"] for item in result["punc_sentences"]],
            ["วันนี้ฝนตก", "เจ้าหน้าที่เตือนภัย"],
        )
        self.assertEqual(
            [item["boundary_source"] for item in result["punc_sentences"]],
            ["wtpsplit", "wtpsplit"],
        )
        self.assertTrue(all(item["semantic_boundary"] for item in result["punc_sentences"]))

    def test_short_span_merges_into_previous_sentence(self):
        def requester(body):
            return json.dumps({
                "token_count": 4,
                "end_indices": [0, 1, 3],
            })

        punc = WtpsplitBoundaryPunc(
            WtpsplitBoundaryConfig(min_span_s=1.0),
            requester=requester,
        )
        result = punc.process_with_timestamp(
            [[
                ["ก่อนหน้า", 0.0, 2.0],
                ["2คือ", 2.0, 2.4],
                ["หลัง", 2.4, 3.2],
                ["ต่อ", 3.2, 4.0],
            ]],
            ["utt"],
        )[0]

        self.assertEqual(
            [item["punc_text"] for item in result["punc_sentences"]],
            ["ก่อนหน้า2คือ", "หลังต่อ"],
        )

    def test_parser_accepts_spans_as_compatibility_input(self):
        self.assertEqual(
            parse_wtpsplit_end_indices(
                json.dumps({
                    "token_count": 3,
                    "spans": [{"start": 0, "end": 1}, {"start": 2, "end": 2}],
                }),
                3,
            ),
            [1, 2],
        )


if __name__ == "__main__":
    unittest.main()
