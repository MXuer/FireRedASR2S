import json
import unittest

from semantic_asr.adapters.qwen_semantic_boundary import (
    QwenSemanticBoundaryConfig,
    QwenSemanticBoundaryPunc,
    fallback_duration_end_indices,
    parse_boundary_end_indices,
)


class QwenSemanticBoundaryTest(unittest.TestCase):
    def test_valid_end_indices_create_sentences_without_rewriting_text(self):
        calls = []

        def requester(body):
            calls.append(body)
            return json.dumps({
                "token_count": 4,
                "end_indices": [1, 3],
                "text": "changed text must be ignored",
            })

        punc = QwenSemanticBoundaryPunc(
            QwenSemanticBoundaryConfig(language="th_th"),
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

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(calls[0]["response_format"], {"type": "json_object"})
        self.assertEqual(
            [item["punc_text"] for item in result["punc_sentences"]],
            ["วันนี้ฝนตก", "เจ้าหน้าที่เตือนภัย"],
        )

    def test_response_format_json_can_be_disabled_for_incompatible_backends(self):
        calls = []

        def requester(body):
            calls.append(body)
            return json.dumps({"token_count": 1, "end_indices": [0]})

        punc = QwenSemanticBoundaryPunc(
            QwenSemanticBoundaryConfig(response_format_json=False),
            requester=requester,
        )
        punc.process_with_timestamp([[["hello", 0.0, 0.2]]], ["utt"])

        self.assertNotIn("response_format", calls[0])

    def test_invalid_first_response_retries_with_validation_error(self):
        responses = iter([
            "not json",
            json.dumps({"token_count": 3, "end_indices": [0, 2]}),
        ])
        prompts = []

        def requester(body):
            prompts.append(body["messages"][-1]["content"])
            return next(responses)

        punc = QwenSemanticBoundaryPunc(
            QwenSemanticBoundaryConfig(max_retries=1),
            requester=requester,
        )
        result = punc.process_with_timestamp(
            [[["a", 0.0, 0.2], ["b", 0.2, 0.4], ["c", 0.4, 0.6]]],
            ["utt"],
        )[0]

        self.assertEqual(len(prompts), 2)
        self.assertIn("previous response was invalid", prompts[1])
        self.assertEqual([item["punc_text"] for item in result["punc_sentences"]], ["a", "b c"])

    def test_malformed_responses_fall_back_to_duration_boundaries(self):
        def requester(body):
            return json.dumps({"token_count": 4, "end_indices": [1]})

        punc = QwenSemanticBoundaryPunc(
            QwenSemanticBoundaryConfig(max_retries=1, fallback_max_span_s=1.0),
            requester=requester,
        )
        result = punc.process_with_timestamp(
            [[
                ["one", 0.0, 0.6],
                ["two", 0.6, 1.1],
                ["three", 1.1, 1.7],
                ["four", 1.7, 2.1],
            ]],
            ["utt"],
        )[0]

        self.assertEqual(
            [item["punc_text"] for item in result["punc_sentences"]],
            ["one two", "three four"],
        )

    def test_parser_accepts_continuous_spans_as_compatibility_input(self):
        self.assertEqual(
            parse_boundary_end_indices(
                json.dumps({
                    "token_count": 3,
                    "spans": [{"start": 0, "end": 1}, {"start": 2, "end": 2}],
                }),
                3,
            ),
            [1, 2],
        )

    def test_parser_rejects_non_continuous_or_incomplete_output(self):
        with self.assertRaises(ValueError):
            parse_boundary_end_indices(
                json.dumps({"token_count": 4, "end_indices": [1, 2]}),
                4,
            )
        with self.assertRaises(ValueError):
            parse_boundary_end_indices(
                json.dumps({
                    "token_count": 3,
                    "spans": [{"start": 0, "end": 0}, {"start": 2, "end": 2}],
                }),
                3,
            )

    def test_fallback_duration_end_indices_cover_all_tokens(self):
        self.assertEqual(
            fallback_duration_end_indices(
                [["de", 0.0, 0.7], ["und", 0.7, 1.4], ["ar", 1.4, 2.2]],
                1.0,
            ),
            [1, 2],
        )


if __name__ == "__main__":
    unittest.main()
