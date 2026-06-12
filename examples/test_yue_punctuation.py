import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from semantic_asr.adapters.yue_punctuation import (
    YuePunctuation,
    YuePunctuationConfig,
    punctuate_tokens_from_labels,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="我 今日 返工 你 去 邊")
    parser.add_argument("--model_name_or_path", default="nizzzo/zh-yue-punctuation-restore-v3")
    parser.add_argument("--local_model_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_model_load", type=int, default=0)
    args = parser.parse_args()

    config = YuePunctuationConfig(
        model_name_or_path=args.model_name_or_path,
        local_model_dir=args.local_model_dir,
        device=args.device,
    )
    tokens = [token for token in args.text.split() if token]
    timestamp = [[token, i * 0.2, (i + 1) * 0.2] for i, token in enumerate(tokens)]
    if args.skip_model_load:
        labels = ["O"] * max(0, len(tokens) - 1) + ["QUESTION_MARK"] if tokens else []
        preview = punctuate_tokens_from_labels(tokens, labels)
        print(json.dumps({"config": config.__dict__, "timestamp": timestamp, "preview": preview}, ensure_ascii=False, indent=2))
        return

    punc = YuePunctuation(config)
    result = punc.process_with_timestamp([timestamp], ["test"])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
