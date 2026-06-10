import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from semantic_asr.adapters.naqta_punctuation import (
    NaqtaPunctuation,
    NaqtaPunctuationConfig,
    punctuate_tokens_from_labels,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="هذا اختبار هل تسمعني")
    parser.add_argument("--model_name_or_path", default="MostafaMaroof/Naqta")
    parser.add_argument("--local_model_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_model_load", type=int, default=0)
    args = parser.parse_args()

    config = NaqtaPunctuationConfig(
        model_name_or_path=args.model_name_or_path,
        local_model_dir=args.local_model_dir,
        device=args.device,
    )
    timestamp = [[token, i * 0.2, (i + 1) * 0.2] for i, token in enumerate(args.text.split())]
    if args.skip_model_load:
        preview = punctuate_tokens_from_labels(
            args.text.split(),
            ["O"] * max(0, len(args.text.split()) - 1) + ["PERIOD"],
        )
        print(json.dumps({"config": config.__dict__, "timestamp": timestamp, "preview": preview}, ensure_ascii=False, indent=2))
        return

    punc = NaqtaPunctuation(config)
    result = punc.process_with_timestamp([timestamp], ["test"])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
