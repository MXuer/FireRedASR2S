import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from semantic_asr.adapters.xlm_roberta_punctuation import (
    XlmRobertaPunctuation,
    XlmRobertaPunctuationConfig,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="hello world how are you")
    parser.add_argument("--model", default="pcs_47lang")
    parser.add_argument("--skip_model_load", type=int, default=0)
    args = parser.parse_args()

    config = XlmRobertaPunctuationConfig(model=args.model)
    timestamp = [[token, i * 0.2, (i + 1) * 0.2] for i, token in enumerate(args.text.split())]
    if args.skip_model_load:
        print(json.dumps({"config": config.__dict__, "timestamp": timestamp}, ensure_ascii=False, indent=2))
        return

    punc = XlmRobertaPunctuation(config)
    result = punc.process_with_timestamp([timestamp], ["test"])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
