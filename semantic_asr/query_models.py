import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from semantic_asr.language_support import list_languages_by_model, list_models_by_language


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    language_parser = subparsers.add_parser("language")
    language_parser.add_argument("language")
    language_parser.add_argument("--role", default=None, choices=("vad", "asr", "timestamp", "punc"))

    model_parser = subparsers.add_parser("model")
    model_parser.add_argument("model")
    model_parser.add_argument("--role", default=None, choices=("vad", "asr", "timestamp", "punc"))

    args = parser.parse_args()
    if args.command == "language":
        result = list_models_by_language(args.language, role=args.role)
    else:
        result = list_languages_by_model(args.model, role=args.role)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
