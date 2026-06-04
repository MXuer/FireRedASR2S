import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from semantic_asr.adapters.mms_forced_aligner import MmsForcedAlignerConfig, MmsForcedAlignerTimestampProvider


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="pretrained_models/mmsalign/model.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--language", default="zh_cn")
    parser.add_argument("--text", default="你好 世界")
    parser.add_argument("--skip_model_load", type=int, default=1)
    args = parser.parse_args()

    config = MmsForcedAlignerConfig(
        model_path=args.model_path,
        device=args.device,
        language=args.language,
    )
    if args.skip_model_load:
        provider = object.__new__(MmsForcedAlignerTimestampProvider)
        provider.config = config
        print(json.dumps({
            "config": config.__dict__,
            "tokens": provider._prepare_tokens(args.text),
        }, ensure_ascii=False, indent=2))
        return

    raise SystemExit("Real MMS aligner smoke test should be run through a full pipeline or a segment fixture.")


if __name__ == "__main__":
    main()
