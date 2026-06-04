import argparse
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from typing import Callable

import soundfile as sf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from semantic_asr.adapters.dolphin import DolphinAsr, DolphinAsrConfig
from semantic_asr.adapters.firered import build_firered_punc
from semantic_asr.adapters.funasr_nano import FunAsrNano, FunAsrNanoConfig, FunAsrNanoTimestampProvider
from semantic_asr.adapters.mms_forced_aligner import MmsForcedAlignerConfig, MmsForcedAlignerTimestampProvider
from semantic_asr.adapters.qwen3_asr import Qwen3Asr, Qwen3AsrConfig
from semantic_asr.adapters.qwen3_forced_aligner import Qwen3ForcedAlignerConfig, Qwen3ForcedAlignerTimestampProvider
from semantic_asr.adapters.seamless_m4t import SeamlessM4TAsr, SeamlessM4TConfig
from semantic_asr.adapters.silero import SileroVad, SileroVadConfig
from semantic_asr.adapters.whisper_large import WhisperLarge, WhisperLargeConfig, WhisperLargeTimestampProvider
from semantic_asr.adapters.xlm_roberta_punctuation import XlmRobertaPunctuation, XlmRobertaPunctuationConfig
from semantic_asr.core import SpeechSegment
from semantic_asr.firered_runtime.fireredvad import FireRedVad, FireRedVadConfig


LANGUAGES = {
    "ar_sa": {"file": "ar_sa.wav", "base": "ar", "qwen": "ar", "whisper": "ar", "seamless": "arb", "mms": "ar_sa", "dolphin": "ar"},
    "en_us": {"file": "en_us.wav", "base": "en", "qwen": "en", "whisper": "en", "seamless": "eng", "mms": "en_us", "dolphin": None},
    "hi_in": {"file": "hi_in.wav", "base": "hi", "qwen": "hi", "whisper": "hi", "seamless": "hin", "mms": "hi_in", "dolphin": "hi"},
    "ja_jp": {"file": "ja_jp.wav", "base": "ja", "qwen": "ja", "whisper": "ja", "seamless": "jpn", "mms": "ja_jp", "dolphin": "ja"},
    "ko_kr": {"file": "ko_kr.wav", "base": "ko", "qwen": "ko", "whisper": "ko", "seamless": "kor", "mms": "ko_kr", "dolphin": "ko"},
    "pt_br": {"file": "pt_br.wav", "base": "pt", "qwen": "pt", "whisper": "pt", "seamless": "por", "mms": "pt_br", "dolphin": None},
    "ru_ru": {"file": "ru_ru.wav", "base": "ru", "qwen": "ru", "whisper": "ru", "seamless": "rus", "mms": "ru_ru", "dolphin": "ru"},
    "th_th": {"file": "th_th.wav", "base": "th", "qwen": "th", "whisper": "th", "seamless": "tha", "mms": "th_th", "dolphin": "th"},
    "vi_vn": {"file": "vi_vn.wav", "base": "vi", "qwen": "vi", "whisper": "vi", "seamless": "vie", "mms": "vi_in", "dolphin": "vi"},
}


PUNC_TEXTS = {
    "en_us": "hello world how are you today",
    "ru_ru": "привет мир как дела сегодня",
    "ja_jp": "こんにちは 世界 今日は 元気 です",
    "hi_in": "नमस्ते दुनिया आप कैसे हैं",
    "ar_sa": "مرحبا بالعالم كيف حالك اليوم",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", default="data/test")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--max_seconds", type=float, default=8.0)
    parser.add_argument("--models", default="silero_vad,firered_vad,funasr_nano,whisper_large,qwen3_asr,dolphin,seamless_m4t,xlm_roberta_punc,firered_punc,qwen3_aligner,mms_aligner")
    parser.add_argument("--languages", default=None, help="Comma-separated language ids from the built-in test matrix.")
    parser.add_argument("--hub", default="hf")
    args = parser.parse_args()

    if args.languages:
        requested_languages = {item.strip() for item in args.languages.split(",") if item.strip()}
        unknown_languages = requested_languages.difference(LANGUAGES)
        if unknown_languages:
            raise ValueError(f"Unknown test languages: {sorted(unknown_languages)}")
        for language in list(LANGUAGES):
            if language not in requested_languages:
                del LANGUAGES[language]

    outdir = Path(args.outdir or f"output/model_smoke_tests/{time.strftime('%Y%m%d_%H%M%S')}")
    outdir.mkdir(parents=True, exist_ok=True)
    selected = {item.strip() for item in args.models.split(",") if item.strip()}

    context = {
        "audio_dir": args.audio_dir,
        "max_seconds": args.max_seconds,
        "languages": collect_audio_info(args.audio_dir),
    }
    results: list[dict] = []
    asr_texts: dict[str, str] = {}
    segments: dict[str, SpeechSegment] = {}

    def record(test_name: str, fn: Callable[[], dict]):
        started = time.time()
        try:
            payload = fn()
            status = "pass"
        except Exception:
            payload = {"traceback": traceback.format_exc()}
            status = "fail"
        item = {
            "test": test_name,
            "status": status,
            "elapsed_s": round(time.time() - started, 3),
            "payload": payload,
        }
        results.append(item)
        (outdir / f"{test_name}.json").write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"test": test_name, "status": status, "elapsed_s": item["elapsed_s"]}, ensure_ascii=False), flush=True)
        cleanup_cuda()

    if "silero_vad" in selected:
        record("silero_vad_all_languages", lambda: run_silero(args.audio_dir, args.max_seconds))
    if "firered_vad" in selected:
        record("firered_vad_representative", lambda: run_firered_vad(args.audio_dir, args.max_seconds))
    if "funasr_nano" in selected:
        record("funasr_nano_all_languages", lambda: run_funasr(args.audio_dir, args.max_seconds, args.hub, asr_texts, segments))
    if "whisper_large" in selected:
        record("whisper_large_all_languages", lambda: run_whisper(args.audio_dir, args.max_seconds, asr_texts, segments))
    if "qwen3_asr" in selected:
        record("qwen3_asr_all_languages", lambda: run_qwen3_asr(args.audio_dir, args.max_seconds, asr_texts, segments))
    if "dolphin" in selected:
        record("dolphin_supported_languages", lambda: run_dolphin(args.audio_dir, args.max_seconds, asr_texts, segments))
    if "seamless_m4t" in selected:
        record("seamless_m4t_all_languages", lambda: run_seamless(args.audio_dir, args.max_seconds, asr_texts, segments))
    if "xlm_roberta_punc" in selected:
        record("xlm_roberta_punctuation_samples", run_xlm_roberta_punc)
    if "firered_punc" in selected:
        record("firered_punc_chinese_sample", run_firered_punc)
    if "qwen3_aligner" in selected:
        record("qwen3_forced_aligner_en_sample", lambda: run_qwen3_aligner(asr_texts, segments))
    if "mms_aligner" in selected:
        record("mms_forced_aligner_hi_sample", lambda: run_mms_aligner(asr_texts, segments))

    summary = {
        "context": context,
        "results": results,
        "pass_count": sum(1 for item in results if item["status"] == "pass"),
        "fail_count": sum(1 for item in results if item["status"] == "fail"),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(outdir / "summary.json"), "pass_count": summary["pass_count"], "fail_count": summary["fail_count"]}, ensure_ascii=False))
    if summary["fail_count"]:
        raise SystemExit(1)


def collect_audio_info(audio_dir: str) -> dict:
    info = {}
    for language, meta in LANGUAGES.items():
        path = Path(audio_dir) / meta["file"]
        if not path.exists():
            info[language] = {"exists": False, "path": str(path)}
            continue
        sf_info = sf.info(path)
        info[language] = {
            "exists": True,
            "path": str(path),
            "sample_rate": sf_info.samplerate,
            "channels": sf_info.channels,
            "duration_s": round(sf_info.duration, 3),
        }
    return info


def load_audio(audio_dir: str, language: str, max_seconds: float) -> tuple[int, object]:
    path = Path(audio_dir) / LANGUAGES[language]["file"]
    wav, sample_rate = sf.read(path, dtype="int16")
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1).astype("int16")
    if max_seconds > 0:
        wav = wav[: int(max_seconds * sample_rate)]
    return sample_rate, wav


def make_segment(language: str, sample_rate: int, wav) -> SpeechSegment:
    return SpeechSegment(language, 0.0, len(wav) / sample_rate, sample_rate, wav)


def summarize_asr(results: list[dict]) -> dict:
    return {
        item["uttid"]: {
            "text_len": len(item.get("text", "")),
            "text_preview": item.get("text", "")[:120],
            "timestamp_count": len(item.get("timestamp") or []),
        }
        for item in results
    }


def run_silero(audio_dir: str, max_seconds: float) -> dict:
    vad = SileroVad(SileroVadConfig())
    output = {}
    for language in LANGUAGES:
        sample_rate, wav = load_audio(audio_dir, language, max_seconds)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sf.write(tmp.name, wav, sample_rate)
        try:
            result = vad.detect(tmp.name)
        finally:
            os.unlink(tmp.name)
        output[language] = {"segment_count": len(result.get("timestamps") or []), "first_segments": (result.get("timestamps") or [])[:3]}
    return output


def run_firered_vad(audio_dir: str, max_seconds: float) -> dict:
    vad = FireRedVad.from_pretrained("pretrained_models/FireRedVAD/VAD", FireRedVadConfig())
    output = {}
    for language in (item for item in ("en_us", "ru_ru", "th_th") if item in LANGUAGES):
        sample_rate, wav = load_audio(audio_dir, language, max_seconds)
        segment = make_segment(language, sample_rate, wav)
        result, _ = vad.detect(segment.wav)
        output[language] = {"segment_count": len(result.get("timestamps") or []), "first_segments": (result.get("timestamps") or [])[:3]}
    return output


def run_funasr(audio_dir: str, max_seconds: float, hub: str, asr_texts: dict, segments: dict) -> dict:
    local_model = os.path.expanduser("~/.cache/modelscope/hub/models/FunAudioLLM/Fun-ASR-Nano-2512")
    use_local_model = os.path.isdir(local_model)
    model_name = local_model if use_local_model else "FunAudioLLM/Fun-ASR-Nano-2512"
    model_hub = "ms" if use_local_model else hub
    model = FunAsrNano(FunAsrNanoConfig(model=model_name, device="cuda:0", hub=model_hub, batch_size=1, preserve_punctuation=True))
    languages = [language for language, meta in LANGUAGES.items() if meta["base"] in {"zh", "en", "ja"}]
    results = []
    for language in languages:
        model.config.language = language
        sample_rate, wav = load_audio(audio_dir, language, max_seconds)
        segment = make_segment(language, sample_rate, wav)
        item = model.transcribe([language], [(sample_rate, wav)])[0]
        results.extend(FunAsrNanoTimestampProvider().add_timestamps([item], [segment]))
        segments.setdefault(language, segment)
    for item in results:
        if item.get("text"):
            asr_texts.setdefault(item["uttid"], item["text"])
    return summarize_asr(results)


def run_whisper(audio_dir: str, max_seconds: float, asr_texts: dict, segments: dict) -> dict:
    model = WhisperLarge(WhisperLargeConfig(device="cuda:0"))
    provider = WhisperLargeTimestampProvider()
    outputs = []
    for language, meta in LANGUAGES.items():
        model.config.language = language
        sample_rate, wav = load_audio(audio_dir, language, max_seconds)
        segment = make_segment(language, sample_rate, wav)
        result = model.transcribe([language], [(sample_rate, wav)])
        result = provider.add_timestamps(result, [segment])
        outputs.extend(result)
        segments.setdefault(language, segment)
        if result[0].get("text"):
            asr_texts.setdefault(language, result[0]["text"])
    return summarize_asr(outputs)


def run_qwen3_asr(audio_dir: str, max_seconds: float, asr_texts: dict, segments: dict) -> dict:
    model = Qwen3Asr(Qwen3AsrConfig(model=cached_snapshot("Qwen/Qwen3-ASR-1.7B"), device_map="cuda:0", max_inference_batch_size=4, max_new_tokens=128))
    results = []
    for language, meta in LANGUAGES.items():
        model.config.language = language
        sample_rate, wav = load_audio(audio_dir, language, max_seconds)
        segment = make_segment(language, sample_rate, wav)
        item = model.transcribe([language], [(sample_rate, wav)])[0]
        results.append(item)
        segments.setdefault(language, segment)
        if item.get("text"):
            asr_texts.setdefault(language, item["text"])
    return summarize_asr(results)


def run_dolphin(audio_dir: str, max_seconds: float, asr_texts: dict, segments: dict) -> dict:
    model = DolphinAsr(DolphinAsrConfig(device="cuda:0", word_timestamp=True))
    results = []
    for language, meta in LANGUAGES.items():
        if not meta["dolphin"]:
            continue
        model.config.language = language
        sample_rate, wav = load_audio(audio_dir, language, max_seconds)
        segment = make_segment(language, sample_rate, wav)
        item = model.transcribe([language], [(sample_rate, wav)])[0]
        results.append(item)
        segments.setdefault(language, segment)
        if item.get("text"):
            asr_texts.setdefault(language, item["text"])
    return summarize_asr(results)


def run_seamless(audio_dir: str, max_seconds: float, asr_texts: dict, segments: dict) -> dict:
    model = SeamlessM4TAsr(SeamlessM4TConfig(
        model=cached_snapshot("facebook/seamless-m4t-v2-large"),
        device="cuda:0",
    ))
    results = []
    for language, meta in LANGUAGES.items():
        model.config.language = language
        sample_rate, wav = load_audio(audio_dir, language, max_seconds)
        segment = make_segment(language, sample_rate, wav)
        item = model.transcribe([language], [(sample_rate, wav)])[0]
        results.append(item)
        segments.setdefault(language, segment)
        if item.get("text"):
            asr_texts.setdefault(language, item["text"])
    return summarize_asr(results)


def run_xlm_roberta_punc() -> dict:
    model = XlmRobertaPunctuation(XlmRobertaPunctuationConfig())
    output = {}
    for language, text in PUNC_TEXTS.items():
        timestamp = [[token, index * 0.2, (index + 1) * 0.2] for index, token in enumerate(text.split())]
        result = model.process_with_timestamp([timestamp], [language])[0]
        sentences = result.get("punc_sentences", [])
        output[language] = {"sentence_count": len(sentences), "preview": sentences[:2]}
    return output


def run_firered_punc() -> dict:
    model = build_firered_punc()
    timestamp = [["今天", 0.0, 0.2], ["天气", 0.2, 0.4], ["很好", 0.4, 0.6], ["我们", 0.8, 1.0], ["开始", 1.0, 1.2], ["测试", 1.2, 1.4]]
    result = model.process_with_timestamp([timestamp], ["zh_sample"])[0]
    sentences = result.get("punc_sentences", [])
    return {"sentence_count": len(sentences), "preview": sentences[:3]}


def run_qwen3_aligner(asr_texts: dict, segments: dict) -> dict:
    language = "en_us" if "en_us" in asr_texts and "en_us" in segments else next(iter(asr_texts))
    model = Qwen3ForcedAlignerTimestampProvider(Qwen3ForcedAlignerConfig(model=cached_snapshot("Qwen/Qwen3-ForcedAligner-0.6B"), device_map="cuda:0", language=language, batch_size=1))
    asr_result = {"uttid": language, "text": asr_texts[language], "confidence": 0, "timestamp": []}
    result = model.add_timestamps([asr_result], [segments[language]])[0]
    return summarize_asr([result])


def run_mms_aligner(asr_texts: dict, segments: dict) -> dict:
    language = "hi_in" if "hi_in" in asr_texts and "hi_in" in segments else next(iter(asr_texts))
    model = MmsForcedAlignerTimestampProvider(MmsForcedAlignerConfig(device="cuda:0", language=language))
    asr_result = {"uttid": language, "text": asr_texts[language], "confidence": 0, "timestamp": []}
    result = model.add_timestamps([asr_result], [segments[language]])[0]
    return summarize_asr([result])


def cleanup_cuda():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def cached_snapshot(repo_id: str) -> str:
    repo_dir = Path(os.path.expanduser("~/.cache/huggingface/hub")) / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    snapshots = sorted(path for path in repo_dir.glob("*") if path.is_dir())
    return str(snapshots[-1]) if snapshots else repo_id


if __name__ == "__main__":
    main()
