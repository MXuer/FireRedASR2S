# Test Audio Matrix

This project should keep a small multilingual audio set that validates the
current module catalog without trying to cover every supported language.

## Audio Requirements

For each language below, prepare:

- one clean short clip around 60 seconds for fast smoke tests;
- a normalized transcript for forced alignment checks;
- natural multi-sentence speech with real pauses and punctuation-worthy
  boundaries;
- mono 16 kHz WAV when possible, or a source file that can be converted
  deterministically before tests.

For long-audio behavior, keep one 10 minute or longer sample for the core
regression languages `zh_cn`, `en_us` and `ru_ru`.

## Must-Have Languages

| Language | Why it is needed | Main module paths covered |
| --- | --- | --- |
| `zh_cn` | Primary Chinese path and FireRed punctuation coverage. Also checks Chinese token splitting for aligners. | Silero/FunASR native timestamp/FireRedPunc; FireRed VAD; Qwen3-ASR + Qwen3 aligner or MMS aligner; Dolphin word timestamps; XLM-R punctuation for Chinese. |
| `en_us` | Broad baseline for Whisper, Qwen3-ASR, Seamless and MMS. | Whisper native timestamps/punctuation; Qwen3-ASR + Qwen3 aligner or MMS aligner; Seamless + forced aligner; XLM-R punctuation. |
| `ru_ru` | Existing Russian profile and non-Chinese multilingual regression. | FireRed VAD + Whisper + Qwen3-ForcedAligner + ASR text punctuation; Whisper/Qwen/Dolphin overlap; MMS forced aligner; XLM-R punctuation. |
| `ja_jp` | CJK language with different tokenization from Chinese. | FunASR, Whisper, Qwen3-ASR, Dolphin, Qwen3 aligner, MMS aligner and XLM-R punctuation. |
| `th_th` | No-space script; useful for sentence-boundary and alignment stress. | Whisper, Qwen3-ASR, Dolphin, Qwen3 aligner and MMS aligner. |
| `hi_in` | Indic script path where Qwen3-ForcedAligner is not available, forcing MMS coverage. | Whisper, Qwen3-ASR, Dolphin, MMS forced aligner and XLM-R punctuation. |

These six languages give coverage for:

- both VAD adapters: language-agnostic Silero and FireRed VAD;
- native timestamp ASR: FunASR, Whisper and Dolphin;
- external timestamp providers: Qwen3-ForcedAligner and MMS forced aligner;
- punctuation strategies: FireRedPunc, ASR-native punctuation, ASR-text splitting
  and XLM-R punctuation;
- spacing-sensitive scripts: Chinese/Japanese, Thai, Indic and Latin/Cyrillic.

## Optional Expansion Languages

| Language | Additional coverage |
| --- | --- |
| `ar_sa` | Right-to-left Arabic script; good for MMS, Qwen3-ASR, Dolphin and XLM-R punctuation. |
| `vi_vn` | Vietnamese tone marks and MMS language-code coverage. |
| `ko_kr` | Korean CJK-like segmentation; supported by Qwen3 aligner and MMS. |
| `bn_bd` | Bengali script; useful second Indic-style sample for MMS/XLM-R. |
| `pt_br` | Latin-script non-English sample with MMS region code and Qwen coverage. |

## Suggested Test Set Layout

```text
data/test_audio/
  zh_cn/
    short.wav
    short.txt
    long.wav
  en_us/
    short.wav
    short.txt
    long.wav
  ru_ru/
    short.wav
    short.txt
    long.wav
  ja_jp/
    short.wav
    short.txt
  th_th/
    short.wav
    short.txt
  hi_in/
    short.wav
    short.txt
```

## Test Intent

Use short clips for adapter output-shape tests and quick pipeline smoke tests.
Use transcripts to validate forced aligner token coverage and timestamp
monotonicity. Use long clips only for VAD merge/padding, sentence segmentation
and output writer regression checks.
