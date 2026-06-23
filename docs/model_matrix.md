# Model Matrix

This matrix summarizes the current model adapters and capability metadata. The
source of truth is `semantic_asr/language_support.py`; update both the metadata
and this document when adding or changing a model.

## VAD

| Model | Languages | Batch | Native timestamps | Native punctuation | Notes |
| --- | --- | --- | --- | --- | --- |
| `silero` | `*` | no | no | no | Language-agnostic speech activity detector. |
| `firered_vad` | `*` | no | no | no | Language-agnostic in the current pipeline contract. |
| `ten_vad` | `*` | no | no | no | TEN VAD adapter, 16 kHz, 256-sample frames. |

## ASR

| Model | Languages | Batch | Native timestamps | Native punctuation | Notes |
| --- | --- | --- | --- | --- | --- |
| `funasr_nano` | `zh`, `en`, `ja` | yes | yes | yes | Fun-ASR-Nano-2512. |
| `firered_asr` | `zh` | yes | yes | no | FireRedASR2 AED Chinese ASR. Use `return_timestamp=true` with `firered_asr_native`. |
| `whisper_large` | multilingual Whisper set | adapter-dependent | yes | yes | Word timestamps requested by adapter config. |
| `qwen3_asr_1_7b` | `ar`, `cs`, `da`, `de`, `el`, `en`, `es`, `fa`, `fi`, `fil`, `fr`, `hi`, `hu`, `id`, `it`, `ja`, `ko`, `mk`, `ms`, `nl`, `pl`, `pt`, `ro`, `ru`, `sv`, `th`, `tr`, `vi`, `yue`, `zh` | yes | no | yes | Adapter maps canonical ids to full model language names. |
| `dolphin` | 40 Eastern languages plus Chinese dialect aliases | no | yes | yes | Word timestamps require `word_timestamp`; `predict_time` is sentence-level only. |
| `seamless_m4t_v2_large` | currently mapped: `ar`, `en`, `hi`, `ja`, `ko`, `pt`, `ru`, `th`, `vi`, `zh` | no | no | yes | Adapter maps canonical ids to Seamless/FLORES codes. |

## Timestamp Providers

| Model | Languages | Batch | Native timestamps | Native punctuation | Notes |
| --- | --- | --- | --- | --- | --- |
| `funasr_native` | `zh`, `en`, `ja` | no | validates ASR-native | no | Validates Fun-ASR-Nano token timestamps. |
| `firered_asr_native` | `zh` | no | validates ASR-native | no | Validates FireRedASR2 timestamps. |
| `whisper_native` | `*` | no | validates ASR-native | no | Validates Whisper word timestamps. |
| `qwen3_forced_aligner` | `de`, `en`, `es`, `fr`, `it`, `ja`, `ko`, `pt`, `ru`, `th`, `zh` | yes | forced alignment | no | Official Qwen3 forced aligner. |
| `mms_forced_aligner` | `ar_sa`, `bg_bg`, `bn_bd`, `de_de`, `en_us`, `en_gb`, `es_mx`, `fa_ir`, `fr_fr`, `hi_in`, `ja_jp`, `ko_kr`, `pt_br`, `ru_ru`, `th_th`, `vi_vn`, `zh_cn`, and other mapped MMS ids | adapter supports parallel workers | forced alignment | no | Vendored MMS runtime. CJK should be treated as char-level. |

## Punctuation / Boundary

| Model | Languages | Batch | Native timestamps | Native punctuation | Notes |
| --- | --- | --- | --- | --- | --- |
| `firered_punc` | `zh` | no | no | external | Chinese punctuation. |
| `ct_punc` | `zh` | no | no | external | FunASR `ct-punc`; current FunASR model asserts single input. |
| `asr_native` | `*` | no | no | ASR-native | Uses ASR-emitted punctuation. |
| `asr_text` | `*` | no | no | ASR text | Splits existing ASR text by punctuation. |
| `qwen_semantic_boundary` | `*` | no | no | boundary strategy | Uses Qwen to return index-only semantic boundaries. Must validate JSON coverage and never trust rewritten text. |
| `naqta` | `ar` | no | no | external | Arabic punctuation restoration. |
| `yue_punctuation` | `yue` | no | no | external | Cantonese punctuation restoration. |
| `xlm_roberta_punctuation` | 47 languages including `zh`, `en`, `ar`, `de`, `ja`, `ko`, `pt`, `ru`, `vi` | no | no | external | Quality must be validated per language/domain. |

## Translation

Translation is not part of the mandatory four-stage ASR pipeline.

| Model | Role | Notes |
| --- | --- | --- |
| Hunyuan-MT | translation sidecar | Used by service/WebUI for bilingual review. Chinese-source to `zh_cn` should write identity translation and skip GPU. |

## Update Checklist

When adding a model:

1. Add adapter code.
2. Register it in `semantic_asr/registry.py`.
3. Add language and capability metadata in `semantic_asr/language_support.py`.
4. Add `docs/models/<model>.md`.
5. Add a smoke example or unit test.
6. Update this matrix.
7. Record install/model path notes in `tasks/progress.md` if the local environment changed.
