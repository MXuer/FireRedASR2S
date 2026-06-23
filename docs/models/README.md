# Models

This directory records installation notes, local environment details and smoke
commands for individual model adapters.

For the full capability table, see [../model_matrix.md](../model_matrix.md).
For canonical language ids and model-native language mapping, see
[../language_configuration.md](../language_configuration.md).

## VAD Models

| Model | Component | Summary | Details |
| --- | --- | --- | --- |
| FireRed VAD | `firered_vad` | Local FireRed speech activity detector. Used as a strong general VAD option and can expose frame-level speech probabilities. | [firered_asr.md](firered_asr.md) |
| Silero VAD | `silero` | Lightweight language-agnostic VAD. Useful for quick multilingual experiments. | [silero_vad.md](silero_vad.md) |
| TEN VAD | `ten_vad` | Frame-based VAD using 16 kHz, 256-sample frames. Useful when testing stricter speech-island boundaries. | [ten_vad.md](ten_vad.md) |

## ASR Models

| Model | Component | Summary | Details |
| --- | --- | --- | --- |
| FireRedASR2 | `firered_asr` | Chinese ASR adapter using the vendored FireRed runtime. Can return native timestamps when configured. | [firered_asr.md](firered_asr.md) |
| Fun-ASR-Nano-2512 | `funasr_nano` | Chinese, English and Japanese ASR with native timestamps and punctuation. Supports batch inference. | [funasr_nano.md](funasr_nano.md) |
| Whisper large | `whisper_large` | Multilingual batched ASR with native punctuation. Use forced alignment for timestamps. | [whisper_large.md](whisper_large.md) |
| GigaAM-v3 | `gigaam_v3` | Russian e2e RNNT ASR with native punctuation, text normalization, word timestamps and batched inference. | [gigaam_v3.md](gigaam_v3.md) |
| Qwen3-ASR-1.7B | `qwen3_asr_1_7b` | Multilingual ASR covering 30 languages. Uses model-native full language names internally. | [qwen3_asr.md](qwen3_asr.md) |
| Dolphin | `dolphin` | Eastern-language ASR with word timestamp support through `word_timestamp`. | [dolphin.md](dolphin.md) |
| Seamless M4T v2 large | `seamless_m4t_v2_large` | Multilingual speech model currently mapped for selected validated languages. | [seamless_m4t_v2_large.md](seamless_m4t_v2_large.md) |

## Timestamp Providers

| Model | Component | Summary | Details |
| --- | --- | --- | --- |
| ASR native timestamp | `funasr_native`, `firered_asr_native`, `gigaam_v3_native` | Validates timestamps already emitted by the ASR model. | [funasr_nano.md](funasr_nano.md), [firered_asr.md](firered_asr.md), [gigaam_v3.md](gigaam_v3.md) |
| MMS forced aligner | `mms_forced_aligner` | Forced alignment provider using vendored MMS runtime. Important for ASR models without reliable token timestamps. | [mms_forced_aligner.md](mms_forced_aligner.md) |
| Qwen3 ForcedAligner | `qwen3_forced_aligner` | Qwen forced-alignment model for supported languages. | [qwen3_forced_aligner.md](qwen3_forced_aligner.md) |

## Punctuation And Boundary Models

| Model | Component | Summary | Details |
| --- | --- | --- | --- |
| ASR-native punctuation | `asr_native`, `asr_text` | Uses punctuation already present in ASR output, or splits existing ASR text by punctuation. | [../architecture/sentence_boundary_strategy.md](../architecture/sentence_boundary_strategy.md) |
| FireRedPunc | `firered_punc` | Chinese punctuation model from the FireRed family. | [firered_asr.md](firered_asr.md) |
| FunASR ct-punc | `ct_punc` | Chinese punctuation model. Current FunASR implementation should be called one item at a time. | [ct_punc.md](ct_punc.md) |
| XLM-R punctuation | `xlm_roberta_punctuation` | Multilingual punctuation/fullstop/truecase model supporting 47 languages. Quality should be verified per language/domain. | [xlm_roberta_punctuation_fullstop_truecase.md](xlm_roberta_punctuation_fullstop_truecase.md) |
| Naqta | `naqta` | Arabic punctuation restoration model. | [naqta.md](naqta.md) |
| Yue punctuation | `yue_punctuation` | Cantonese punctuation restoration model. | [yue_punctuation.md](yue_punctuation.md) |
| Qwen semantic boundary | `qwen_semantic_boundary` | LLM-based boundary strategy for languages or domains where punctuation models are weak. It must return validated spans and must not rewrite text. | [qwen_semantic_boundary.md](qwen_semantic_boundary.md) |

## Translation Models

Translation is not part of the mandatory VAD/ASR/timestamp/punctuation pipeline.
It is used by the service/WebUI as a review sidecar.

| Model | Summary | Details |
| --- | --- | --- |
| Hunyuan-MT | Optional translation service for bilingual review. Chinese-source to `zh_cn` should use identity translation and skip GPU. | [../hunyuan_mt_translation_design.md](../hunyuan_mt_translation_design.md) |

## Adding A Model

When adding a model, update all of these surfaces:

1. Adapter code under `semantic_asr/adapters/` or service code if it is a sidecar.
2. Registry entry in `semantic_asr/registry.py`.
3. Language and capability metadata in `semantic_asr/language_support.py`.
4. A model note in this directory.
5. A smoke example or unit test.
6. [../model_matrix.md](../model_matrix.md).
7. This README.
