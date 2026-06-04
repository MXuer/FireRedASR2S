# Unified Language Configuration

Pipeline profiles configure language once, at the top level, using a canonical
lowercase language-region id such as `zh_cn`, `en_us`, or `th_th`.

Do not put model-native values such as `Chinese`, `Russian`, `eng`, `cmn`,
`lang_sym`, or `src_lang` in component parameters.

`semantic_asr.language_mapping` converts the profile language internally:

| Canonical id | Qwen3-ASR | Qwen3 aligner | FunASR | Whisper | MMSAlign | Seamless | Dolphin |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `zh_cn` | `Chinese` | `Chinese` | `Chinese` | `zh` | `cmn` | `cmn` | `zh`, `CN` |
| `en_us` | `English` | `English` | `English` | `en` | `eng` | `eng` | `en`, `US` |
| `ru_ru` | `Russian` | `Russian` | unsupported | `ru` | `rus` | `rus` | `ru`, `RU` |
| `th_th` | `Thai` | `Thai` | unsupported | `th` | `nod` | `tha` | `th`, `TH` |

Model adapters raise an error when their model does not support the configured
canonical language.
