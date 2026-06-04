# Qwen3-ASR-1.7B

Source: https://github.com/QwenLM/Qwen3-ASR

## Install

The current environment already has `qwen-asr==0.0.6` installed from the
official GitHub repository. The package currently needs the torch pytree
compatibility shim in `semantic_asr.compat` when used with
`torch==2.1.0+cu118` and `transformers==4.57.6`.

`requirements.txt` records:

```text
qwen-asr @ git+https://github.com/QwenLM/Qwen3-ASR.git
```

## Language Support

Qwen3-ASR supports these 30 languages:

`Chinese`, `English`, `Cantonese`, `Arabic`, `German`, `French`, `Spanish`,
`Portuguese`, `Indonesian`, `Italian`, `Korean`, `Russian`, `Thai`,
`Vietnamese`, `Japanese`, `Turkish`, `Hindi`, `Malay`, `Dutch`, `Swedish`,
`Danish`, `Finnish`, `Polish`, `Czech`, `Filipino`, `Persian`, `Greek`,
`Hungarian`, `Macedonian`, and `Romanian`.

The official model API expects full language names rather than short codes.
Users configure the canonical pipeline id such as `th_th`; the adapter sends
`Thai` to `Qwen3ASRModel.transcribe`.

The related Qwen3 forced aligner is a timestamp provider and currently records
11 supported languages: `de`, `en`, `es`, `fr`, `it`, `ja`, `ko`, `pt`, `ru`,
`th`, and `zh`.

## Timestamps And Punctuation

Qwen3-ASR text output can include punctuation. For this pipeline, the default
ASR adapter does not require native word timestamps; pair it with
`qwen3_forced_aligner` or another forced aligner unless
`return_time_stamps=True` is verified for the exact checkpoint.

The adapter supports batch transcription through
`Qwen3ASRModel.transcribe(audio=[...])`.

## Standalone Test

Configuration/import shape only:

```bash
conda run -n fireredasr2s python semantic_asr/examples/test_qwen3_asr.py --skip_model_load 1
```

Real model smoke test:

```bash
CUDA_VISIBLE_DEVICES=4 conda run -n fireredasr2s python semantic_asr/examples/test_qwen3_asr.py \
  --wav_path data/test/short.wav \
  --language en_us \
  --max_seconds 30
```
