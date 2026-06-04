# Dolphin

Source: https://github.com/DataoceanAI/Dolphin

## Install

Dolphin must be installed from the official GitHub repository, not PyPI:

```bash
pip uninstall -y dataoceanai-dolphin dolphin
pip install "dolphin @ git+https://github.com/DataoceanAI/Dolphin.git"
```

After installation, verify the imported package points to the GitHub version
before accepting real-model smoke test results.

Model files are expected under:

```text
~/.cache/dolphin
```

The adapter expands this path with `os.path.expanduser`.

Dolphin imports `transformers`, so the adapter calls
`semantic_asr.compat.patch_torch_pytree_for_transformers()` before importing
the package in this environment.

## Language Support

Dolphin supports 40 Eastern languages plus Chinese dialect region codes. The
pipeline catalog records the broad language codes exposed by
`dolphin.languages.LANGUAGE_CODES`, including `zh`, `ja`, `th`, `ru`, `ko`,
`id`, `vi`, `ct`, `hi`, `ur`, `ms`, `uz`, `ar`, `fa`, `bn`, `ta`, `te`, `ug`,
`gu`, `my`, `tl`, `kk`, `or`, `ne`, `mn`, `km`, `jv`, `lo`, `si`, `fil`, `ps`,
`pa`, `kab`, `ba`, `ks`, `tg`, `su`, `mr`, `ky`, and `az`.

Use adapter params `lang_sym` and `region_sym` for Dolphin's official language
and dialect symbols.

## Timestamps And Punctuation

The adapter calls `dolphin.transcribe(..., word_timestamp=True)` by default and
normalizes returned word/token timestamps into pipeline format.

Do not use Dolphin's `predict_time` for this pipeline's mandatory word-level
timestamp contract; that parameter is sentence-level timing.

Dolphin can emit punctuated text, so it can be paired with `asr_text` or
`asr_native` punctuation strategies after verifying exact timestamp token
output.

Dolphin's public transcribe API is single-file oriented, so this adapter does
not mark batch inference support.

## Standalone Test

Configuration/import shape only:

```bash
conda run -n fireredasr2s python semantic_asr/examples/test_dolphin.py --skip_model_load 1
```

Real model smoke test:

```bash
CUDA_VISIBLE_DEVICES=4 conda run -n fireredasr2s python semantic_asr/examples/test_dolphin.py \
  --wav_path data/test/short.wav \
  --lang_sym zh \
  --word_timestamp 1 \
  --max_seconds 30
```
