# FunASR CT-Punc

Registry name: `ct_punc`

Role: `punc`

Language: Chinese (`zh_cn`, `zh`)

## Source

FunASR built-in punctuation model:

```python
from funasr import AutoModel

model = AutoModel(model="ct-punc")
res = model.generate(input="那今天的会就到这里吧 happy new year 明年见")
```

The local environment already has `funasr` installed. The model is expected to
be available from the local ModelScope/FunASR cache.

## Pipeline Behavior

`semantic_asr.adapters.ct_punc.CtPunc` receives timestamp tokens, reconstructs
plain text with Chinese tokens joined directly and English/digit runs separated
by spaces, calls FunASR `AutoModel(model="ct-punc")`, then maps the returned
punctuated `text` back onto the original timestamp sequence with
`split_text_by_punctuation()`.

The underlying FunASR CT-Transformer inference path asserts `len(data_in) == 1`,
so this adapter intentionally calls `generate()` one item at a time even when
the outer pipeline sends a punctuation batch.

`configs/zh_cn.json` and `configs/zh_cn_mms_starprobe.json` now use `ct_punc`
instead of `firered_punc`.

## Standalone Smoke

```bash
conda run -n fireredasr2s python -c "from funasr import AutoModel; model=AutoModel(model='ct-punc', disable_update=True); print(model.generate(input='那今天的会就到这里吧 happy new year 明年见'))"
```

Expected output includes punctuated text similar to:

```text
那今天的会就到这里吧，happy new year,明年见。
```

## Validation

```bash
conda run -n fireredasr2s python -m unittest tests.test_punctuation_strategies tests.test_language_support tests.test_config_runner
```
