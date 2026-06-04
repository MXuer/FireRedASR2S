# MMS Forced Aligner

Reference implementation:

```python
from l2s.align.align import ALIGNER
from l2s.text_norm.text_normalize import LANG2TEXTNORMALIZER

aligner = ALIGNER(model_path="pretrained_models/mmsalign/model.pt", device="cuda:0")
texts = re.sub(r"[\u4e00-\u9fa5]", lambda x: " " + x[0] + " ", texts).split()
device_aligns = aligner.align(
    texts,
    device_wav_file,
    names,
    use_star=True,
    language="zh_cn",
    raw_transcripts=texts,
)
```

The snippet above is the source reference. The project implementation uses
vendored code under `semantic_asr.mms_runtime`, not the external `l2s` package.

## Pipeline Role

This is a `timestamp` provider named `mms_forced_aligner`.

It receives ASR text plus the matching VAD audio segment and returns token
timestamps as:

```python
[token, start_s, end_s]
```

## Install And Model Files

The MMS alignment runtime needed by the adapter is vendored inside
`semantic_asr.mms_runtime`; the adapter does not import an external `l2s`
package.

The default model path is:

```text
pretrained_models/mmsalign/model.pt
```

The current environment needs `torch` and `torchaudio`. The adapter aligns the
in-memory `SpeechSegment.wav` audio directly and does not write temporary
segment wav files.

## Language Support

Configure the canonical pipeline language id. The adapter maps it through
`semantic_asr.mms_runtime.model_registry.MMS_CODE_MAP`; for example, `zh_cn`
becomes MMSAlign's native `cmn`.

For Chinese, the adapter automatically inserts spaces around each CJK character
before alignment, matching the reference code.

## Standalone Test

Configuration/tokenization shape only:

```bash
conda run -n fireredasr2s python semantic_asr/examples/test_mms_forced_aligner.py \
  --skip_model_load 1 \
  --language zh_cn \
  --text "你好 世界"
```

Real alignment should be validated with a short segment fixture or a full
pipeline profile once `pretrained_models/mmsalign/model.pt` is available.
