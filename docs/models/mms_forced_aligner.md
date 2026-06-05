# MMS Forced Aligner

The adapter uses the project-owned MMS alignment runtime:

```python
from semantic_asr.adapters.mms_forced_aligner import (
    MmsForcedAlignerConfig,
    MmsForcedAlignerTimestampProvider,
)

provider = MmsForcedAlignerTimestampProvider(
    MmsForcedAlignerConfig(
        model_path="pretrained_models/mmsalign/model.pt",
        device="cuda:0",
        language="zh_cn",
    )
)
```

The required source was extracted into `semantic_asr.mms_runtime`; the obsolete
top-level `l2s` source tree is not required.

## Pipeline Role

This is a `timestamp` provider named `mms_forced_aligner`.

It receives ASR text plus the matching VAD audio segment and returns token
timestamps as:

```python
[token, start_s, end_s]
```

The pipeline feeds MMS with raw VAD speech segments by default. Semantic
sentence merging is performed after token timestamps are available.

`use_star` is fixed to `False` in the adapter. This avoids inserting `<star>`
between every input token, which can otherwise force gaps between adjacent
tokens and make word timestamps less precise. Because `use_star=False` assigns
the full segment span across tokens, keeping MMS on raw VAD segments is
important to avoid expanding the first and last token into surrounding silence.

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

For Chinese, Korean and Japanese, the adapter automatically inserts spaces
around no-space script characters before alignment.

## Standalone Test

Configuration/tokenization shape only:

```bash
conda run -n fireredasr2s python examples/test_mms_forced_aligner.py \
  --skip_model_load 1 \
  --language zh_cn \
  --text "你好 世界"
```

Real alignment should be validated with a short segment fixture or a full
pipeline profile once `pretrained_models/mmsalign/model.pt` is available.
