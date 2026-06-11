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

The pipeline feeds MMS with ASR VAD segments derived from raw VAD. Very short
raw VAD islands are merged into nearby speech when possible, or skipped when
isolated, before ASR and MMS alignment. Semantic sentence merging is performed
after token timestamps are available.

`use_star` is fixed to `False` in the adapter. This avoids inserting `<star>`
between every input token, which can otherwise force gaps between adjacent
tokens and make word timestamps less precise. Because `use_star=False` assigns
the full segment span across tokens, ASR VAD postprocessing keeps MMS close to
real speech while avoiding isolated microsegments that are too short for CTC.

Before calling torchaudio forced alignment, the MMS runtime checks whether the
target sequence is alignable:

- empty target after uroman/dictionary filtering;
- CTC-impossible target where `frames < target_chars + repeats`.

Those cases fall back to monotonic approximate token timestamps for the current
ASR segment instead of failing the whole audio. The generated JSON records this
under `timestamp_segments[].timestamp_fallback`.

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

## Parallel Segment Workers

MMS alignment does not expose a native batch API in this adapter. For long audio
with many raw VAD segments, configure segment-level worker processes:

```json
{
  "components": {
    "timestamp": {
      "name": "mms_forced_aligner",
      "params": {
        "device": "cuda:0",
        "num_workers": 2
      }
    }
  }
}
```

Each worker process loads one MMS aligner instance. This improves throughput
when GPU memory allows multiple model copies.

## Language Support

Configure the canonical pipeline language id. The adapter maps it through
`semantic_asr.mms_runtime.model_registry.MMS_CODE_MAP`; for example, `zh_cn`
becomes MMSAlign's native `cmn`.

For Chinese, Korean and Japanese, the adapter automatically inserts spaces
around no-space script characters before alignment.

MMS alignment does not handle numeric-form tokens reliably. The adapter keeps a
separate alignment-token stream for MMS: numeric spans are temporarily aligned
as `<star>`, while the original surface text is preserved in the returned
timestamp. Numeric spans include standalone numbers such as `1952`, currency
and percent expressions such as `$ 100 %`, common currency codes such as
`USD 100`, units such as `20 kg`, and ranges such as `100 - 200`. These are
grouped into one timestamp token before alignment so MMS only sees one
placeholder for the unsupported expression.

This placeholder is distinct from the optional `use_star` noise tokens inserted
by MMS; inserted stars are filtered, numeric placeholder stars are restored to
their original text.

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
