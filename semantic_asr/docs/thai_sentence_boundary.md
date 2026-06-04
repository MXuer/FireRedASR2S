# Thai Sentence-Boundary Strategy

## Problem

Thai is not supported by the current XLM-R punctuation model. Thai writing can
use punctuation, but sentence-final punctuation is often absent and is not a
reliable semantic sentence-boundary signal.

The pipeline should therefore not invent punctuation merely to split Thai ASR
text into sentences.

## Recommended Strategy

Keep the fourth mandatory pipeline stage, but treat it as a sentence-boundary
strategy rather than requiring it to be a punctuation-restoration model.

For Thai, use aligned word/token timestamps and:

1. Split at strong inter-token pauses, initially `0.8s` by default.
2. When a segment exceeds a target duration, initially `15s`, split at the
   strongest nearby pause.
3. Force a split before the maximum duration, initially `30s`, at the nearest
   available token boundary.
4. Preserve the ASR text exactly; do not add punctuation that was not emitted
   by the ASR model.
5. Record the boundary reason, such as `strong_pause`, `target_duration`, or
   `max_duration`, for later evaluation.

The long VAD segments remain useful because they preserve recognition context.
The Thai sentence-boundary strategy runs after ASR and timestamp alignment, so
it can produce shorter semantic output sentences without shortening the audio
segments sent to ASR.

## Future Improvement

Once reference transcripts and sentence boundaries are available, evaluate a
Thai-specific text sentence-boundary model as an additional soft signal. It
should be combined with timestamp pauses rather than replacing acoustic
evidence.
