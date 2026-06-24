# GPU Resource Strategy Notes

## Status

This is a design note, not an implemented scheduler. It records the current
resource-management direction for later implementation.

For dependency-isolated model serving and lazy load/unload policy, see
[`model_lifecycle_sidecar.md`](model_lifecycle_sidecar.md).

## Problem

Semantic ASR pipelines contain four model stages:

1. VAD
2. ASR
3. Timestamp / forced alignment
4. Punctuation / semantic boundary

Any stage may use GPU, and some implementations can also run on CPU. The
current pipeline builds all components together, so GPU-heavy models can remain
resident after their stage has finished. For example, Whisper ASR may stay on
GPU while MMS forced alignment starts, causing additive memory pressure.

## Recommended Direction

Move from "one pipeline owns all models" to a staged resource model:

```text
job
  -> vad stage
  -> asr stage
  -> timestamp / align stage
  -> punctuation stage
  -> boundary + output stage
```

Each model should declare a resource profile:

```json
{
  "name": "whisper_large",
  "role": "asr",
  "device_kind": "gpu",
  "estimated_gpu_mem_gb": 10,
  "batchable": true,
  "keep_warm": true
}
```

```json
{
  "name": "mms_forced_aligner",
  "role": "timestamp",
  "device_kind": "gpu",
  "estimated_gpu_mem_gb": 2,
  "batchable": false,
  "keep_warm": false
}
```

The scheduler can then decide which stages should be resident, which should be
released, and which should run on CPU.

## Short-Term Policy

Add a future `resource_policy` section to profile configs:

```json
{
  "resource_policy": {
    "mode": "stage",
    "release_asr_before_timestamp": false,
    "release_timestamp_before_punc": true,
    "prefer_cpu_vad": true
  }
}
```

Default behavior should remain conservative:

- ASR models stay warm when batch throughput matters.
- Timestamp models are loaded only when needed unless they become a service
  worker.
- Punctuation and translation should run in separate services when they are
  large enough to block ASR throughput.

## Medium-Term Worker Layout

Use role-specific workers rather than one worker holding every model:

```text
vad_queue -> asr_queue -> align_queue -> punc_queue -> output_queue
```

Example deployment:

```text
GPU 6/7: ASR workers
GPU 4/5: timestamp / MMS workers
GPU 2: punctuation / translation workers
CPU: VAD, output, lightweight punctuation
```

This avoids ASR and aligner models stacking in the same process unless that is
intentional.

## Model Lifecycle Rules

| Model type | Suggested lifecycle |
| --- | --- |
| Whisper / Qwen ASR / Seamless | Keep-warm GPU workers, use model-native batching. |
| GigaAM-v3 | Keep warm when Russian jobs are common; use batch inference. |
| MMS aligner | Keep `num_workers=1` in the current per-audio pipeline. Consider persistent align workers before increasing worker count. |
| FireRedVAD / TenVAD / Silero | Prefer CPU or low-priority GPU placement; avoid stealing ASR cards. |
| FireRedPunc / ct-punc / XLM-R | Small models can be on CPU or a low-priority GPU. |
| Qwen semantic boundary / translation | Run as a separate service, not inline with ASR workers. |

## Current Evidence

- MMS multi-worker testing on a 300s Russian file showed no speedup with the
  current temporary worker-pool design. Worker startup and model loading
  dominated useful work.
- MMS emission reuse reduced timestamp time, but the improvement was modest
  compared with the larger architecture issue.
- ASR throughput should usually come from one warm model with internal batch
  inference, not many duplicated ASR model processes.

## Open Design Question

The main unresolved question is when to release ASR memory before timestamping.

- For single long jobs with a heavy aligner, releasing ASR may improve memory
  headroom.
- For batch ASR service throughput, keeping ASR warm is usually faster.

This should become a profile-level policy rather than a hard-coded behavior.
