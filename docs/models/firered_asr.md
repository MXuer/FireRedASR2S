# FireRedASR2

FireRedASR2 is integrated as two composable components:

- ASR: `firered_asr`
- Timestamp: `firered_asr_native`

The native timestamp component validates and passes through token timestamps
returned by FireRedASR2. Use it with `firered_asr` and keep
`return_timestamp=true`.

Example config fragment:

```json
{
  "components": {
    "asr": {
      "name": "firered_asr",
      "params": {
        "asr_type": "aed",
        "model_dir": "pretrained_models/FireRedASR2-AED",
        "return_timestamp": true,
        "beam_size": 3
      }
    },
    "timestamp": {
      "name": "firered_asr_native",
      "params": {}
    }
  }
}
```

Notes:

- Current language metadata marks FireRedASR2 as Chinese (`zh`) because the
  checked-in runtime and default model path are FireRedASR2-AED Chinese.
- `firered_asr_native` fails fast if ASR output has no `timestamp` field.
- No extra package installation is needed beyond the existing FireRed runtime
  dependencies already used by this repository.
