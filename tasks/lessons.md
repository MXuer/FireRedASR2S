# Lessons

- When adding any new model adapter, also add documentation for required downloads, installation and environment configuration.
- Every new model adapter must include a standalone test or example that verifies the model can load and its normalized output shape is correct.
- For non-trivial work, add TODO items to `tasks/todo.md` before implementation and update each item as it is completed.
- When recording model language support, use the explicit supported-language list from the user or primary model documentation; do not infer from incomplete page tags.
- For timestamp providers that receive `SpeechSegment.wav`, do not write temporary segment wav files unless unavoidable; prefer in-memory audio paths and extract needed external code into `semantic_asr` instead of depending on loose external packages.
- Install Dolphin from the official `DataoceanAI/Dolphin` GitHub repository, not from PyPI; do not accept Dolphin smoke-test results from a PyPI build as final compatibility evidence.
- Network access is approved for project work; use it directly when model downloads, dependency installation, or source verification are needed.
- Real GPU commands must run outside the filesystem sandbox so CUDA devices are visible; use GPUs 4, 5, 6, and 7 through `CUDA_VISIBLE_DEVICES`.
- Do not infer that a punctuation model supports Thai from its multilingual name; keep Thai out of XLM-R punctuation tests unless the model's explicit support list includes it.
- Smoke-test runners filtered to one language must not retain hard-coded assumptions that other language fixtures are present.
- Qwen3-ASR language configuration must be converted to the model API's full English language names; do not pass short language codes directly.
- Fun-ASR-Nano language support in this project is limited to Chinese, English and Japanese; do not infer broader support from generic multilingual examples.
- Do not expose model-native language values in pipeline profiles. Use one canonical top-level language-region id and keep all model-specific conversion in the centralized mapping layer.
- Normalize final sentence intervals after output VAD alignment and before any output writer; all JSON, CSV, SRT and TextGrid outputs must receive the same monotonic, non-overlapping sentence list.
- When snapping a sentence boundary to VAD silence, restrict the search to a nearby candidate-boundary gap; do not use a distant silence inside a large aligned-word gap because it can incorrectly expand sentence intervals.
- Treat target duration as a pressure signal, but keep maximum duration bounded. When no audio-safe boundary exists after `max_sentence_s`, use a semantic-complete active-speech boundary as the least-bad cap; only semantic-incomplete active boundaries should keep waiting for silence.
- Do not merge across long raw-VAD pauses. If the silence between adjacent speech islands reaches `max_merge_vad_silence_s` (default 1.0s), keep the boundary even when punctuation suggests a continuation.
- Do not cut at the first over-max terminal punctuation inside continuous speech. When max duration is exceeded, choose the lowest-speech-probability recent semantic boundary in the current group.
