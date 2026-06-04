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
