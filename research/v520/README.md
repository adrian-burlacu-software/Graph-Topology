# V520 — Gemma 4 device-safe LLM interface

V520 fixes the Gemma 4 inference boundary at the framework level.

- Requires Transformers >= 5.5.3 (Gemma 4 device-map auto fix).
- Requires Accelerate >= 1.7.0.
- First attempts single-GPU CUDA placement when available.
- Falls back to `device_map="auto"` with an explicit offload directory when the full model does not fit.
- Verifies there are no live `meta` parameters/buffers after dispatch.
- Routes inputs through the actual input-embedding device.
- Keeps the V517 cognitive/ingestion architecture intact.
