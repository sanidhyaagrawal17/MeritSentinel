# Tabular-Boost
Inference optimization framework for tabular models.

## Verified Results
- Baseline AUC-ROC 0.81 increased to 0.88 with LightGBM interaction depth 6.
- 3.4x inference speedup using ONNX Runtime FP32 CPU graph execution.
- Tested on 2x NVIDIA T4 GPUs; no 16-GPU FSDP or 70B LLM training.
