# Short Summary

Llama 3.2 3B in llama.cpp: Q4_0 baseline, new `Q4_HQQ` format (CPU + Vulkan), quantized KV cache, and `--mmproj-backend`. Full metrics and logs: `REPORT.md`, `logs-final/`.

## Modifications

**Task 1.** Converted the HF checkpoint to GGUF F16, quantized to Q4_0, measured CPU inference (`-c 4096`, `-t 4`). Q4_0 is 3.3× smaller on disk and ~2.5× faster on decode than F16; peak RAM drops 42%, less than the file-size ratio because of KV cache and SIMD repack.

**Task 2a.** Implemented HQQ 4-bit blocks (20 bytes / 32 weights, 5.0 bpw): ggml type `Q4_HQQ`, quantize/dequant, `llama-quantize`, and AVX2 `ggml_vec_dot_q4_hqq_q8_0`. CPU decode is ~2× slower than Q4_0 (~7.6 vs ~15.3 tok/s); peak RSS is 34% lower because Q4_HQQ is not in the repack path. Output stays coherent on the test prompts.

**Task 2b.** Registered `Q4_HQQ` for `--cache-type-k/v`. Same Q4_0 weights: KV at `-c 4096` shrinks by 308 MB (−8% RSS). Decode −13%. Greedy text with a fixed seed diverges from f16 KV but stays on topic.

**Task 2c.** Vulkan weight kernels use canonical HQQ dequant (`w = (q − zero) / scale`). On a GTX 1650, 29/29 layers offload; decode is 3.8× the CPU Q4_HQQ figure. `--cache-type-k/v q4_hqq` runs and cuts the KV buffer 448 → 140 MiB. GPU KV is Q4_1 parameters stored in Q4_HQQ-typed blocks; flash attention specializes the type as `Q4_1`. Metal/OpenCL were not added.

**Task 3.** `--mmproj-backend DEVICE` selects the CLIP/mmproj ggml device only. `params.devices` / `--device` are unchanged. On SmolVLM-256M, the LLM stays on `Vulkan1` while CLIP is `Vulkan0`, `Vulkan1`, or CPU (`--no-mmproj-offload`). Encode time follows the CLIP device; the caption is the same in all three logs.

## Issues

- The first CPU `vec_dot` was scalar (~2.3 tok/s). AVX2 recovered most of the gap; Q4_0 still leads via AVX512/VNNI and repack.
- Canonical HQQ dequant on the Vulkan flash-attention KV path produced garbage tokens (`???`). Writing Q4_1 `(d, m)` into the HQQ fields and mapping FA to `Q4_1` restored coherent output. CPU KV remains real HQQ.
- `--mmproj-backend CPU` is rejected by `parse_device_list`; CPU projector is `--no-mmproj-offload`.
- Default log verbosity (`-lv 3`) drops llama/CLIP INFO. Proof of GPU offload and CLIP device needs `-lv 4` and `--log-file` (PowerShell `Tee-Object` wraps stderr as `NativeCommandError`).

## Further work

- AVX512/VNNI `vec_dot` and a Q4_HQQ CPU repack kernel.
- Vulkan KV write/read with CPU-canonical `scale` / `zero`, not the Q4_1 alias.
- Allow `CPU` in `--mmproj-backend` without `--no-mmproj-offload`.
- Perplexity (or a broader eval) for Q4_HQQ vs Q4_0; the quality notes here are single-prompt.
