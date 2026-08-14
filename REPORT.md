# QVAC Foundation Engineer Assignment - Final Report

1. **Report** - deliverables per task (metrics, scripts, log references).
2. **Appendix** - methodology, environment, extended notes. Cross-referenced as `[A1.x]`, `[A2.x]`, etc.

Environment: [A0](#a0-environment).

---

# Report

## Task 1: Quantize and Run a Model

LLaMA 3.2 3B → GGUF F16 → Q4_0, CPU inference (`-t 4`, `-c 4096`, `-n 128`). Measurement details: [A1.2](#a12-measurements).

### Scripts

| Step | Tool | Output |
| --- | --- | --- |
| HF → GGUF F16 | `llama.cpp/convert_hf_to_gguf.py` | `Llama-3.2-3B/llama-3.2-3b-f16.gguf` (6.0 GB) |
| F16 → Q4_0 | `llama.cpp/build/bin/llama-quantize … Q4_0` | `Llama-3.2-3B/llama-3.2-3b-q4_0.gguf` (1.8 GB) |

Commands: [A1.1](#a11-conversion-and-quantization).

### Results (warm runs)

Shared settings: `-c 4096`, `-n 128`, `-t 4`, second run after cold discard per model/prompt. All warm logs: `File system inputs: 0`, `Major page faults: 0`.

#### Prompt 1: "What is bitcoin?" (5 prompt tokens)

Logs: `logs-final/task1-f16-prompt1.log`, `logs-final/task1-q4_0-prompt1.log`

| Metric | F16 | Q4_0 | Q4_0 vs F16 |
| --- | --- | --- | --- |
| First token latency | 169.82 ms | 148.82 ms | 1.14× faster |
| Avg generation speed | 6.22 tok/s | 15.38 tok/s | 2.47× faster |
| Peak RAM | 6.60 GB | 3.86 GB | −42% |

#### Prompt 2: "Write a Python function to reverse a list." (10 prompt tokens)

Logs: `logs-final/task1-f16-prompt2.log`, `logs-final/task1-q4_0-prompt2.log`

| Metric | F16 | Q4_0 | Q4_0 vs F16 |
| --- | --- | --- | --- |
| First token latency | 276.73 ms | 290.62 ms | ~same |
| Avg generation speed | 6.21 tok/s | 15.32 tok/s | 2.47× faster |
| Peak RAM | 6.60 GB | 3.86 GB | −42% |

#### llama-bench (supplementary)

Logs: `logs-final/task1-f16-bench.md`, `logs-final/task1-q4_0-bench.md` · details: [A1.2](#a12-measurements).

| Model | pp5 (tok/s) | pp17 (tok/s) | tg128 (tok/s) |
| --- | --- | --- | --- |
| F16 | 21.54 ± 4.89 | 40.89 ± 1.86 | 6.21 ± 0.26 |
| Q4_0 | 43.24 ± 2.16 | 64.41 ± 4.05 | 17.25 ± 1.28 |

`tg128` matches completion decode speed (F16 ~6.2, Q4_0 ~15.3 tok/s). Q4_0 is ~2.8× faster on tg128 in bench.

### Task 1 - conclusions

| Aspect | F16 | Q4_0 | Takeaway |
| --- | --- | --- | --- |
| Disk size | 6.0 GB | 1.8 GB | Q4_0 is **3.3× smaller** on disk |
| Decode speed | ~6.2 tok/s | ~15.3 tok/s | Q4_0 is **~2.5× faster**; decode streams all weights each step |
| Prefill / first token | 170–277 ms | 149–291 ms | Modest or no gain; prefill is batched matmul, less weight-bound than decode |
| Peak RAM | 6.60 GB | 3.86 GB | **−42%** RSS; savings less than 3.3× file ratio due to KV cache + repack - [A1.2](#a12-measurements) |
| Output quality | Coherent on P1; messy code on P2 | Coherent on P1; correct `[::-1]` but loops on P2 | No garbage tokens; single-prompt observation - [A1.2](#a12-measurements) |

---

## Task 2: Q4_HQQ Quantization Format

New 4-bit HQQ format (`Q4_HQQ`, 5.0 bpw). Subtasks: CPU weights (2a), KV cache (2b), GPU kernels (2c).

### 2a. CPU weight quantization

Llama 3.2 3B F16 → `Q4_HQQ`, CPU inference with the same protocol as Task 1 (`-c 4096`, `-n 128`, `-t 4`, warm run). Q4_0 numbers are from Task 1. Implementation and measurement notes: [A2.1](#a21-task-2a---cpu-weights).

#### Scripts

| Step | Tool | Output |
| --- | --- | --- |
| F16 → Q4_HQQ | `llama.cpp/build/bin/llama-quantize … Q4_HQQ` | `Llama-3.2-3B/llama-3.2-3b-q4hqq.gguf` (2.0 GB) |

Command: [A2.1](#a21-task-2a---cpu-weights).

#### Results (warm runs)

Shared settings: `-c 4096`, `-n 128`, `-t 4`, second run after cold discard. Both Q4_HQQ logs: `File system inputs: 0`, `Major page faults: 0`.

##### Prompt 1: "What is bitcoin?" (5 prompt tokens)

Logs: `logs-final/task1-q4_0-prompt1.log`, `logs-final/task2-q4hqq-prompt1.log`

| Metric | Q4_0 | Q4_HQQ | Q4_HQQ vs Q4_0 |
| --- | --- | --- | --- |
| First token latency | 148.82 ms | 501.81 ms | 3.37× slower |
| Avg generation speed | 15.38 tok/s | 7.58 tok/s | 2.03× slower |
| Peak RAM | 3.86 GB | 2.56 GB | −34% |

##### Prompt 2: "Write a Python function to reverse a list." (10 prompt tokens)

Logs: `logs-final/task1-q4_0-prompt2.log`, `logs-final/task2-q4hqq-prompt2.log`

| Metric | Q4_0 | Q4_HQQ | Q4_HQQ vs Q4_0 |
| --- | --- | --- | --- |
| First token latency | 290.62 ms | 1054.78 ms | 3.63× slower |
| Avg generation speed | 15.32 tok/s | 7.69 tok/s | 1.99× slower |
| Peak RAM | 3.86 GB | 2.56 GB | −34% |

##### llama-bench (supplementary)

Logs: `logs-final/task1-q4_0-bench.md`, `logs-final/task2-q4hqq-bench.md`

| Model | pp5 (tok/s) | pp17 (tok/s) | tg128 (tok/s) |
| --- | --- | --- | --- |
| Q4_0 | 43.24 ± 2.16 | 64.41 ± 4.05 | 17.25 ± 1.28 |
| Q4_HQQ | 9.67 ± 1.01 | 9.60 ± 0.45 | 7.69 ± 0.32 |

`tg128` matches completion decode (~7.6 tok/s). Prefill in bench stays near decode for Q4_HQQ (no Q4_0-style pp ≫ tg gap).

#### Task 2a - conclusions

| Aspect | Q4_0 | Q4_HQQ | Takeaway |
| --- | --- | --- | --- |
| Disk size | 1.8 GB (4.5 bpw) | 2.0 GB (5.0 bpw) | Q4_HQQ is **~11% larger**, matching 20 vs 18 bytes per 32 weights |
| Decode speed | ~15.3 tok/s | ~7.6 tok/s | Q4_HQQ is **~2× slower**; AVX2 `vec_dot`, no AVX512/VNNI and no SIMD repack - [A2.1](#a21-task-2a---cpu-weights) |
| Prefill / first token | 149–291 ms | 502–1055 ms | **~3.4–3.6× slower**; prefill is compute-bound, so the kernel gap shows more than on decode |
| Peak RAM | 3.86 GB | 2.56 GB | **−34%** RSS: Q4_HQQ has no repack copy (~1.5 GB on Q4_0) |
| Output quality | Coherent on P1; `[::-1]` then loops on P2 | Coherent on P1; working `[::-1]` with a mismatched comment on P2 | No garbage tokens; comparable base-model quality - [A2.1](#a21-task-2a---cpu-weights) |

### 2b. KV cache quantization

Register `Q4_HQQ` as `--cache-type-k/v`. Same Q4_0 weights as Task 1; only the in-memory K/V format changes. Prompt 1, `-s 42 --temp 0` so quality is comparable. Details: [A2.2](#a22-task-2b---kv-cache).

#### Results (warm runs)

Logs: `logs-final/task2-kv-f16-prompt1.log`, `logs-final/task2-kv-q4hqq-prompt1.log`  
Settings: `llama-3.2-3b-q4_0.gguf`, `-c 4096`, `-n 128`, `-t 4`. Both: `File system inputs: 0`.

| Metric | F16 KV (default) | Q4_HQQ KV | Change |
| --- | --- | --- | --- |
| First token latency | 135.67 ms | 140.69 ms | ~same |
| Avg generation speed | 13.57 tok/s | 11.81 tok/s | −13% |
| Peak RAM | 3.86 GB | 3.56 GB | **−308 MB (−8%)** |

#### Task 2b - conclusions

| Aspect | F16 KV | Q4_HQQ KV | Takeaway |
| --- | --- | --- | --- |
| Model file | Q4_0, 1.8 GB | same | Weights unchanged |
| KV at `-c 4096` | ~0.46 GB | ~0.14 GB (5/16 of f16) | Measured RSS drop **308 MB** matches the KV formula - [A2.2](#a22-task-2b---kv-cache) |
| Decode | 13.57 tok/s | 11.81 tok/s | Modest overhead from quantized V + flash attention |
| Prefill | 136 ms | 141 ms | Unaffected at this prompt length |
| Output quality | Coherent Bitcoin article | Coherent, same topic | Same seed/greedy decode **diverges from the first tokens**; no garbage - [A2.2](#a22-task-2b---kv-cache) |

### 2c. GPU kernels (Vulkan)

Q4_HQQ matmul/dequant on Vulkan. Native Windows, GTX 1650 (`Vulkan1`). Metal/OpenCL not implemented (assignment allows one GPU backend). Details and KV caveat: [A2.3](#a23-task-2c---vulkan).

Proof logs (`-lv 4`): `file type = Q4_HQQ`, `using device Vulkan1 (NVIDIA GeForce GTX 1650)`, `offloaded 29/29 layers to GPU`.

#### Results

Logs: `logs-final/task2-q4hqq-vulkan-prompt1.log`, `logs-final/task2-q4hqq-vulkan-kv-prompt1.log`  
Settings: `llama-3.2-3b-q4hqq.gguf`, `-ngl 99 --device Vulkan1 --fit off`, `-c 4096`, `-n 64`, `-s 42 --temp 0`. CPU row is Task 2a prompt 1 (same model).

##### Weights on GPU vs CPU (f16 KV)

| Metric | CPU Q4_HQQ (2a) | Vulkan Q4_HQQ | Vulkan vs CPU |
| --- | --- | --- | --- |
| First token latency | 501.81 ms | 81.69 ms | **6.1× faster** |
| Avg generation speed | 7.58 tok/s | 28.69 tok/s | **3.8× faster** |
| Device | CPU, 4 threads | Vulkan1, 29/29 layers | — |

##### Vulkan KV: f16 vs `q4_hqq`

| Metric | F16 KV | Q4_HQQ KV | Change |
| --- | --- | --- | --- |
| First token latency | 81.69 ms | 75.87 ms | ~same |
| Avg generation speed | 28.69 tok/s | 28.68 tok/s | ~same |
| KV buffer | 448 MiB (f16) | **140 MiB** (q4_hqq) | **−308 MiB (−69%)** |
| GPU self (model+KV+compute) | 2715 MiB | 2407 MiB | −308 MiB |

#### Task 2c - conclusions

| Aspect | Takeaway |
| --- | --- |
| Weights | Canonical HQQ dequant on GPU (`w = (q − zero) / scale`). Full offload on GTX 1650 |
| Decode / prefill | **~3.8× / ~6×** vs CPU Q4_HQQ on this card |
| KV flag | `--cache-type-k/v q4_hqq` runs and shrinks KV 448 → 140 MiB; greedy text matches f16 KV on this prompt |
| KV encoding | GPU KV is **Q4_1 laid into Q4_HQQ-typed blocks**, not CPU-canonical HQQ headers. FA treats the type as `Q4_1` - [A2.3](#a23-task-2c---vulkan) |
| Quality | Coherent Bitcoin paragraph; no `???` / crash |

---

## Task 3: `--mmproj-backend`

CLI flag that picks a ggml device **only for the multimodal projector**. `--device` / `-ngl` for the LLM stay unchanged. Details: [A3](#a3-task-3---details).

```text
--mmproj-backend {DEVICE}     # DEVICE from --list-devices, e.g. Vulkan0
```

CPU projector uses `--no-mmproj-offload` (`parse_device_list` rejects `CPU`).

#### Results

Windows, GTX 1650 (`Vulkan1`) + Intel UHD (`Vulkan0`). Model: SmolVLM-256M Instruct Q8_0. Image: `test-1-positive.png`. `-lv 4`.

Logs: `logs-final/task3-llm-vulkan1-clip-vulkan0.log`, `…-vulkan1.log`, `…-cpu.log`

In every run the LLM stays on NVIDIA:

```text
llama_prepare_model_devices: using device Vulkan1 (NVIDIA GeForce GTX 1650)
load_tensors: offloaded 31/31 layers to GPU
load_tensors:      Vulkan1 model buffer size =   136.47 MiB
```

| CLIP flag | CLIP log line | Encode / chunk | CLIP compute buffer |
| --- | --- | --- | --- |
| `--mmproj-backend Vulkan0` | `CLIP using Vulkan0 backend` | 333–371 ms | Vulkan0 21 MiB |
| `--mmproj-backend Vulkan1` | `CLIP using Vulkan1 backend` | 158–176 ms | Vulkan1 21 MiB |
| `--no-mmproj-offload` | `CLIP using CPU backend` | 1467–1541 ms | CPU 21 MiB |

All three answers: *"An article about a powder-coated surface."*

#### Task 3 - conclusions

| Aspect | Takeaway |
| --- | --- |
| Isolation | LLM device is `Vulkan1` in all three logs; only the CLIP line and encode time change |
| Cross-device | LLM on NVIDIA while CLIP runs on Intel (`Vulkan0`) or CPU |
| Encode time | Tracks CLIP device: Vulkan1 fastest, CPU slowest |
| Plumbing | `common_params.mmproj_device` → `mtmd` → `clip` → `ggml_backend_dev_init`; `params.devices` untouched |

---

# Appendix

## A0. Environment

| Item | CPU (Tasks 1, 2a, 2b) | GPU (Tasks 2c, 3) |
| --- | --- | --- |
| OS | Linux 6.18, x86_64 | Native Windows |
| RAM / VRAM | 15 GiB | GTX 1650, 4152 MiB (`Vulkan1`) |
| Threads | 4 (`-t 4`, 8 logical cores) | same host CPU; compute on GPU |
| ISA | AVX2, AVX512, AVX512_VNNI (`REPACK = 1`) | — |
| llama.cpp commit | `112e9ca` | same tree, MSVC Release |
| Build | `cmake -B build -DGGML_NATIVE=ON` | `cmake -B build -G "Visual Studio 17 2022" -A x64 -DGGML_VULKAN=ON` |

---

## A1. Task 1 - details

### A1.1. Conversion and quantization

```bash
conda activate llama-cpp
pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

huggingface-cli download meta-llama/Llama-3.2-3B --local-dir Llama-3.2-3B

python llama.cpp/convert_hf_to_gguf.py Llama-3.2-3B \
  --outfile Llama-3.2-3B/llama-3.2-3b-f16.gguf --outtype f16

./llama.cpp/build/bin/llama-quantize \
  Llama-3.2-3B/llama-3.2-3b-f16.gguf \
  Llama-3.2-3B/llama-3.2-3b-q4_0.gguf Q4_0
```

### A1.2. Measurements

All completion runs used `llama-completion` with `--perf` and `/usr/bin/time -v`. Runs were ordered model-by-model: all F16 measurements first, then all Q4_0. For each (model, prompt) pair we discarded one cold run and logged the second as the reported result. Fixed parameters were `-c 4096`, `-n 128`, `-t 4`, and `--simple-io`.

#### Context length

Without `-c`, context length comes from model metadata. For Llama 3.2 that is 131072 tokens. The KV cache is allocated for the full context up front, regardless of prompt length:

```text
2 (K and V) × 28 layers × 8 KV heads × 128 head_dim × 131072 tokens × 2 bytes (f16) = 14.9 GB
```

In an earlier experiment on this 15 GiB machine, omitting `-c` pushed peak RSS to 14946 MB and dropped generation speed to 13.3 tok/s. With `-c 4096`, the KV portion shrinks to about 0.46 GB:

```text
2 × 28 × 8 × 128 × 4096 × 2 bytes ≈ 0.46 GB
```

#### Repack and memory

`system_info` reports `REPACK = 1`. For Q4_0, llama.cpp repacks weights into a SIMD-friendly layout and keeps a separate in-RAM copy (about 1.5 GB). We did not pass `--no-repack`: the goal is to reflect real default inference, not a minimal footprint mode.

| Component | F16 | Q4_0 |
| --- | --- | --- |
| Model weights (mmap) | 6.0 GB | 1.8 GB |
| KV cache (4096 tokens) | ~0.46 GB | ~0.46 GB |
| Repack weights | - | ~1.5 GB |
| Total measured (peak RSS) | 6.60 GB | 3.86 GB |

For F16 the arithmetic matches directly: 6.0 + 0.46 = 6.46 GB versus 6.60 GB measured. For Q4_0, the gap between 1.8 GB on disk and 3.86 GB in RAM is explained by repack plus KV cache (1.8 + 0.46 + 1.5 ≈ 3.76 GB). In the earlier draft, `--no-repack` on Q4_0 dropped peak RSS from 3.75 GB to 2.29 GB, which corresponds to weights plus KV cache alone.

#### First token latency and generation speed

We take first token latency from the `prompt eval time` line in `--perf` output. llama.cpp does not expose a separate TTFT metric, but `prompt eval time` is practically equivalent: inference splits into prefill (whole prompt in one batch) and decode (one token per step). The first token is sampled from the logits of the last prefill position, without an extra decode step. This shows up in the logs - with `-n 128`, `eval time` lists 127 runs, not 128.

Average generation speed comes from the `eval time` line (tok/s over decode steps). Formally, sampling of the first token could be folded into TTFT; in our runs `sampling time` was ~37–41 ms across all tokens, which is negligible compared to prefill and decode.

#### Warmup

The first run on a cold page cache reads weights from disk lazily during prefill, so `prompt eval time` can reflect disk bandwidth instead of compute. In an earlier cold Q4_0 run this produced 8194 ms prefill per token versus ~75 ms per decode token - an impossible ordering if the measurement were compute-bound. After warmup, prefill dropped to ~120–290 ms depending on prompt length.

The warm-run indicator we require in the final logs is `File system inputs: 0` and `Major page faults: 0` at the end of the `/usr/bin/time -v` block. All four `logs-final/task1-*-prompt*.log` files satisfy this.

#### llama-bench

As a cross-check we ran `llama-bench` per model:

```bash
./build/bin/llama-bench -m <model.gguf> -p 5,17 -n 128 -t 4 -r 5 -o md
```

Bench separates prompt processing (`pp5`, `pp17`) from text generation (`tg128`) and excludes tokenization and sampling. The `tg128` numbers align with completion decode speed (F16: 6.21 vs 6.22 tok/s; Q4_0: 17.25 vs ~15.3 tok/s). Prefill bench values are higher than completion prefill because the benchmark uses isolated pp tests without the full CLI path.

The bench logs print `ggml_vulkan: No devices found` yet label the backend as Vulkan. These CPU-only runs match completion throughput, so we treat the results as CPU-equivalent. Note that `pp17` is only an approximation for prompt 2 - the actual prompt tokenizes to 10 tokens, not 17.

#### Output quality

Both models behave like a base model rather than an instruction-tuned assistant. On prompt 1, F16 and Q4_0 both produce coherent article-style text about Bitcoin with no special-token artifacts.

On prompt 2, Q4_0 returns a working `return lst[::-1]` but then repeats the same paragraph about Python lists. F16 also produces runnable logic but with broken formatting (`def  reverse_list ( l )`) and test-case phrasing. Repetition is not suppressed: `repeat_penalty = 1.000` in all runs. This is a single-prompt observation; a systematic quality comparison would need perplexity or a broader eval set.

---

## A2. Task 2 - details

### A2.1. Task 2a - CPU weights

`Q4_HQQ` follows the assignment block layout: 32 weights per block, FP16 `scale` and `zero`, packed 4-bit quants (`2 + 2 + 16 = 20` bytes, 5.0 bpw). Quantization is `q = clamp(round(w * scale + zero), 0, 15)` with `scale = 15 / (max - min)` and `zero = -min * scale`. Dequantization is `w = (q - zero) / scale`.

The type is registered through ggml (`GGML_TYPE_Q4_HQQ = 43`), `llama-quantize`, and the model loader. CPU inference uses `quantize_row_q4_hqq` / `dequantize_row_q4_hqq` plus `ggml_vec_dot_q4_hqq_q8_0`. The x86 path is AVX2 (`ggml/src/ggml-cpu/arch/x86/quants.c`); ARM falls back to the scalar generic kernel. There is no Q4_HQQ entry in the CPU repack table.

```bash
./llama.cpp/build/bin/llama-quantize \
  Llama-3.2-3B/llama-3.2-3b-f16.gguf \
  Llama-3.2-3B/llama-3.2-3b-q4hqq.gguf Q4_HQQ
```

Completion runs used the Task 1 protocol: `llama-completion --perf`, `/usr/bin/time -v`, `-c 4096`, `-n 128`, `-t 4`, `--simple-io`. One cold run was discarded per prompt. Both `logs-final/task2-q4hqq-prompt*.log` files show `File system inputs: 0` and `Major page faults: 0`.

#### File size and memory

The 11% file-size increase versus Q4_0 is the block overhead: Q4_0 stores 18 bytes per 32 weights (4.5 bpw), Q4_HQQ stores 20 (5.0 bpw). Bench reports 1.78 GiB vs 1.94 GiB, which matches `ls` (1.8 GB vs 2.0 GB).

Peak RSS moves the other way: 3.86 GB (Q4_0) → 2.56 GB (Q4_HQQ). Q4_0 keeps a SIMD-friendly in-RAM repack of the weights (~1.5 GB; [A1.2](#a12-measurements)). Q4_HQQ is not in that repack path, so RSS is close to mmap weights plus the shared f16 KV cache:

```text
2.0 GB (weights) + 0.46 GB (KV at 4096) ≈ 2.46 GB  vs  2.56 GB measured
```

`system_info` still prints `REPACK = 1` because the CPU build supports repack; it simply has nothing to do for this type.

#### Speed

Decode is about 2× slower than Q4_0 (~7.6 vs ~15.3 tok/s). Prefill is 3.4–3.6× slower. The extra 0.5 bpw cannot explain that gap under a memory-bound decode model. The kernel does: Q4_0 uses AVX512/VNNI plus a repacked layout; Q4_HQQ uses an AVX2 `vec_dot` with an extra zero-point correction (`sum w y = (d/scale) * (sum q y − zero * sum y)`). Prefill is batched matmul and therefore more sensitive to that arithmetic path, which is why `pp5`/`pp17` in `llama-bench` sit next to `tg128` for Q4_HQQ (~9.6 vs 7.7 tok/s) instead of pulling ahead the way Q4_0 does (43–64 vs 17 tok/s).

`tg128` in bench (7.69 ± 0.32 tok/s) matches completion decode (7.58 / 7.69 tok/s). As in Task 1, the bench log prints `ggml_vulkan: No devices found`; throughput is CPU-equivalent.

An earlier scalar-only `vec_dot` ran at ~2.3 tok/s. AVX2 closed most of that hole; remaining headroom is AVX512/VNNI and a Q4_HQQ repack kernel.

#### Output quality

Both formats behave like a base model. On prompt 1, Q4_HQQ writes coherent article-style Bitcoin text (origin, lack of a central issuer, miners) with no special-token artifacts. On prompt 2 it emits a working `return l[::-1]`, then a comment that names `reverse()` instead of slice notation, then an unfinished second definition. Q4_0 also returns `[::-1]` and then loops on the same list paragraph. `repeat_penalty = 1.000` in both runs. This is a single-prompt observation, not a perplexity comparison.

### A2.2. Task 2b - KV cache

This is separate from weight quantization: both runs load `llama-3.2-3b-q4_0.gguf`. The only change is `--cache-type-k q4_hqq --cache-type-v q4_hqq`. `GGML_TYPE_Q4_HQQ` was added to the KV whitelist in `common/arg.cpp` (and the name map in `llama-bench`). Quantize/dequantize and CPU flash-attention for the type already existed from Task 2a. Quantized V cache requires flash attention; llama.cpp turns it on automatically.

```bash
# F16 KV (default)
/usr/bin/time -v ./build/bin/llama-completion \
  -m ../Llama-3.2-3B/llama-3.2-3b-q4_0.gguf \
  -p "What is bitcoin?" -n 128 -c 4096 -t 4 \
  -s 42 --temp 0 --perf --simple-io

# Q4_HQQ KV
/usr/bin/time -v ./build/bin/llama-completion \
  -m ../Llama-3.2-3B/llama-3.2-3b-q4_0.gguf \
  -p "What is bitcoin?" -n 128 -c 4096 -t 4 \
  --cache-type-k q4_hqq --cache-type-v q4_hqq \
  -s 42 --temp 0 --perf --simple-io
```

One cold run was discarded per config. Reported logs are warm (`File system inputs: 0`). Seed and greedy sampling isolate numerical drift in attention from sampler noise.

#### Memory

F16 KV at `-c 4096` is the same 0.46 GB derived in [A1.2](#a12-measurements):

```text
2 × 28 × 8 × 128 × 4096 × 2 bytes ≈ 0.46 GB
```

Q4_HQQ stores 5 bits per value instead of 16, so the KV arena shrinks by 5/16:

```text
0.46 GB × 5/16 ≈ 0.14 GB    →    savings ≈ 0.32 GB
```

Measured peak RSS is 4045348 kB (f16) vs 3729980 kB (q4_hqq), a difference of **308 MB**. Weights and the Q4_0 repack copy are unchanged; only the KV footprint drops. Absolute savings scale with context length.

#### Speed and quality

Prefill is unchanged (136 vs 141 ms). Decode falls from 13.57 to 11.81 tok/s (−13%), consistent with dequant inside flash attention on every step. These decode numbers are not compared to Task 1: Task 1 used `--temp 0.6` without a fixed seed.

Both outputs stay on-topic, article-style Bitcoin text. They are not bit-identical. F16 KV continues *“How can you buy it? These are the questions that have been asked by millions of people around the world.”* Q4_HQQ KV inserts *“What are the benefits?”* and then *“These are all questions that many people have about bitcoin.”* Quantized K/V changes attention logits; the drift compounds over tokens. No special-token artifacts or collapsed output on this prompt.

### A2.3. Task 2c - Vulkan

Task 2c was built and run on native Windows against `Vulkan1` (GTX 1650). Metal and OpenCL were skipped; the assignment asks for Vulkan, Metal, *or* OpenCL.

Weight kernels dequantize with the canonical HQQ map (`d = 1/scale`, `m = −zero/scale`, i.e. `w = (q − zero) / scale`): `dequant_q4_hqq.comp`, `dequant_funcs.glsl`, plus mul_mat / mul_mat_vec pipelines in `ggml-vulkan.cpp`.

```powershell
$exe   = "C:\Users\user\work\llama.cpp\build\bin\Release\llama-completion.exe"
$model = "C:\Users\user\work\Llama-3.2-3B\llama-3.2-3b-q4hqq.gguf"

# Weights, f16 KV
& $exe -m $model -p "What is bitcoin?" -n 64 -c 4096 `
  -s 42 --temp 0 --device Vulkan1 -ngl 99 --fit off `
  --perf -lv 4 --log-file C:\Users\user\work\logs\task2-q4hqq-vulkan-prompt1.log

# Weights + KV typed q4_hqq
& $exe -m $model -p "What is bitcoin?" -n 64 -c 4096 `
  -s 42 --temp 0 --device Vulkan1 -ngl 99 --fit off `
  --cache-type-k q4_hqq --cache-type-v q4_hqq `
  --perf -lv 4 --log-file C:\Users\user\work\logs\task2-q4hqq-vulkan-kv-prompt1.log
```

`-lv 4` is required for the offload proof: ggml/llama INFO is mapped to verbosity 4, so default `-lv 3` drops `using device` / `offloaded` / `file type`. `--log-file` avoids PowerShell `Tee-Object` wrapping stderr as `NativeCommandError`.

Vulkan runs used `-n 64` (63 decode steps) versus `-n 128` on CPU. Token rates remain comparable. CPU 2a used `--temp 0.6` without a fixed seed; Vulkan used greedy `-s 42 --temp 0`.

Both logs show `file type = Q4_HQQ`, `196` `q4_hqq` tensors, `Vulkan1 model buffer size = 1988.90 MiB`, and `offloaded 29/29 layers to GPU`.

#### KV on Vulkan

`--cache-type-k/v q4_hqq` is accepted and the allocator reports `K (q4_hqq): 70 MiB, V (q4_hqq): 70 MiB` versus `K/V (f16): 224+224 MiB`. That is the same 5/16 ratio as CPU Task 2b (448 → 140 MiB). Decode does not slow down on this prompt (28.69 vs 28.68 tok/s). Greedy output is the same Bitcoin paragraph in both Vulkan logs.

The GPU KV path is not CPU-canonical HQQ. `copy_to_quant.comp` writes Q4_1 parameters into the HQQ fields (`scale = d = (max−min)/15`, `zero = min`). Flash attention then specializes `GGML_TYPE_Q4_HQQ` as `FA_TYPE_Q4_1`. Same 20-byte block layout, so the type name and buffer size match; a CPU HQQ dequant of those GPU KV blocks would not. Weight tensors still use real HQQ. An earlier FA path that dequantized KV with `(q − zero) / scale` produced `???` tokens until this remap.

A full HQQ KV shader (write and FA read with `scale`/`zero` as on CPU) is the remaining gap.

---

## A3. Task 3 - details

The assignment asks for `--mmproj-backend {value}` so the projector can sit on a different ggml backend from the language model. CLIP already had a hidden `MTMD_BACKEND_DEVICE` env override. The flag turns that into a parsed `ggml_backend_dev_t` and threads it through the stack without touching `params.devices`:

```text
--mmproj-backend Vulkan0
        → parse_device_list()            (common/arg.cpp)
        → common_params.mmproj_device
        → mtmd_context_params.device     (mtmd-cli.cpp, server, tts, mtmd-debug)
        → clip_context_params.device
        → ggml_backend_dev_init()       (clip.cpp)
```

`parse_device_list` does not accept `CPU`, so the CPU projector stays on `--no-mmproj-offload` (`mmproj_use_gpu = false`), which skips `ggml_backend_dev_init` and logs `CLIP using CPU backend`.

Same Windows Vulkan tree as Task 2c (`llama-mtmd-cli.exe`). Model: `ggml-org/SmolVLM-256M-Instruct-GGUF` (Q8_0 + matching mmproj). Image: `tools/mtmd/tests/test-1-positive.png`.

```powershell
$exe    = "C:\Users\user\work\llama.cpp\build\bin\Release\llama-mtmd-cli.exe"
$model  = "C:\Users\user\work\SmolVLM-256M\SmolVLM-256M-Instruct-Q8_0.gguf"
$mmproj = "C:\Users\user\work\SmolVLM-256M\mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"
$img    = "C:\Users\user\work\SmolVLM-256M\test-1-positive.png"

& $exe -m $model --mmproj $mmproj --image $img `
  -p "What is in this image?" -n 32 --temp 0 -lv 4 `
  --device Vulkan1 -ngl 99 --mmproj-backend Vulkan0 `
  --log-file C:\Users\user\work\logs\task3-llm-vulkan1-clip-vulkan0.log
```

The other two runs differ only in `--mmproj-backend Vulkan1` and `--no-mmproj-offload` (no `--mmproj-backend`). As in Task 2c, `-lv 4` is required so `CLIP using … backend` is written to `--log-file`.

LLM buffer size is 136.47 MiB on `Vulkan1` in all three files. CLIP compute is 21 MiB on whichever backend was selected. `mtmd batch encoding done in …` is the projector cost (three chunks per run): 158–176 ms on Vulkan1, 333–371 ms on Vulkan0, 1467–1541 ms on CPU. That ordering matches the CLIP log line, not the LLM device.

The generated sentence is identical across backends. SmolVLM-256M paraphrases the NYT clipping in the test image; it is not a general VQA score.
