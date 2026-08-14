# QVAC Foundation Engineer Assignment

Llama 3.2 3B in llama.cpp: Q4_0 baseline, new `Q4_HQQ` format (CPU + Vulkan + Metal), quantized KV cache, and `--mmproj-backend`.

Metrics and analysis: [`REPORT.md`](REPORT.md). Logs: [`logs-final/`](logs-final/).

Commands below are from the repo root unless a `cd` is shown.

---

# Summary

## Modifications

**Task 1.** Converted the HF checkpoint to GGUF F16, quantized to Q4_0, measured CPU inference (`-c 4096`, `-t 4`). Q4_0 is 3.3× smaller on disk and ~2.5× faster on decode than F16; peak RAM drops 42%, less than the file-size ratio because of KV cache and SIMD repack.

**Task 2a.** Implemented HQQ 4-bit blocks (20 bytes / 32 weights, 5.0 bpw): ggml type `Q4_HQQ`, quantize/dequant, `llama-quantize`, and AVX2 `ggml_vec_dot_q4_hqq_q8_0`. CPU decode is ~2× slower than Q4_0 (~7.6 vs ~15.3 tok/s); peak RSS is 34% lower because Q4_HQQ is not in the repack path. Output stays coherent on the test prompts.

**Task 2b.** Registered `Q4_HQQ` for `--cache-type-k/v`. Same Q4_0 weights: KV at `-c 4096` shrinks by 308 MB (−8% RSS). Decode −13%. Greedy text with a fixed seed diverges from f16 KV but stays on topic.

**Task 2c.** Weight kernels use canonical HQQ dequant (`w = (q − zero) / scale`) on Vulkan and Metal. GTX 1650: 29/29 layers, decode 3.8× CPU Q4_HQQ. Apple M4: 29/29 layers, decode 5.8× CPU (different machine). `--cache-type-k/v q4_hqq` cuts KV 448 → 140 MiB on both. Vulkan KV stores Q4_1 `(d, m)` in HQQ-typed blocks; Metal KV uses CPU-canonical `scale` / `zero`.

**Task 3.** `--mmproj-backend DEVICE` selects the CLIP/mmproj ggml device only. `params.devices` / `--device` are unchanged. On SmolVLM-256M, the LLM stays on `Vulkan1` while CLIP is `Vulkan0`, `Vulkan1`, or CPU (`--no-mmproj-offload`). Encode time follows the CLIP device; the caption is the same in all three logs.

## Issues

- The first CPU `vec_dot` was scalar (~2.3 tok/s). AVX2 recovered most of the gap; Q4_0 still leads via AVX512/VNNI and repack.
- Canonical HQQ dequant on the Vulkan flash-attention KV path produced garbage tokens (`???`). Writing Q4_1 `(d, m)` into the HQQ fields and mapping FA to `Q4_1` restored coherent output. CPU and Metal KV stay real HQQ.
- `--mmproj-backend CPU` is rejected by `parse_device_list`; CPU projector is `--no-mmproj-offload`.
- Default log verbosity (`-lv 3`) drops llama/CLIP INFO. Proof of GPU offload and CLIP device needs `-lv 4` and `--log-file` (PowerShell `Tee-Object` wraps stderr as `NativeCommandError`).

## Further work

- AVX512/VNNI `vec_dot` and a Q4_HQQ CPU repack kernel.
- Vulkan KV write/read with CPU-canonical `scale` / `zero` (Metal already does this).
- Allow `CPU` in `--mmproj-backend` without `--no-mmproj-offload`.
- Perplexity (or a broader eval) for Q4_HQQ vs Q4_0; the quality notes here are single-prompt.

---

# Setup

## Layout

```text
.
├── llama.cpp/                 # patched tree
├── Llama-3.2-3B/              # HF weights + GGUF outputs
├── SmolVLM-256M/              # Task 3 GGUFs
├── logs-final/                # reported logs
└── REPORT.md
```

## Linux (CPU: Tasks 1, 2a, 2b)

Needs: CMake, a C/C++ compiler, Python 3.12, Hugging Face access to gated Llama 3.2 3B.

```bash
conda create -n llama-cpp python=3.12
conda activate llama-cpp
pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

cd llama.cpp
cmake -B build -DGGML_NATIVE=ON
cmake --build build --config Release -j$(nproc)
cd ..
```

Binaries: `llama.cpp/build/bin/llama-completion`, `llama-quantize`, `llama-bench`.

## Windows (Vulkan: Tasks 2c, 3)

Needs: Visual Studio 2022 (Desktop C++), CMake, [Vulkan SDK](https://vulkan.lunarg.com/). Copy this `llama.cpp` tree and the GGUF files onto the Windows machine.

```powershell
cd C:\Users\user\work\llama.cpp
cmake -B build -G "Visual Studio 17 2022" -A x64 -DGGML_VULKAN=ON
cmake --build build --config Release --target llama-completion -j
cmake --build build --config Release --target llama-mtmd-cli -j
```

List devices (names like `Vulkan0` / `Vulkan1` vary by machine):

```powershell
.\build\bin\Release\llama-completion.exe --list-devices
```

## macOS (Metal: Task 2c)

Needs: Xcode Command Line Tools (`xcode-select --install`) and CMake (`cmake --version`; install from cmake.org or Homebrew if missing). Metal is on by default.

```bash
cd llama.cpp
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release --target llama-completion -j$(sysctl -n hw.ncpu)
```

Copy `Llama-3.2-3B/llama-3.2-3b-q4hqq.gguf` onto the Mac; no HF re-download.

---

# Models

## Llama 3.2 3B (Tasks 1–2)

Gated on Hugging Face: request access to `meta-llama/Llama-3.2-3B`, then `huggingface-cli login`.

```bash
conda activate llama-cpp

huggingface-cli download meta-llama/Llama-3.2-3B --local-dir Llama-3.2-3B

python llama.cpp/convert_hf_to_gguf.py Llama-3.2-3B \
  --outfile Llama-3.2-3B/llama-3.2-3b-f16.gguf \
  --outtype f16

./llama.cpp/build/bin/llama-quantize \
  Llama-3.2-3B/llama-3.2-3b-f16.gguf \
  Llama-3.2-3B/llama-3.2-3b-q4_0.gguf \
  Q4_0

./llama.cpp/build/bin/llama-quantize \
  Llama-3.2-3B/llama-3.2-3b-f16.gguf \
  Llama-3.2-3B/llama-3.2-3b-q4hqq.gguf \
  Q4_HQQ
```

| File | Size | Use |
| --- | --- | --- |
| `llama-3.2-3b-f16.gguf` | 6.0 GB | Task 1 baseline |
| `llama-3.2-3b-q4_0.gguf` | 1.8 GB | Task 1, Task 2b (weights) |
| `llama-3.2-3b-q4hqq.gguf` | 2.0 GB | Task 2a, Task 2c |

If these GGUFs already exist, skip convert/quantize.

## SmolVLM-256M (Task 3)

```bash
huggingface-cli download \
  ggml-org/SmolVLM-256M-Instruct-GGUF \
  SmolVLM-256M-Instruct-Q8_0.gguf \
  mmproj-SmolVLM-256M-Instruct-Q8_0.gguf \
  --local-dir ./SmolVLM-256M
```

Test image: `llama.cpp/tools/mtmd/tests/test-1-positive.png`.

---

# Run

CPU runs: discard one cold pass, keep the second. Warm logs show `File system inputs: 0` and `Major page faults: 0` in `/usr/bin/time -v`. Always `-c 4096` (Llama 3.2 train context is 131072; full KV is ~14.9 GB).

Prompts:

```text
What is bitcoin?
Write a Python function to reverse a list.
```

Prompt 2 as used in the logs (tokenizes to 10 tokens):

```text
A Python function that reverses a list:

python
def reverse_list(lst):
```

## Task 1 — F16 vs Q4_0 (CPU)

From `llama.cpp/`:

```bash
/usr/bin/time -v ./build/bin/llama-completion \
  -m ../Llama-3.2-3B/llama-3.2-3b-f16.gguf \
  -p "What is bitcoin?" -n 128 -c 4096 -t 4 \
  --perf --simple-io 2>&1 | tee ../logs-final/task1-f16-prompt1.log
```

Same for prompt 2, and again with `llama-3.2-3b-q4_0.gguf`.

Optional bench:

```bash
./build/bin/llama-bench -m ../Llama-3.2-3B/llama-3.2-3b-q4_0.gguf -p 5,17 -n 128 -t 4 -r 5 -o md
```

## Task 2a — Q4_HQQ weights (CPU)

```bash
/usr/bin/time -v ./build/bin/llama-completion \
  -m ../Llama-3.2-3B/llama-3.2-3b-q4hqq.gguf \
  -p "What is bitcoin?" -n 128 -c 4096 -t 4 \
  --perf --simple-io 2>&1 | tee ../logs-final/task2-q4hqq-prompt1.log
```

## Task 2b — Q4_HQQ KV cache (CPU)

Same Q4_0 weights; only K/V type changes. Quantized V enables flash attention automatically.

```bash
/usr/bin/time -v ./build/bin/llama-completion \
  -m ../Llama-3.2-3B/llama-3.2-3b-q4_0.gguf \
  -p "What is bitcoin?" -n 128 -c 4096 -t 4 \
  -s 42 --temp 0 --perf --simple-io \
  2>&1 | tee ../logs-final/task2-kv-f16-prompt1.log

/usr/bin/time -v ./build/bin/llama-completion \
  -m ../Llama-3.2-3B/llama-3.2-3b-q4_0.gguf \
  -p "What is bitcoin?" -n 128 -c 4096 -t 4 \
  --cache-type-k q4_hqq --cache-type-v q4_hqq \
  -s 42 --temp 0 --perf --simple-io \
  2>&1 | tee ../logs-final/task2-kv-q4hqq-prompt1.log
```

## Task 2c — Vulkan (Windows)

`-lv 4` is required so `file type` / `using device` / `offloaded` appear. `--log-file` instead of `tee` on PowerShell.

```powershell
$exe   = "C:\Users\user\work\llama.cpp\build\bin\Release\llama-completion.exe"
$model = "C:\Users\user\work\Llama-3.2-3B\llama-3.2-3b-q4hqq.gguf"

& $exe -m $model -p "What is bitcoin?" -n 64 -c 4096 `
  -s 42 --temp 0 --device Vulkan1 -ngl 99 --fit off `
  --perf -lv 4 --log-file C:\Users\user\work\logs\task2-q4hqq-vulkan-prompt1.log

& $exe -m $model -p "What is bitcoin?" -n 64 -c 4096 `
  -s 42 --temp 0 --device Vulkan1 -ngl 99 --fit off `
  --cache-type-k q4_hqq --cache-type-v q4_hqq `
  --perf -lv 4 --log-file C:\Users\user\work\logs\task2-q4hqq-vulkan-kv-prompt1.log
```

Expect: `file type = Q4_HQQ`, `offloaded 29/29 layers to GPU`, coherent text. With KV: `K (q4_hqq)` / `V (q4_hqq)`, 140 MiB.

## Task 2c — Metal (macOS)

From `llama.cpp/` on the Mac:

```bash
./build/bin/llama-completion \
  -m ../Llama-3.2-3B/llama-3.2-3b-q4hqq.gguf \
  -p "What is bitcoin?" -n 64 -c 4096 -ngl 99 --perf -lv 4 \
  2>&1 | tee ../logs-final/task2-q4hqq-metal-prompt1.log

./build/bin/llama-completion \
  -m ../Llama-3.2-3B/llama-3.2-3b-q4hqq.gguf \
  -p "What is bitcoin?" -n 64 -c 4096 -ngl 99 \
  --cache-type-k q4_hqq --cache-type-v q4_hqq --perf -lv 4 \
  2>&1 | tee ../logs-final/task2-q4hqq-metal-kv-prompt1.log
```

Expect: `MTL0 (Apple M4)`, `29/29 layers`, `MTL : EMBED_LIBRARY = 1`, coherent text. With KV: 140 MiB `q4_hqq`.

## Task 3 — `--mmproj-backend` (Windows)

```text
--mmproj-backend {DEVICE}     # name from --list-devices, e.g. Vulkan0
```

`--device` / `-ngl` stay on the LLM. CPU projector: `--no-mmproj-offload` (not `--mmproj-backend CPU`).

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

Other two runs: `--mmproj-backend Vulkan1`, and `--no-mmproj-offload` (drop `--mmproj-backend`).

Expect LLM always on `Vulkan1`; CLIP line and encode time follow the projector device.

---

# Logs

| File | What |
| --- | --- |
| `logs-final/task1-*-prompt*.log`, `task1-*-bench.md` | Task 1 CPU |
| `logs-final/task2-q4hqq-prompt*.log`, `task2-q4hqq-bench.md` | Task 2a CPU |
| `logs-final/task2-kv-f16-prompt1.log`, `task2-kv-q4hqq-prompt1.log` | Task 2b CPU KV |
| `logs-final/task2-q4hqq-vulkan-prompt1.log`, `…-vulkan-kv-prompt1.log` | Task 2c Vulkan |
| `logs-final/task2-q4hqq-metal-prompt1.log`, `…-metal-kv-prompt1.log` | Task 2c Metal |
| `logs-final/task3-llm-vulkan1-clip-*.log` | Task 3 |
