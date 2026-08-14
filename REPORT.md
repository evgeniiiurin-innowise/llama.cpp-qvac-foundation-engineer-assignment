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

<!-- TODO -->

---

## Task 3: `--mmproj-backend`

<!-- TODO -->

---

# Appendix

## A0. Environment

| Item | Value |
| --- | --- |
| OS | WSL2, Linux 6.18, x86_64 |
| RAM | 15 GiB |
| CPU threads used | 4 (`-t 4`, 8 logical cores available) |
| ISA | AVX2, AVX512, AVX512_VNNI (`REPACK = 1`) |
| llama.cpp commit | `112e9ca` |
| Build | `cmake -B build -DGGML_NATIVE=ON`, Release |
| Inference backend | CPU (no GPU offload) |

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

The bench logs print `ggml_vulkan: No devices found` yet label the backend as Vulkan; on this WSL host there is no usable GPU, and the throughput matches CPU completion, so we treat the results as CPU-equivalent. Note that `pp17` is only an approximation for prompt 2 - the actual prompt tokenizes to 10 tokens, not 17.

#### Output quality

Both models behave like a base model rather than an instruction-tuned assistant. On prompt 1, F16 and Q4_0 both produce coherent article-style text about Bitcoin with no special-token artifacts.

On prompt 2, Q4_0 returns a working `return lst[::-1]` but then repeats the same paragraph about Python lists. F16 also produces runnable logic but with broken formatting (`def  reverse_list ( l )`) and test-case phrasing. Repetition is not suppressed: `repeat_penalty = 1.000` in all runs. This is a single-prompt observation; a systematic quality comparison would need perplexity or a broader eval set.

---

## A2. Task 2 - details

<!-- TODO -->

---

## A3. Task 3 - details

<!-- TODO -->
