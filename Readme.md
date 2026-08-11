# Task 1: Quantize and Run a Model

Llama 3.2 3B (base, not Instruct) -> GGUF F16 -> Q4_0, inference on CPU and measuring performance metrics.

## Setup

Environment: WSL2 (Ubuntu), x86_64, 8 logical cores, 15 GiB RAM, CPU-only, AVX2 / AVX512 / AVX512_VNNI. All commands are run from the repository root, except steps 4-5, which are run from `llama.cpp/`.

```bash
# 1. Build llama.cpp
cd llama.cpp
cmake -B build -DGGML_NATIVE=ON
cmake --build build --config Release -j$(nproc)
cd ..

# 2. Python environment for conversion
conda create -n llama-cpp python=3.12
conda activate llama-cpp
pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

# 3. Download the model (gated: requires an approved request on HF)
huggingface-cli download meta-llama/Llama-3.2-3B --local-dir Llama-3.2-3B

# 4. Convert HF -> GGUF F16
python llama.cpp/convert_hf_to_gguf.py Llama-3.2-3B \
  --outfile Llama-3.2-3B/llama-3.2-3b-f16.gguf \
  --outtype f16

# 5. Quantize F16 -> Q4_0
./llama.cpp/build/bin/llama-quantize \
  Llama-3.2-3B/llama-3.2-3b-f16.gguf \
  Llama-3.2-3B/llama-3.2-3b-q4_0.gguf \
  Q4_0
```

Step 3 requires downloading HF-format weights (`config.json` + `*.safetensors`). The `original/` folder with `consolidated.00.pth` is not suitable for this converter.

## Run inference

| Run | Command |
| --- | --- |
| F16, P1 | `/usr/bin/time -v ./build/bin/llama-completion -m ../Llama-3.2-3B/llama-3.2-3b-f16.gguf -p "What is bitcoin?" -n 128 -c 4096 --perf --simple-io 2>&1 \| tee ../logs/prompt1-f16.log` |
| F16, P2 | `/usr/bin/time -v ./build/bin/llama-completion -m ../Llama-3.2-3B/llama-3.2-3b-f16.gguf -p "A Python function that reverses a list:\n\npython\ndef reverse_list(lst):\n" -n 128 -c 4096 --perf --simple-io 2>&1 \| tee ../logs/prompt2-f16.log` |
| Q4_0, P1 | `/usr/bin/time -v ./build/bin/llama-completion -m ../Llama-3.2-3B/llama-3.2-3b-q4_0.gguf -p "What is bitcoin?" -n 128 -c 4096 --perf --simple-io 2>&1 \| tee ../logs/prompt1.log` |
| Q4_0, P2 | `/usr/bin/time -v ./build/bin/llama-completion -m ../Llama-3.2-3B/llama-3.2-3b-q4_0.gguf -p "A Python function that reverses a list:\n\npython\ndef reverse_list(lst):\n" -n 128 -c 4096 --perf --simple-io 2>&1 \| tee ../logs/prompt2.log` |

Each run was executed twice; the report contains the second run. See the [Warmup](#warmup) section for the justification.

## Results

For all runs: `-n 128`, `-c 4096`, and 4 threads.

### Prompt 1: "What is bitcoin?" (5 prompt tokens)

Logs: `logs/prompt1-f16.log`, `logs/prompt1.log`

| Metric | F16 | Q4_0 | Q4_0 win |
| --- | --- | --- | --- |
| Model file size | 6.0 GB | 1.8 GB | 3.3x |
| First token latency | 185.28 ms | 119.56 ms | 1.55x |
| Prefill per token | 37.06 ms | 23.91 ms | 1.55x |
| Prefill speed | 26.99 tok/s | 41.82 tok/s | 1.55x |
| Avg generation speed | 6.09 tok/s | 16.11 tok/s | 2.64x |
| Generation per token | 164.10 ms | 62.06 ms | 2.64x |
| Time to generate 127 tokens | 20840.21 ms | 7882.19 ms | 2.64x |
| Peak RAM | 6.50 GB | 3.75 GB | 1.73x |

### Prompt 2: "A Python function that reverses a list" (17 prompt tokens)

Logs: `logs/prompt2-f16.log`, `logs/prompt2.log`

| Metric | F16 | Q4_0 | Q4_0 win |
| --- | --- | --- | --- |
| Model file size | 6.0 GB | 1.8 GB | 3.3x |
| First token latency | 402.88 ms | 255.52 ms | 1.58x |
| Prefill per token | 23.70 ms | 15.03 ms | 1.58x |
| Prefill speed | 42.20 tok/s | 66.53 tok/s | 1.58x |
| Avg generation speed | 6.44 tok/s | 16.30 tok/s | 2.53x |
| Generation per token | 155.33 ms | 61.34 ms | 2.53x |
| Time to generate 127 tokens | 19727.27 ms | 7790.58 ms | 2.53x |
| Peak RAM | 6.50 GB | 3.76 GB | 1.73x |

## Measurements

### Context length explicitly set: `-c 4096`

Without `-c`, context length comes from model metadata. For Llama 3.2 that is 131072 tokens.

KV cache is allocated for the whole context at once, regardless of prompt length:

```
2 (K and V) x 28 layers x 8 KV heads x 128 head_dim x 131072 tokens x 2 bytes (f16) = 14.9 GB
```

Those 14.9 GB were measured before adding the `-c` flag: peak RSS was 14946 MB on a 15 GiB RAM machine, and generation speed dropped to 13.3 tok/s. With `-c 4096`, KV cache takes about 0.46 GB.

### Repack enabled on purpose

In `system_info`, `REPACK = 1`. For Q4_0, llama.cpp repacks weights into a layout optimized for SIMD, and keeps a separate in-RAM copy (about ~1.5 GB).

The `--no-repack` flag was not used intentionally: the goal is to show memory usage of real inference. repack is the default behavior and brings speed improvements. However, while interpreting numbers, remember that this extra copy affects the arithmetic: 1.8 GB weights on disk vs 3.75 GB peak RSS in RAM.

### Memory breakdown

| Component | F16 | Q4_0 |
| --- | --- | --- |
| Model weights (mmap) | 6.0 GB | 1.8 GB |
| KV cache (4096 tokens) | ~0.46 GB | ~0.46 GB |
| Repack weights | not applicable | ~1.5 GB |
| Total measured | 6.50 GB | 3.75 GB |

For F16 the arithmetic matches directly: 6.0 + 0.46 = 6.46 GB vs measured 6.50 GB.

For Q4_0, the repack contribution was verified experimentally: with `--no-repack`, peak RSS drops from 3.75 GB to 2.29 GB, which corresponds to `weights + KV cache`.

### How first token latency is computed

Take the `prompt eval time` line from the `--perf` output.

There is no separate TTFT metric in llama.cpp, but `prompt eval time` is practically equivalent.
Inference is split into two phases: prefill processes the whole prompt as one batch, and decode generates one token per step.
The first token is sampled from the logits of the last position of the prefill pass, without a separate decode step.
This is visible in the logs: with `-n 128`, `eval time` contains only 127 runs, not 128.

Formally, `prompt eval time` should also include sampling of the first token. But `sampling time = 35 ms` across all 133 tokens is only a small percentage. So separately measuring it is not meaningful.

### Warmup

The first run on a cold page cache reads weights from SSD lazily, during prefill. As a result, `prompt eval time` reflects disk speed instead of compute speed.
In the first Q4_0 measurement, this gave 8194 ms (prefill per token) versus 75 ms (generation per token) which is incorrect ordering for compute (prefill should not be slower than decode that much). After warmup, it became 119 ms.

The warmup indicator in the log is: `File system inputs: 0` and `Major page faults: 0`.

It is separately noted that `prompt1-f16.log` is formally cold (it shows `File system inputs: 12565808` blocks of 512 bytes, i.e. 6.4 GB read from disk), but it did not affect the metrics: warmup is enabled in llama.cpp by default and warmed the pages.

### Output quality

Both models produce coherent output without special tokens and behave like a base model, not an assistant.
For "What is bitcoin?", the output is in the style of a titled article rather than a direct answer.

On the first prompt, there is no visible difference in quality: both outputs are factually acceptable.

On the second prompt, F16 is subjectively better.
It outputs a working function using `new_lst.insert(0, x)` with a correct explanation, and then it moves to a similar example in JavaScript.
Q4_0 also wrote working code (`return lst[::-1]`), but the explanation does not match the code.
It mentions `reverse()` and "slice notation to access the last few elements", and then the model got stuck repeating one paragraph.
Repetition is made worse because the repeat penalty is effectively disabled: `repeat_penalty = 1.000`.

## Conclusions

* Q4_0 quantization speeds up generation by **2.5-2.6x** (16.1-16.3 vs 6.1-6.4 tok/s).
  Decode is limited by memory bandwidth, not arithmetic. For each token, all model weights must be streamed through memory. The Q4_0 model has about 3.3x fewer bytes, hence the speedup.
* Prefill is accelerated more modestly: **1.55-1.58x**. Prefill uses prompt batching and is dominated by matrix multiplications; the weight size matters less there.
* Peak RAM savings (**1.73x**) is much smaller than disk savings (**3.3x**). Reason: both models share the same KV cache, and for Q4_0 there is also an in-RAM repacked copy of weights.
* No noticeable quality degradation for Q4_0 was found, but on the second prompt F16 looks better: its explanation matches the code and it does not loop.
  This is only one observation on one prompt; to conclude about systematic degradation, a perplexity benchmark (or similar) would be needed.

