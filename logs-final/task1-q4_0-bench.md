ggml_vulkan: No devices found.
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama 3B Q4_0                  |   1.78 GiB |     3.21 B | Vulkan     |  -1 |             pp5 |         43.24 ± 2.16 |
| llama 3B Q4_0                  |   1.78 GiB |     3.21 B | Vulkan     |  -1 |            pp17 |         64.41 ± 4.05 |
| llama 3B Q4_0                  |   1.78 GiB |     3.21 B | Vulkan     |  -1 |           tg128 |         17.25 ± 1.28 |

build: 112e9ca (11)
