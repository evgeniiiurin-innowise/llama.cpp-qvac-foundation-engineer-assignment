ggml_vulkan: No devices found.
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama 3B Q4_HQQ                |   1.94 GiB |     3.21 B | Vulkan     |  -1 |             pp5 |          9.67 ± 1.01 |
| llama 3B Q4_HQQ                |   1.94 GiB |     3.21 B | Vulkan     |  -1 |            pp17 |          9.60 ± 0.45 |
| llama 3B Q4_HQQ                |   1.94 GiB |     3.21 B | Vulkan     |  -1 |           tg128 |          7.69 ± 0.32 |

build: 112e9ca (11)
