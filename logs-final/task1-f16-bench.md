ggml_vulkan: No devices found.
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| llama 3B F16                   |   5.98 GiB |     3.21 B | Vulkan     |  -1 |             pp5 |         21.54 ± 4.89 |
| llama 3B F16                   |   5.98 GiB |     3.21 B | Vulkan     |  -1 |            pp17 |         40.89 ± 1.86 |
| llama 3B F16                   |   5.98 GiB |     3.21 B | Vulkan     |  -1 |           tg128 |          6.21 ± 0.26 |

build: 112e9ca (11)
