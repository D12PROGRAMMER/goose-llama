# goose-llama

llama.cpp's inference path for GGUF models (qwen2, llama), rewritten in [Goose](https://github.com/aardappel/goose): CPU, and CUDA with `--gpu`.

Support: [GoFundMe](https://gofund.me/af0569852)

## Results

i9-13900KF, DDR5-6000, Windows 11, Qwen2.5-1.5B-Instruct, CPU only, llama.cpp b10566. Median of 3 runs; data in `results/results.json`.

| tokens/s | goose-llama | llama.cpp |
|---|---|---|
| Q8_0 decode | 49.1 | 45.8 |
| Q8_0 speculative | 124.2 | 92.1 |
| Q8_0 prefill, 128 tokens | 313.1 | 298.5 |
| Q4_K_M decode | 75.0 | 73.7 |
| Q4_K_M speculative | 165.6 | 117.1 |
| Q4_K_M prefill, 128 tokens | 341.1 | 291.5 |

- **Decode:** 256 tokens, no drafting, vs `llama-bench` tg128.
- **Speculative:** `test/p_code.txt` with n-gram drafts, vs `llama-lookup`.
- **Prefill:** vs `llama-bench` pp128.

Perplexity, `test/ppl.txt` with `-c 128` (goose-llama vs llama.cpp): Q8_0 10.41 vs 10.43; Q4_K_M 10.77 vs 10.86.

### GPU

RTX 4070 Ti, Qwen2.5-1.5B-Instruct, `--gpu` vs llama.cpp CUDA (`-ngl 99`, commit bb4caa7). Median of 5 runs; data in `results/results_gpu.json`.

| tokens/s | Q8_0 goose-llama | Q8_0 llama.cpp | Q4_K_M goose-llama | Q4_K_M llama.cpp |
|---|---|---|---|---|
| decode | 253.8 | 236.5 | 362.5 | 336.2 |
| decode, 2048 context | 238.1 | 227.3 | 334.3 | 319.5 |
| prefill, 128 tokens | 19649 | 13159 | 21794 | 14397 |
| prefill, 512 tokens | 33327 | 24575 | 28610 | 23442 |
| prefill, 1024 tokens | 32435 | 23537 | 28331 | 22769 |
| prefill, 2048 tokens | 30702 | 23258 | 26980 | 22537 |
| speculative | 736.6 | 464.2 | 793.3 | 588.9 |

Kernels are Goose `cuda fn`s (the patch adds the backend): int8 tensor-core GEMMs on tile-major weights, f16 tensor-core flash attention for prompts, dp4a decode, CUDA graphs.

![](results/fig1_throughput.svg)
![](results/fig2_decode_threads.svg)
![](results/fig3_prefill_length.svg)

## Build

```
git clone https://github.com/aardappel/goose ..\goose
cd ..\goose && git checkout 74c641c && git apply ..\goose-llama\goose\goose-llama.patch && cd ..\goose-llama
tools\build_goose.bat
..\goose\build\goose.exe build llama.goose --cc clang --cuda-unchecked -o llama.exe
tools\pinhi.bat FFFF llama.exe -m model.gguf -p "Hello" -n 64
llama.exe -m model.gguf -p "Hello" -n 64 --gpu
```

The build needs the CUDA toolkit (nvcc); `--gpu` needs an NVIDIA GPU.

Reproduce: `python test/compare.py && python test/figs.py`; GPU: `python test/compare_gpu.py`

## License

`goose/goose-llama.patch`, `test/ppl*.txt` and `test/p_pp*.txt` (from Goose's docs) are Apache-2.0; see `goose/LICENSE`.
