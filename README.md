---
license: apache-2.0
size_categories:
  - n<1K
task_categories:
  - text-generation
language:
  - en
tags:
  - chain-of-thought
  - reasoning
  - code
  - cuda
  - arm
  - cpp17
  - geodesy
  - nvidia
  - nemotron
  - groq
pretty_name: "Nemotron-3-Ultra-550B-Systems-CoT: Systems Programming Chain-of-Thought Dataset"
configs:
  - config_name: verified
    default: true
    data_files:
      - split: train
        path: musespark_systems_verified.jsonl
  - config_name: full_benchmark
    data_files:
      - split: test
        path: musespark_systems_full_benchmark.jsonl
---

# Nemotron-3-Ultra-550B-Systems-CoT: Systems Programming Chain-of-Thought Dataset & Multi-Provider Evaluation

**Author:** Lalit Chauhan ([@La1itchauhan](https://github.com/La1itchauhan))  
**Audit Stamp:** `2026-11-10T16:24:09Z` | **Git Revision:** `local-working-tree`  
**Raw Harvested Responses:** `143` | **Deduplicated Benchmark Runs (`full_benchmark`):** `134` | **Verified Split Rows (`verified`):** `36`  
**Distinct Task Families Covered:** `40` in benchmark (`21` in verified) out of `40` total families (`2,500`-prompt bank).

---

## 1. Overview & Scope

`Nemotron-3-Ultra-550B-Systems-CoT` is an open-source dataset and evaluation harness collecting step-by-step Chain-of-Thought (`<think>...</think>`) traces and self-contained C11, C++17, and Python code across four systems domains:
1. **`nvidia_cuda_tensorrt`**: Host-emulated CUDA warp reductions, shared-memory bank-conflict-free matrix transpose, FP4/INT8 groupwise dequantization, and aerial LiDAR voxel downsampling. *(Note: C++17 tasks execute as portable CPU thread-block emulations so they can be checked without a discrete GPU; run with `nvcc` for device-kernel benchmarking.)*
2. **`arm_embedded_rtl`**: Bare-metal C11 (`-std=c11`) MMIO register drivers with Write-1-to-Clear (`W1C`) semantics, lock-free SPSC DMA descriptor rings, Q15.16 fixed-point drone ESC PID loops, and MAVLink v2 frame telemetry parsers.
3. **`modern_cpp17_pybind11`**: Modern C++17 (`-std=c++17`) rational time-range resamplers, `alignas(64)` arena allocators, lock-free work-stealing deques, and 6-DOF IMU+GNSS Kalman filters.
4. **`s2_geodesic_spatial`**: WGS84 spherical excess polygon area (L'Huilier's theorem), S2 Hilbert cell ID encoders, UAV photogrammetry Ground Sample Distance (GSD) ray-DEM intersection, and WGS84-to-NED waypoint converters.

---

## 2. Verification Pipeline & Honest `verification_level` Accounting

Every row in `full_benchmark` and `verified` includes an explicit `verification_level` column:
- **`executed`**: Compiled with native `gcc -std=c11` / `g++ -std=c++17` (with `-Wall -Wextra` and AddressSanitizer/UBSan when supported by the host toolchain) or run in `python3`, exited with code `0`, and passed Agent 4 domain bug checks.
- **`compiled_only`**: Compiled with native `gcc`/`g++` without binary execution.
- **`syntax_fallback`**: Checked in a cloud container where `gcc`/`g++` was not installed (`delimiter_balance_fallback` + C11 keyword enforcement + Agent 4 static bug checks). To re-verify all C/C++ rows to `verification_level == "executed"`, run `python3 musespark_harvester/verify_and_push_hf.py --require-executed` on a workstation or container with `build-essential` (`gcc`/`g++`) installed.

### Current `verified` Split Breakdown (`agent1_compiler | agent2_exec_mode | verification_level`)
- `g++ -std=c++17 -fsyntax-only -Wall -Wextra | g++_cpp_asan_ubsan_exec | executed`: **18** rows
- `gcc -std=c11 -fsyntax-only -Wall -Wextra | gcc_c_asan_ubsan_exec | executed`: **14** rows
- `g++ -std=c++17 -fsyntax-only -Wall -Wextra | g++_cpp_binary_exec | executed`: **3** rows
- `python3 -m py_compile | python3_runtime_sandbox | executed`: **1** rows

---

## 3. Per-Model & Provider Evaluation Summary

*Note on Experimental Confounds:* Comparing `nvidia/nemotron-3-ultra-550b-a55b` on NVIDIA NIM against `openai/gpt-oss-120b` and `openai/gpt-oss-20b` on Groq Cloud reflects joint differences in **model architecture, parameter count, serving provider queues, and output length**, not isolated silicon hardware alone. Both first-attempt pass rates and deduplicated best-of-N rates are reported below with **95% Wilson confidence intervals**.

| Model | Endpoint Provider | Deduplicated Samples ($n$) | Syntax Valid (`Agent 1`) | First-Attempt Verified (95% Wilson CI) | Best-of-N Verified (95% Wilson CI) | Latency P50 / P95 (ms) | Median Throughput | Mean Overthinking Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `nvidia/nemotron-3-ultra-550b-a55b` | NVIDIA NIM API (GPU) | 54 | 28/54 (51.85%) | 7/54 (12.96%, 95% CI [6.42–24.42%]) | 10/54 (18.52%, 95% CI [10.38–30.84%]) | 127617.26 / 324343.6 ms | 50.81 tok/s | 0.2257 |
| `openai/gpt-oss-120b` | Groq Cloud API (LPU) | 72 | 54/72 (75.0%) | 25/72 (34.72%, 95% CI [24.75–46.24%]) | 25/72 (34.72%, 95% CI [24.75–46.24%]) | 20339.64 / 71660.59 ms | 141.82 tok/s | 0.1928 |
| `openai/gpt-oss-20b` | Groq Cloud API (LPU) | 8 | 3/8 (37.5%) | 1/8 (12.5%, 95% CI [2.24–47.09%]) | 1/8 (12.5%, 95% CI [2.24–47.09%]) | 25739.28 / 57114.86 ms | 145.98 tok/s | 0.3461 |

---

## 4. Per-Split Domain Summary

| Split | Deduplicated Samples ($n$) | Syntax Valid (`Agent 1`) | Runtime / Sandbox Pass (`Agent 2`) | First-Attempt Verified (95% Wilson CI) | Best-of-N Verified (95% Wilson CI) | Latency P50 / P95 (ms) | Median Throughput | Mean Overthinking Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `nvidia_cuda_tensorrt` | 35 | 15/35 (42.86%) | 6/35 (17.14%) | 6/35 (17.14%, 95% CI [8.1–32.68%]) | 6/35 (17.14%, 95% CI [8.1–32.68%]) | 31590.6 / 246259.46 ms | 96.65 tok/s | 0.2908 |
| `arm_embedded_rtl` | 32 | 29/32 (90.62%) | 14/32 (43.75%) | 12/32 (37.5%, 95% CI [22.93–54.75%]) | 14/32 (43.75%, 95% CI [28.17–60.67%]) | 27513.48 / 130033.63 ms | 74.05 tok/s | 0.1635 |
| `modern_cpp17_pybind11` | 34 | 16/34 (47.06%) | 7/34 (20.59%) | 7/34 (20.59%, 95% CI [10.35–36.8%]) | 7/34 (20.59%, 95% CI [10.35–36.8%]) | 28789.51 / 219704.18 ms | 95.28 tok/s | 0.1981 |
| `s2_geodesic_spatial` | 33 | 25/33 (75.76%) | 10/33 (30.3%) | 8/33 (24.24%, 95% CI [12.83–41.02%]) | 9/33 (27.27%, 95% CI [15.07–44.22%]) | 31721.58 / 244474.38 ms | 113.69 tok/s | 0.2028 |

---

## 5. Teacher Model Output Licensing & Attribution

The repository code and verification scripts are licensed under **Apache-2.0** (`LICENSE`). The generated model outputs in the dataset are governed by the respective upstream model licenses:

| Source Model | Provider | Upstream Model License | Redistribution & Derivative Terms |
| :--- | :--- | :--- | :--- |
| `nvidia/nemotron-3-ultra-550b-a55b` | NVIDIA NIM | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) | Permits commercial and research use, redistribution of model outputs, and training derivative models. |
| `openai/gpt-oss-120b` & `openai/gpt-oss-20b` | Groq Cloud | Apache-2.0 | Permissive open-weights license permitting output redistribution and derivative model training with attribution. |

---

## 6. Quickstart: Load & Inspect Verification Levels

```python
from datasets import load_dataset

# Load the verified split
ds = load_dataset("La1itchauhan/Nemotron-3-Ultra-550B-Systems-CoT", "verified", split="train")
df = ds.to_pandas()
print(df[["agent1_compiler", "agent2_exec_mode", "verification_level"]].value_counts())

# Filter strictly to native binary/script executed rows
df_executed = df[df["verification_level"] == "executed"]
print(f"Native executed rows: {len(df_executed)}")
```

---

## 7. Citation

```bibtex
@misc{chauhan2026systems_cot,
  title={Nemotron-3-Ultra-550B-Systems-CoT: Systems Programming Chain-of-Thought Dataset and Multi-Provider Evaluation Harness},
  author={Chauhan, Lalit},
  year={2026},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/La1itchauhan/Nemotron-3-Ultra-550B-Systems-CoT}}
}
```
