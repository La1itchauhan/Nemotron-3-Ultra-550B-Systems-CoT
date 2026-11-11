# Nemotron-3-Ultra-550B-Systems-CoT

**A 7-Agent Compiler-Verified Chain-of-Thought Dataset & Multi-Silicon Infrastructure Benchmark for Systems, Silicon, C++17, and Geodesic Engineering**

[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow.svg)](https://huggingface.co/datasets/La1itchauhan/Nemotron-3-Ultra-550B-Systems-CoT)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Verification Level](https://img.shields.io/badge/Verification-100%25%20Native%20Executed-brightgreen.svg)](#1-gate-a-verification-proof-n--36)
[![Toolchain](https://img.shields.io/badge/Toolchain-GCC%2012.3%20%7C%20G%2B%2B%2012.3%20%7C%20ASan%20%7C%20UBSan-orange.svg)](#2-verification-architecture)

**Author:** Lalit Chauhan ([@La1itchauhan](https://github.com/La1itchauhan))  
**Hugging Face Dataset Hub:** [`La1itchauhan/Nemotron-3-Ultra-550B-Systems-CoT`](https://huggingface.co/datasets/La1itchauhan/Nemotron-3-Ultra-550B-Systems-CoT) (Commit: `58fa509ae3`)  
**Audit Stamp:** `2026-11-10T16:24:09Z`  

---

## 1. Gate A Verification Proof: Published `verified` Split ($n = 36$)

Every row (**36 / 36, 100%**) in `musespark_systems_verified.jsonl` was compiled and executed as a native binary (or Python sandbox) with `verification_level == "executed"`. There are **0** `delimiter_balance_fallback` rows and **0** `skipped_compiler_absent` rows. All `arm_embedded_rtl` rows are strictly compiled as ISO C11 (`gcc -std=c11`):

| Count | Split | Extracted Lang | Compiler Command (`Agent 1`) | Runtime Sandbox (`Agent 2`) | Verification Level |
| :---: | :--- | :---: | :--- | :--- | :---: |
| **14** | `arm_embedded_rtl` | `c` | `gcc -std=c11 -fsyntax-only -Wall -Wextra` | `gcc_c_asan_ubsan_exec` | **`executed`** |
| **8** | `s2_geodesic_spatial` | `cpp` | `g++ -std=c++17 -fsyntax-only -Wall -Wextra` | `g++_cpp_asan_ubsan_exec` | **`executed`** |
| **7** | `modern_cpp17_pybind11` | `cpp` | `g++ -std=c++17 -fsyntax-only -Wall -Wextra` | `g++_cpp_asan_ubsan_exec` | **`executed`** |
| **3** | `nvidia_cuda_tensorrt` | `cpp` | `g++ -std=c++17 -fsyntax-only -Wall -Wextra` | `g++_cpp_asan_ubsan_exec` | **`executed`** |
| **3** | `nvidia_cuda_tensorrt` | `cpp` | `g++ -std=c++17 -fsyntax-only -Wall -Wextra` | `g++_cpp_binary_exec` | **`executed`** |
| **1** | `s2_geodesic_spatial` | `python` | `python3 -m py_compile` | `python3_runtime_sandbox` | **`executed`** |
| **0** | *(Any split)* | *(Any)* | `delimiter_balance_fallback` | `skipped_compiler_absent` | *(None)* |

---

## 2. Synchronized Dataset & Audit Metrics

From `council_audit_report.json` (Stamp: `2026-11-10T16:24:09Z`):
- **Raw Harvested Traces:** `143`
- **Deduplicated Benchmark Rows (`full_benchmark`):** `134` covering all **40 / 40** (10 × 4) distinct task families
- **Native-Executed Verified Rows (`verified`):** `36` covering **21** distinct task families (**26.87%** overall native execution pass rate)

### Silicon & Provider Leaderboard (with 95% Wilson Confidence Intervals)

| Model (Provider) | Total Runs ($n$) | First-Attempt Pass@1 (95% Wilson CI) | Compiler Pass % (95% Wilson CI) | Self-Test Exec % | All-4-Gate Verified % (95% Wilson CI) | Native Executed Rows |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `openai/gpt-oss-120b` (Groq) | 72 | **34.72%** (25/72) [24.8%, 46.2%] | **75.00%** [63.9%, 83.6%] | 34.72% | **34.72%** [24.8%, 46.2%] | **25** |
| `nvidia/nemotron-3-ultra-550b-a55b` (NIM) | 54 | **12.96%** (7/54) [6.4%, 24.4%] | **51.85%** [38.9%, 64.6%] | 18.52% | **18.52%** [10.4%, 30.8%] | **10** |
| `openai/gpt-oss-20b` (Groq) | 8 | **12.50%** (1/8) [2.2%, 47.1%] | **37.50%** [13.7%, 69.4%] | 25.00% | **12.50%** [2.2%, 47.1%] | **1** |

### Domain Split Summary

| Split | Samples ($n$) | Syntax Valid (`Agent 1`) | Runtime Pass (`Agent 2`) | First-Attempt Verified (95% Wilson CI) | Best-of-N Verified (95% Wilson CI) | Latency P50 / P95 (ms) | Median Throughput | Mean Overthinking Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `nvidia_cuda_tensorrt` | 35 | 15/35 (42.86%) | 6/35 (17.14%) | 6/35 (17.14%, 95% CI [8.10–32.68%]) | 6/35 (17.14%, 95% CI [8.10–32.68%]) | 31590.6 / 246259.46 ms | 96.65 tok/s | 0.2908 |
| `arm_embedded_rtl` | 32 | 29/32 (90.62%) | 14/32 (43.75%) | 12/32 (37.50%, 95% CI [22.93–54.75%]) | 14/32 (43.75%, 95% CI [28.17–60.67%]) | 27513.48 / 130033.63 ms | 74.05 tok/s | 0.1635 |
| `modern_cpp17_pybind11` | 34 | 16/34 (47.06%) | 7/34 (20.59%) | 7/34 (20.59%, 95% CI [10.35–36.80%]) | 7/34 (20.59%, 95% CI [10.35–36.80%]) | 28789.51 / 219704.18 ms | 95.28 tok/s | 0.1981 |
| `s2_geodesic_spatial` | 33 | 25/33 (75.76%) | 10/33 (30.30%) | 8/33 (24.24%, 95% CI [12.83–41.02%]) | 9/33 (27.27%, 95% CI [15.07–44.22%]) | 31721.58 / 244474.38 ms | 113.69 tok/s | 0.2028 |

---

## 3. Interviewer Mode: Concise Engineering Answers to All 12 Questions

### Q1. Who is this dataset for, and what decision does it let them make?
**Answer:** It is built for post-training / SFT engineers fine-tuning 7B–14B open code models (e.g., `Qwen2.5-Coder-7B-Instruct`) on systems programming (ISO C11 bare-metal firmware, C++17, CUDA host/device orchestration, and WGS84/S2 geometry). It lets them measure whether filtering synthetic Chain-of-Thought traces through native compiler + ASan/UBSan binary execution (`verified`) improves held-out compile-and-run rates compared to training on unfiltered completions (`full_benchmark`).

### Q2. For a row showing `skipped_compiler_absent`, what exactly does "verified" mean?
**Answer:** In v1, `skipped_compiler_absent` meant no compiler or binary ran at all—only bracket counting and regex checks ran. Calling those rows "compiler-verified" was wrong. We installed `gcc 12.3.0`, `g++ 12.3.0`, `libasan8`, and `libubsan1`, added `verification_level`, and enforced `--require-executed` so `verified` now contains exactly **0** `skipped_compiler_absent` rows.

### Q3. Your ARM tasks are C11 but the code is tagged C++. What actually ran, and what passes in one language but fails in the other?
**Answer:** In v1, fenced ```` ```cpp ```` blocks in `arm_embedded_rtl` were routed to `g++`. In v2, `arm_embedded_rtl` enforces `extracted_language = "c"` and runs `gcc -std=c11 -Wall -Wextra`.
- **Valid in C11, invalid in C++17:** implicit `void*` to typed pointer conversion (`uint32_t *p = malloc(64);`), `restrict` pointers (`uint32_t *restrict p`), and out-of-order array designated initializers (`int a[4] = {[2]=5, [0]=1}`).
- **Valid in C++17, invalid in strict C11:** bare `alignas(64)` or `bool` without `#include <stdalign.h>` / `#include <stdbool.h>` (which `gcc -std=c11` caught in `arm_embedded_rtl_0401`), plus `constexpr`, `namespace`, and templates.

### Q4. The same model writes the solution and the tests. What bug does that hide, and how would you build an independent oracle?
**Answer:** Self-written `assert()` tests hide shared-misconception (tautological) bugs: if the model uses a wrong CRC polynomial or misunderstands Write-1-to-Clear (W1C) hardware latches, it writes the exact same wrong expectation in `main()`.
An independent oracle strips the model's `main()` and links the target function against a fixed, human-written test harness with golden test vectors (official MAVLink X.25 CRC-16 frames, GeographicLib WGS84 reference coordinates, and a register-level W1C peripheral emulator).

### Q5. Write me a wrong MMIO driver that passes Agent 4.
**Answer:** In v1, Agent 4 only grepped for `volatile`, `0x`, and `w1c`/`|=`, so this broken driver passed v1 while clearing unhandled interrupts on real W1C hardware:
```c
void irq_ack_buggy(void) {
    volatile uint32_t *status = (volatile uint32_t *)0x40001000UL;
    *status |= (1u << 0); // BUG: Read-Modify-Write on W1C status register!
}
```
In v2, `w1c_no_rmw_hazard_ok` rejects `|=` and `&=` on status registers, and the runtime harness emulates W1C bit-clearing (`reg &= ~val_written`).

### Q6. Why select the best of N runs and then call it Pass@1?
**Answer:** Selecting a passing trace across retries is Best-of-$N$, not Pass@1. `verify_and_push_hf.py` now computes true First-Attempt Pass@1 (`attempt == 0` per `task_id`) separately from Best-of-$N$ / Self-Test Execution, alongside 95% Wilson score confidence intervals.

### Q7. Besides silicon, what differs between your Groq and NIM runs? Design the experiment that isolates silicon.
**Answer:** Our current table compares different models (`openai/gpt-oss-120b` vs `nvidia/nemotron-3-ultra-550b-a55b`), quantization formats, server queues, and output lengths. To isolate silicon, deploy the exact same model checkpoint and precision (e.g., Llama-3.3-70B-Instruct FP8) on both Groq LPU and a dedicated H100/B200 TensorRT-LLM/vLLM server, fix `temperature=0`, `max_tokens=4096`, and concurrency $C \in \{1, 8, 32\}$ across $N=500$ prompts, and measure server-side TTFT and ITL/TPOT.

### Q8. Your 20B finding has $n=8$ and one effort level. What would you need to call it a law?
**Answer:** With $n=8$ deduplicated tasks (95% CI [2.2%, 47.1%]) and a single effort setting, it is an exploratory observation, not a scaling law. A scaling law requires sweeping 4–5 reasoning token budgets (512 to 8192 tokens) across multiple model sizes with $N \ge 100$ tasks per cell and fitting an out-of-sample parametric curve.

### Q9. The report says 62 verified and the card says 38. Which is true, and how do you stop that drift?
**Answer:** Neither v1 number was native-executed. In v2, `council_audit_report.json` (36 native-executed verified rows out of 134 deduplicated benchmark rows, stamped `2026-11-10T16:24:09Z`) is the single source of truth that programmatically generates both `README.md` and `GITHUB_README.md`, enforced by `make check-consistency` in CI.

### Q10. Show me one trace where your verifier caught a real bug.
**Answer:** In `modern_cpp17_pybind11_0003`, the trace passed `g++ -std=c++17 -fsyntax-only -Wall -Wextra` cleanly (`agent1_syntax_ok == true`), but Agent 2's UBSan execution caught a fatal 64-bit signed integer overflow:
```text
runtime error: signed integer overflow: 671154617459008663 * 119 cannot be represented in type 'long int' in operator+(Rational const&, Rational const&)
```
excluding it from `verified`.

### Q11. You have one GPU and a week. What experiment proves this data helps?
**Answer:** Run the 3-arm QLoRA ablation implemented in `musespark_harvester/train_qlora_3arm_eval.py`: fine-tune `Qwen2.5-Coder-7B-Instruct` (4-bit NF4, rank=16) on:
- **Arm 1:** Base model (zero training baseline)
- **Arm 2:** $N$ unverified `full_benchmark` traces
- **Arm 3:** $N$ native-executed `verified` traces

then evaluate all 3 arms on the 8 held-out task families using `verify_and_push_hf.py --require-executed`.

### Q12. Walk me through your llama.cpp PR (#30722) and which parts of this pipeline you can rewrite from scratch.
**Answer:** From this pipeline, I can write from scratch: (a) the `subprocess.run` `gcc -std=c11` / `g++ -std=c++17 -fsanitize=address,undefined` execution harness and `SYS_mmap` bare-metal MMIO page mapper, (b) the C11 W1C status register anti-RMW checker, and (c) the `AsyncOpenAI` semaphore harvester and Wilson score interval calculator. For llama.cpp PR #30722, stick strictly to the exact diff, reproduction command, and reviewer feedback on your GitHub PR.

---

## 4. Quickstart: Reproduce & Load

### Load with Hugging Face `datasets`

```python
from datasets import load_dataset

# Load the 100% native-executed compiler-verified split
ds_verified = load_dataset("La1itchauhan/Nemotron-3-Ultra-550B-Systems-CoT", "verified", split="train")
print(f"Verified rows: {len(ds_verified)}")
print(ds_verified[0]["messages"])

# Load the full empirical benchmark with compiler diagnostics & latency telemetry
ds_bench = load_dataset("La1itchauhan/Nemotron-3-Ultra-550B-Systems-CoT", "full_benchmark", split="test")
print(f"Benchmark rows: {len(ds_bench)}")
```

### Local Test & Consistency Check

```bash
# Verify council self-test
make test

# Check report vs documentation consistency
make check-consistency
```

---

## 5. Citation

```bibtex
@misc{chauhan2026nemotron3_ultra_systems_cot,
  title={Nemotron-3-Ultra-550B-Systems-CoT: A 7-Agent Compiler-Verified Chain-of-Thought Dataset and Multi-Silicon Infrastructure Benchmark for Systems, Silicon, C++17, and Geodesic Engineering},
  author={Chauhan, Lalit},
  year={2026},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/La1itchauhan/Nemotron-3-Ultra-550B-Systems-CoT}}
}
```
