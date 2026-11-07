#!/usr/bin/env python3
"""
Nemotron-3-Ultra-550B-Systems-CoT: Multi-Silicon Asynchronous Harvester
Author: Lalit Chauhan (@La1itchauhan)
License: Apache-2.0

Harvests deep Chain-of-Thought (<think>) systems reasoning across:
- NVIDIA NIM (nvidia/nemotron-3-ultra-550b-a55b)
- Groq LPU (openai/gpt-oss-120b & openai/gpt-oss-20b)
- Baseten (meta-models/muse-spark-open-1026)
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

try:
    from openai import AsyncOpenAI
except ImportError:
    print("[ERROR] 'openai' package not installed. Run: pip install openai")
    sys.exit(1)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
PROMPT_BANK_FILE = OUTPUT_DIR / "musespark_prompt_bank.jsonl"
RAW_RESPONSES_FILE = OUTPUT_DIR / "musespark_raw_responses.jsonl"
BENCHMARK_FILE = OUTPUT_DIR / "musespark_systems_full_benchmark.jsonl"

SPLIT_SPECS = [
    {
        "split": "nvidia_cuda_tensorrt",
        "domain": "NVIDIA CUDA / TensorRT / AI Cluster Systems",
        "lang": "cpp",
        "subcategories": [
            ("Warp-Level Reduction & Online Softmax", "float16 warp shuffle reduction (__shfl_xor_sync), numerical stability, bank-conflict-free tiling"),
            ("Bank-Conflict-Free Shared-Memory Matrix Transpose", "32x33 padded shared memory tile to eliminate 32-way bank conflicts, coalesced global reads/writes"),
            ("NVFP4 / INT8 Groupwise Dequantization GEMV", "4-bit floating point scale-factor unrolling, vectorized float4 loads, register caching"),
            ("FlashAttention-Style Online Tile Rescaling", "softmax running max and sum accumulator update without materializing N*N attention matrix"),
            ("Aerial LiDAR Warp-Aggregated Voxel Downsampling", "atomicAggInc warp-aggregated counter increment, atomicCAS voxel hash map"),
            ("PagedAttention Block Table Physical-Logical Page Resolver", "KV cache block table lookup, dynamic allocation, non-contiguous physical page access"),
        ],
    },
    {
        "split": "arm_embedded_rtl",
        "domain": "ARM System Architecture / Bare-Metal C / Embedded Firmware",
        "lang": "c",
        "subcategories": [
            ("ARMv8-A MMIO Peripheral Register Driver with W1C Status Bits", "volatile register access, atomic bitmask manipulation, Write-1-to-Clear interrupt status registers"),
            ("ARM GICv3 Interrupt Controller Priority Router", "ICC_PMR_EL1 priority masking, distributor and redistributor configuration, nested ISR dispatch"),
            ("Lock-Free Single-Producer Single-Consumer DMA Descriptor Ring", "memory barriers (__dmb, __isb), cache line alignment (alignas(64)), circular head/tail management"),
            ("Fixed-Point Q15.16 PID Motor Controller for Drone ESCs", "32-bit integer arithmetic with 16 fractional bits, anti-windup clamping, zero floating-point emulation"),
            ("Zero-Copy MAVLink v2 Frame Telemetry Parser & CRC-16 Accumulator", "state-machine packet decode, payload framing, ITU X.25 CRC checksum validation"),
        ],
    },
    {
        "split": "modern_cpp17_pybind11",
        "domain": "High-Performance Modern C++17 & Systems Architecture",
        "lang": "cpp",
        "subcategories": [
            ("Rational Time-Range Resampler & Media Track Gap Resolver", "OpenTimelineIO-style exact rational time arithmetic (int64_t numerator, double rate), zero floating-point drift"),
            ("Zero-Copy Strided Multidimensional Tensor Buffer Descriptor", "row-major / column-major stride computation, pybind11 buffer_info compatible view, bounds checking"),
            ("Cache-Aligned alignas(64) Slab & Arena Memory Allocator", "64-byte L1 cache-line alignment to prevent false sharing, bump allocation with block chaining"),
            ("Lock-Free Work-Stealing Task Deque using std::atomic", "Chase-Lev work-stealing deque, std::memory_order acquire/release/seq_cst semantics"),
            ("6-DOF IMU + GNSS Extended Kalman Filter State Estimator", "orientation quaternion kinematics, state covariance propagation, measurement residual update"),
        ],
    },
    {
        "split": "s2_geodesic_spatial",
        "domain": "Geodesic Spatial Computing & Satellite / UAV Mathematics",
        "lang": "cpp",
        "subcategories": [
            ("Spherical Excess Geodesic Polygon Area on WGS84 Ellipsoid", "Girard / L'Huilier spherical excess formula, authalic latitude transformation, nanometer area precision"),
            ("S2 Cell Cube-Face Projection & 64-bit Hilbert Curve Encoder", "tangent coordinate mapping, space-filling Hilbert curve traversal, uint64_t S2CellId generation"),
            ("Drone Photogrammetry GSD & Camera Ray-DEM Intersection Solver", "ground sample distance (GSD = H*sensor_w / (f*image_w)), pinhole ray-plane terrain intersection"),
            ("WGS84 Geodetic to Local Tangent Plane (NED) Coordinate Transformer", "ECEF to North-East-Down transformation matrix, reference origin anchoring, sub-millimeter precision"),
            ("Geofence Convex-Hull Crossing Detector with Haversine Fallback", "great-circle distance calculation, winding number polygon inclusion check for drone flight paths"),
        ],
    },
]


def generate_prompt_bank(target_count: int = 2500) -> List[Dict]:
    """Generates deterministic, high-signal prompt bank across 4 domain splits."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts = []
    per_split = target_count // len(SPLIT_SPECS)
    remainder = target_count % len(SPLIT_SPECS)

    for s_idx, spec in enumerate(SPLIT_SPECS):
        count = per_split + (1 if s_idx < remainder else 0)
        split_name = spec["split"]
        domain = spec["domain"]
        lang = spec["lang"]
        subcats = spec["subcategories"]

        for i in range(count):
            subcat_title, subcat_desc = subcats[i % len(subcats)]
            task_id = f"{split_name}_{i+1:04d}"
            
            prompt_text = (
                f"[Domain: {domain}]\n"
                f"Design and implement a complete, production-grade, self-contained, compilable {lang.upper()} solution for "
                f"**{subcat_title}**.\n\n"
                f"Core Requirements & Invariants:\n"
                f"- Architecture context: {subcat_desc}.\n"
                f"- Follow modern standards (C++17 for C++, C11 for C, Python 3.10+ for Python).\n"
                f"- Include complete types, header inclusions, proper memory management, and zero dummy TODO stubs.\n"
                f"- Include a self-contained unit test / verification harness in `main()` with assert() validations.\n"
                f"- Provide deep architectural rationale and step-by-step mathematical/hardware reasoning before the code block."
            )
            prompt_sha = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()

            prompts.append({
                "task_id": task_id,
                "split": split_name,
                "subcategory": subcat_title,
                "language": lang,
                "difficulty": "Principal Systems Engineer",
                "prompt_sha256": prompt_sha,
                "prompt": prompt_text,
            })

    with open(PROMPT_BANK_FILE, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"[PROMPT BANK] Generated {len(prompts)} unique prompts -> {PROMPT_BANK_FILE}")
    return prompts


def load_completed_tasks() -> Set[Tuple[str, str]]:
    """Loads already completed (model_id, task_id) pairs to support crash-safe resumption."""
    completed = set()
    for filepath in [RAW_RESPONSES_FILE, BENCHMARK_FILE]:
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        m = record.get("model") or record.get("model_id") or ""
                        t = record.get("task_id") or ""
                        if m and t:
                            completed.add((m, t))
                    except Exception:
                        continue
    return completed


def get_provider_config(provider: str, custom_model: Optional[str] = None):
    """Resolves provider endpoints, credentials, and default models."""
    provider = provider.lower()
    if provider == "nvidia":
        api_key = os.getenv("NVIDIA_API_KEY", "")
        base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        model = custom_model or os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
        extra_body = {"chat_template_kwargs": {"enable_thinking": True}}
        default_max_tokens = 16384
    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "")
        base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        model = custom_model or "openai/gpt-oss-120b"
        extra_body = {}
        default_max_tokens = 8192
    elif provider == "baseten":
        api_key = os.getenv("BASETEN_API_KEY", "")
        base_url = os.getenv("BASETEN_BASE_URL", "https://inference.baseten.co/v1")
        model = custom_model or os.getenv("BASETEN_MODEL", "meta-models/muse-spark-open-1026")
        extra_body = {}
        default_max_tokens = 4096
    else:
        raise ValueError(f"Unknown provider '{provider}'. Expected: nvidia, groq, baseten")

    return {
        "provider": provider,
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "extra_body": extra_body,
        "default_max_tokens": default_max_tokens,
    }


def extract_reasoning_and_answer(msg_obj) -> Tuple[str, str]:
    """Extracts Chain-of-Thought reasoning (<think>) and clean final code answer."""
    content = getattr(msg_obj, "content", "") or ""
    reasoning = getattr(msg_obj, "reasoning_content", "") or getattr(msg_obj, "reasoning", "") or ""

    if not reasoning and "<think>" in content and "</think>" in content:
        match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
            content = content[match.end():].strip()

    return reasoning.strip(), content.strip()


async def harvest_task(
    client: AsyncOpenAI,
    task: Dict,
    model_id: str,
    extra_body: Dict,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    file_lock: asyncio.Lock,
) -> Optional[Dict]:
    """Executes a single async task against the selected provider."""
    async with semaphore:
        t0 = time.time()
        try:
            kwargs = {
                "model": model_id,
                "messages": [{"role": "user", "content": task["prompt"]}],
                "max_tokens": max_tokens,
                "temperature": 0.2,
            }
            if extra_body:
                kwargs["extra_body"] = extra_body

            resp = await client.chat.completions.create(**kwargs)
            latency_ms = round((time.time() - t0) * 1000.0, 2)
            choice = resp.choices[0]
            msg = choice.message
            finish_reason = choice.finish_reason or "stop"

            reasoning_trace, final_answer = extract_reasoning_and_answer(msg)
            usage = resp.usage
            prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
            completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
            reasoning_tokens = (
                getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", 0)
                if usage else 0
            )
            total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) if usage else 0

            tokens_per_sec = (
                round(completion_tokens / (latency_ms / 1000.0), 2)
                if latency_ms > 0 and completion_tokens > 0 else 0.0
            )
            overthinking_ratio = (
                round(reasoning_tokens / completion_tokens, 4)
                if completion_tokens > 0 and reasoning_tokens > 0 else 0.0
            )

            record = {
                "task_id": task["task_id"],
                "split": task["split"],
                "subcategory": task["subcategory"],
                "language": task["language"],
                "difficulty": task["difficulty"],
                "prompt_sha256": task["prompt_sha256"],
                "prompt": task["prompt"],
                "model": model_id,
                "reasoning_effort": "high",
                "reasoning_trace": reasoning_trace,
                "final_answer": final_answer,
                "finish_reason": finish_reason,
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "reasoning_tokens": reasoning_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "tokens_per_sec": tokens_per_sec,
                "overthinking_ratio": overthinking_ratio,
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

            async with file_lock:
                with open(RAW_RESPONSES_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(
                f"[HARVEST DONE] {task['task_id']} | model={model_id.split('/')[-1]} | "
                f"latency={latency_ms:.0f}ms | tok/s={tokens_per_sec} | finish={finish_reason}"
            )
            return record

        except Exception as e:
            print(f"[HARVEST ERROR] {task['task_id']} on {model_id}: {e}")
            return None


async def run_harvester(
    provider: str,
    model: Optional[str] = None,
    max_tasks: int = 100,
    concurrency: int = 4,
    max_tokens: Optional[int] = None,
    smoke_test: bool = False,
):
    cfg = get_provider_config(provider, model)
    if not cfg["api_key"]:
        print(f"[ERROR] No API key found for provider '{provider}'. Check your .env file.")
        return

    tokens_limit = max_tokens or cfg["default_max_tokens"]
    if smoke_test:
        max_tasks = 1
        concurrency = 1

    if not PROMPT_BANK_FILE.exists():
        generate_prompt_bank(2500)

    prompt_bank = []
    with open(PROMPT_BANK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompt_bank.append(json.loads(line))

    completed = load_completed_tasks()
    model_id = cfg["model"]
    pending_tasks = [t for t in prompt_bank if (model_id, t["task_id"]) not in completed][:max_tasks]

    print(
        f"[START HARVEST] Provider: {provider.upper()} | Model: {model_id} | "
        f"Tasks to run: {len(pending_tasks)} | Concurrency: {concurrency} | Max tokens: {tokens_limit}"
    )

    if not pending_tasks:
        print("[INFO] All requested tasks already completed for this model.")
        return

    client = AsyncOpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
    semaphore = asyncio.Semaphore(concurrency)
    file_lock = asyncio.Lock()

    tasks = [
        harvest_task(client, t, model_id, cfg["extra_body"], tokens_limit, semaphore, file_lock)
        for t in pending_tasks
    ]
    results = await asyncio.gather(*tasks)
    success = sum(1 for r in results if r is not None)
    print(f"\n[HARVEST FINISHED] Successfully harvested {success}/{len(pending_tasks)} tasks to {RAW_RESPONSES_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Silicon Async Harvester for Systems CoT")
    parser.add_argument("--provider", choices=["nvidia", "groq", "baseten"], default="nvidia")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max-tasks", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    asyncio.run(
        run_harvester(
            provider=args.provider,
            model=args.model,
            max_tasks=args.max_tasks,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            smoke_test=args.smoke_test,
        )
    )


if __name__ == "__main__":
    main()
