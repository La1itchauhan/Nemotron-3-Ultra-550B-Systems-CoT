#!/usr/bin/env python3
from __future__ import annotations
"""
Nemotron-3-Ultra-550B-Systems-CoT: 3-Arm QLoRA Ablation & Verification Harness
Author: Lalit Chauhan (@La1itchauhan)
License: Apache-2.0

Implements the 3-arm ablation experiment proving the empirical value of
native-executed compiler verification for systems programming:
- Arm 1: Base Model (Qwen2.5-Coder-7B-Instruct, zero SFT)
- Arm 2: Unfiltered CoT SFT (Trained on full_benchmark unverified traces)
- Arm 3: Verified CoT SFT (Trained on 100% native gcc/g++ executed traces)

Evaluation:
- 8 held-out task families evaluated with verify_and_push_hf.py --require-executed
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import torch
    from datasets import Dataset, load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer
    HAS_TRAIN_DEPS = True
except ImportError:
    HAS_TRAIN_DEPS = False

# Import VerificationCouncil from local package
try:
    from verify_and_push_hf import VerificationCouncil, extract_code_blocks
except ImportError:
    try:
        from musespark_harvester.verify_and_push_hf import VerificationCouncil, extract_code_blocks
    except ImportError:
        VerificationCouncil = None


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
VERIFIED_PATH = OUTPUT_DIR / "musespark_systems_verified.jsonl"
BENCHMARK_PATH = OUTPUT_DIR / "musespark_systems_full_benchmark.jsonl"
SPLITS_DIR = OUTPUT_DIR / "qlora_splits"


def get_qlora_config(rank: int = 16, alpha: int = 32, dropout: float = 0.05) -> LoraConfig:
    """Returns standard 4-bit QLoRA PEFT configuration for decoder-only code models."""
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_4bit_model_and_tokenizer(model_id: str):
    """Loads target model with 4-bit NormalFloat (NF4) quantization and double quant."""
    if not HAS_TRAIN_DEPS:
        raise ImportError("Training dependencies (torch, transformers, peft, trl, bitsandbytes) required.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def prepare_training_split(arm: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Prepares SFT formatted dataset for target ablation arm."""
    records = []
    if arm == "arm1_base":
        return []

    # Check qlora_splits first if present
    arm3_split = SPLITS_DIR / "arm3_verified_35.jsonl"
    arm2_split = SPLITS_DIR / "arm2_unverified_35.jsonl"

    if arm == "arm3_verified" and arm3_split.exists():
        source_path = arm3_split
    elif arm == "arm2_unverified" and arm2_split.exists():
        source_path = arm2_split
    else:
        source_path = VERIFIED_PATH if arm == "arm3_verified" else BENCHMARK_PATH

    if not source_path.exists():
        raise FileNotFoundError(f"Source dataset not found at {source_path}")

    with open(source_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                records.append(rec)

    if max_samples and len(records) > max_samples:
        records = records[:max_samples]

    print(f"[DATASET] Prepared {len(records)} samples for {arm} from {source_path.name}")
    return records


def run_training_arm(
    arm: str,
    base_model_id: str = DEFAULT_BASE_MODEL,
    output_dir: str = "./qlora_output",
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 2e-4,
    max_samples: Optional[int] = None,
):
    """Executes a single experimental arm of the 3-arm ablation."""
    print(f"\n{'='*70}\n[START ABLATION] Arm: {arm.upper()} | Model: {base_model_id}\n{'='*70}")

    if arm == "arm1_base":
        print("[ARM 1] Base model selected. Skipping training, ready for zero-shot evaluation.")
        return

    data = prepare_training_split(arm, max_samples=max_samples)
    if not data:
        print(f"[ERROR] No training data found for {arm}.")
        return

    hf_dataset = Dataset.from_list([
        {"messages": r.get("messages", [
            {"role": "user", "content": r.get("prompt", "")},
            {"role": "assistant", "content": f"<think>\n{r.get('reasoning_trace', '')}\n</think>\n\n{r.get('final_answer', '')}"}
        ])}
        for r in data
    ])

    model, tokenizer = load_4bit_model_and_tokenizer(base_model_id)
    peft_config = get_qlora_config()
    model = get_peft_model(model, peft_config)

    arm_output = Path(output_dir) / arm
    arm_output.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(arm_output),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=hf_dataset,
        peft_config=peft_config,
        dataset_text_field="messages",
        max_seq_length=4096,
        tokenizer=tokenizer,
        args=training_args,
    )

    print(f"[TRAIN] Launching SFTTrainer for {arm}...")
    trainer.train()
    trainer.save_model(str(arm_output / "final_adapter"))
    print(f"[SUCCESS] Saved {arm} adapter to {arm_output / 'final_adapter'}\n")


def evaluate_predictions_file(pred_file_path: str):
    """Scores a JSONL predictions file through VerificationCouncil (gcc/g++ + ASan/UBSan)."""
    p_path = Path(pred_file_path)
    if not p_path.exists():
        print(f"[ERROR] Prediction file not found: {p_path}")
        return

    print(f"\n{'='*70}\n[SCORING PREDICTIONS] File: {p_path.name}\n{'='*70}")

    if VerificationCouncil is None:
        print("[ERROR] VerificationCouncil module could not be imported.")
        return

    council = VerificationCouncil()
    records = []
    with open(p_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    n = len(records)
    if n == 0:
        print("[ERROR] Prediction file contains 0 valid JSON rows.")
        return

    syntax_passes = 0
    exec_passes = 0
    council_passes = 0

    for idx, r in enumerate(records, 1):
        answer = r.get("final_answer", "") or r.get("prediction", "") or r.get("content", "")
        split = r.get("split", "arm_embedded_rtl")
        lang = r.get("language", "c" if split == "arm_embedded_rtl" else "cpp")

        blocks = extract_code_blocks(answer) if extract_code_blocks else []
        code = blocks[0][1] if blocks else answer

        syn_ok, comp, diag = council.agent1_compiler_syntax(code, lang)
        exec_ok, mode, exec_diag = council.agent2_functional_exec(code, lang)
        cot_ok, stub, stop, _ = council.agent3_cot_and_sanitize(r)
        inv_ok, _ = council.agent4_domain_invariants(split, code)

        verified = syn_ok and exec_ok and inv_ok

        if syn_ok: syntax_passes += 1
        if exec_ok: exec_passes += 1
        if verified: council_passes += 1

        print(f" - [{idx}/{n}] Task: {r.get('task_id', f'task_{idx}'):<30} | Syntax: {syn_ok} | Exec: {exec_ok} | Council: {verified}")

    print("\n" + "="*50)
    print(f"EVALUATION RESULTS: {p_path.name}")
    print("="*50)
    print(f"Total Evaluated Tasks      : {n}")
    print(f"Compiler Syntax Valid      : {syntax_passes}/{n} ({round(syntax_passes/n*100, 2)}%)")
    print(f"Native Executed (Pass@1)   : {exec_passes}/{n} ({round(exec_passes/n*100, 2)}%)")
    print(f"Full Council Verified      : {council_passes}/{n} ({round(council_passes/n*100, 2)}%)")
    print("="*50 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="3-Arm QLoRA Ablation Harness")
    parser.add_argument("--arm", choices=["arm1_base", "arm2_unverified", "arm3_verified", "all"], default=None)
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output-dir", "--out-dir", dest="output_dir", type=str, default="./qlora_ablation_output")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--eval-predictions", type=str, default=None, help="Path to JSONL predictions to score")
    return parser.parse_args()


def main():
    args = parse_args()

    # If --eval-predictions passed, run scoring mode
    if args.eval_predictions:
        evaluate_predictions_file(args.eval_predictions)
        return

    # If --out-dir specified without --arm, generate the splits
    if args.arm is None:
        try:
            from prepare_qlora_splits import generate_splits
        except ImportError:
            from musespark_harvester.prepare_qlora_splits import generate_splits
        generate_splits()
        return

    if args.arm == "all":
        for arm_name in ["arm1_base", "arm2_unverified", "arm3_verified"]:
            run_training_arm(
                arm=arm_name,
                base_model_id=args.base_model,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                max_samples=args.max_samples,
            )
    else:
        run_training_arm(
            arm=args.arm,
            base_model_id=args.base_model,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_samples=args.max_samples,
        )


if __name__ == "__main__":
    main()
