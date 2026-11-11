#!/usr/bin/env python3
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


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
VERIFIED_PATH = OUTPUT_DIR / "musespark_systems_verified.jsonl"
BENCHMARK_PATH = OUTPUT_DIR / "musespark_systems_full_benchmark.jsonl"


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


def parse_args():
    parser = argparse.ArgumentParser(description="3-Arm QLoRA Ablation Harness")
    parser.add_argument("--arm", choices=["arm1_base", "arm2_unverified", "arm3_verified", "all"], default="arm3_verified")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output-dir", type=str, default="./qlora_ablation_output")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
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
