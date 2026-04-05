"""
QLoRA fine-tuning script for Qwen-style chat models.

Expected input files are chat JSONL records with this shape:
{"messages": [{"role":"system","content":"..."}, {"role":"user","content":"..."}, {"role":"assistant","content":"..."}]}

Example:
  python scripts/train_qlora_qwen.py \
    --model-id Qwen/Qwen2.5-3B-Instruct \
    --train-file data/splits/train.jsonl \
    --val-file data/splits/val.jsonl \
    --output-dir checkpoints/qwen_lora
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Qwen chat models")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct", help="HF model id")
    parser.add_argument("--train-file", default="data/splits/train.jsonl", help="Train JSONL")
    parser.add_argument("--val-file", default="data/splits/val.jsonl", help="Validation JSONL")
    parser.add_argument("--output-dir", default="checkpoints/qwen_lora", help="Output directory")
    parser.add_argument("--max-seq-len", type=int, default=768, help="Max sequence length")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def to_text(messages: list[dict[str, str]], tokenizer: AutoTokenizer) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    # Fallback chat format if chat template is unavailable.
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<{role}>\n{content}\n")
    return "\n".join(parts)


def build_dataset(path: Path, tokenizer: AutoTokenizer, max_seq_len: int) -> Dataset:
    rows = read_jsonl(path)
    texts = [to_text(row["messages"], tokenizer) for row in rows]

    ds = Dataset.from_dict({"text": texts})

    def tok(batch: dict[str, list[str]]) -> dict[str, Any]:
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    return ds.map(tok, batched=True, remove_columns=["text"])


def main() -> None:
    args = parse_args()

    train_path = Path(args.train_file)
    val_path = Path(args.val_file)
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Train/val split files not found. Run validate_finetuning_data.py first.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quant_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = build_dataset(train_path, tokenizer, args.max_seq_len)
    val_ds = build_dataset(val_path, tokenizer, args.max_seq_len)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_args = TrainingArguments(
        output_dir=str(out_path),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
        bf16=False,
        gradient_checkpointing=True,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()

    model.save_pretrained(str(out_path / "adapter"))
    tokenizer.save_pretrained(str(out_path / "adapter"))
    print(f"Saved adapter to {out_path / 'adapter'}")


if __name__ == "__main__":
    main()
