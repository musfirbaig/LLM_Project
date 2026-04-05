"""
Validate and split chat fine-tuning data.

Usage:
  python scripts/validate_finetuning_data.py
  python scripts/validate_finetuning_data.py --input data/finetuning_data_chat.jsonl --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_ROLES = ["system", "user", "assistant"]


@dataclass
class RecordStats:
    user_len: int
    assistant_len: int
    combined_len: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate chat-format fine-tuning data")
    parser.add_argument("--input", default="data/finetuning_data_chat.jsonl", help="Input JSONL path")
    parser.add_argument("--out-dir", default="data/splits", help="Directory for split output files")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {idx}: {exc}") from exc
            records.append(obj)
    return records


def _validate_record(obj: dict[str, Any], line_num: int) -> tuple[bool, str, RecordStats | None]:
    msgs = obj.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 3:
        return False, f"line {line_num}: missing/invalid messages list", None

    first_three = msgs[:3]
    roles = [m.get("role") if isinstance(m, dict) else None for m in first_three]
    if roles != REQUIRED_ROLES:
        return False, f"line {line_num}: first three roles must be {REQUIRED_ROLES}, got {roles}", None

    contents: list[str] = []
    for pos, msg in enumerate(first_three, start=1):
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, str) or not content.strip():
            return False, f"line {line_num}: message #{pos} has empty content", None
        contents.append(content.strip())

    stats = RecordStats(
        user_len=len(contents[1]),
        assistant_len=len(contents[2]),
        combined_len=len("\n".join(contents)),
    )
    return True, "", stats


def _fingerprint(obj: dict[str, Any]) -> str:
    msgs = obj.get("messages", [])
    compact = []
    for m in msgs[:3]:
        compact.append({
            "role": m.get("role", "") if isinstance(m, dict) else "",
            "content": (m.get("content", "") if isinstance(m, dict) else "").strip().lower(),
        })
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _print_dist(name: str, values: list[int]) -> None:
    if not values:
        print(f"{name}: no data")
        return
    print(
        f"{name}: min={min(values)} mean={statistics.mean(values):.1f} "
        f"p95={sorted(values)[int(0.95 * (len(values) - 1))]} max={max(values)}"
    )


def main() -> None:
    args = parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    if args.train_ratio <= 0 or args.val_ratio < 0 or (args.train_ratio + args.val_ratio) >= 1:
        raise ValueError("Ratios must satisfy: train_ratio > 0, val_ratio >= 0, and train+val < 1")

    records = _read_jsonl(in_path)
    print(f"Loaded {len(records)} records from {in_path}")

    valid_rows: list[dict[str, Any]] = []
    stats: list[RecordStats] = []
    errors: list[str] = []
    fingerprints: set[str] = set()
    duplicate_count = 0

    for line_num, row in enumerate(records, start=1):
        ok, err, row_stats = _validate_record(row, line_num)
        if not ok:
            errors.append(err)
            continue

        fp = _fingerprint(row)
        if fp in fingerprints:
            duplicate_count += 1
            continue
        fingerprints.add(fp)

        valid_rows.append(row)
        if row_stats is not None:
            stats.append(row_stats)

    print(f"Valid rows: {len(valid_rows)}")
    print(f"Rejected rows: {len(errors)}")
    print(f"Dropped duplicates: {duplicate_count}")

    if errors:
        print("\nSample validation errors:")
        for item in errors[:10]:
            print(f"  - {item}")

    user_lens = [s.user_len for s in stats]
    assistant_lens = [s.assistant_len for s in stats]
    combo_lens = [s.combined_len for s in stats]

    print("\nLength statistics (character-based):")
    _print_dist("user", user_lens)
    _print_dist("assistant", assistant_lens)
    _print_dist("combined", combo_lens)

    if not valid_rows:
        raise RuntimeError("No valid rows left after validation")

    random.seed(args.seed)
    shuffled = valid_rows[:]
    random.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "train.jsonl", train)
    _write_jsonl(out_dir / "val.jsonl", val)
    _write_jsonl(out_dir / "test.jsonl", test)

    print("\nWrote split files:")
    print(f"  - {out_dir / 'train.jsonl'} ({len(train)})")
    print(f"  - {out_dir / 'val.jsonl'} ({len(val)})")
    print(f"  - {out_dir / 'test.jsonl'} ({len(test)})")


if __name__ == "__main__":
    main()
