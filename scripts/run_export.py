"""Export retrieval results to JSON and CSV formats."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: str) -> list[dict[str, Any]]:
      """Load records from a JSONL file."""
      records: list[dict[str, Any]] = []
      with open(path, encoding="utf-8") as fh:
                for line in fh:
                              line = line.strip()
                              if line:
                                                records.append(json.loads(line))
                                    return records


def export_json(records: list[dict[str, Any]], output_path: str) -> None:
      """Write records to a JSON file."""
      Path(output_path).parent.mkdir(parents=True, exist_ok=True)
      with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(records, fh, indent=2, ensure_ascii=False)
            print(f"Exported {len(records)} records to {output_path}")


def export_csv(records: list[dict[str, Any]], output_path: str) -> None:
      """Write records to a CSV file."""
    if not records:
              print("No records to export.")
              return
          Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
              writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
              writer.writeheader()
              writer.writerows(records)
          print(f"Exported {len(records)} records to {output_path}")


def main() -> None:
      parser = argparse.ArgumentParser(description="Export retrieval results")
    parser.add_argument("input", help="Path to input JSONL file")
    parser.add_argument("output", help="Output path (.json or .csv)")
    args = parser.parse_args()

    records = load_jsonl(args.input)
    if args.output.endswith(".csv"):
              export_csv(records, args.output)
else:
        export_json(records, args.output)


if __name__ == "__main__":
      main()
