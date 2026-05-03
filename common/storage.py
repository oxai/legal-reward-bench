from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records

def write_jsonl(path: Path | str, records: Iterable[dict[str, Any]]) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False))
            file.write("\n")
            count += 1
    return count

def read_text(path: Path | str) -> str:
    return Path(path).read_text(encoding="utf-8")