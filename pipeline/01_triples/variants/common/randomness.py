from __future__ import annotations

import hashlib


def deterministic_seed(*parts: object) -> int:
    text = "\0".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def deterministic_index(count: int, *parts: object) -> int:
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    return deterministic_seed(*parts) % count
