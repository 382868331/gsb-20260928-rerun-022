from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def content_id(prefix: str, value: Any) -> str:
    digest = hashlib.sha256(canonical_json(value)).hexdigest()
    return f"{prefix}_{digest}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
