"""Robust number parsing for values that an LLM may return as text ("1.2M", "$3,400", "23%")."""
from __future__ import annotations

import math
import re

_SUFFIX = {"": 1.0, "k": 1e3, "m": 1e6, "mm": 1e6, "mn": 1e6, "b": 1e9, "bn": 1e9, "t": 1e12}
_NUM = re.compile(r"^([-+]?(?:\d+\.?\d*|\.\d+))([a-z]{0,2})$")


def parse_number(value) -> float:
    """Convert numbers / numeric strings to float. Percent strings become fractions."""
    if isinstance(value, bool):
        raise ValueError("boolean is not a number")
    if isinstance(value, (int, float)):
        v = float(value)
        if not math.isfinite(v):
            raise ValueError("non-finite number")
        return v
    s = str(value).strip().lower()
    if not s:
        raise ValueError("empty value")
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace(",", "").replace(" ", "")
    percent = s.endswith("%")
    s = s.rstrip("%")
    s = re.sub(r"^[^\d.\-+]+", "", s)  # leading currency symbols / words
    m = _NUM.match(s)
    if not m or m.group(2) not in _SUFFIX:
        raise ValueError(f"cannot parse number from {value!r}")
    out = float(m.group(1)) * _SUFFIX[m.group(2)]
    if percent:
        out /= 100.0
    return -out if negative else out
