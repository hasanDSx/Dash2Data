"""Central configuration for Dash2Data."""
from __future__ import annotations

import os

# Model name is configurable so a wrong/retired name never requires a code change.
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
FALLBACK_MODELS = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "").split(",") if m.strip()]

RETRYABLE_CODES = (429, 500, 503, 504)
MAX_API_RETRIES = 5
MAX_SPEC_REPAIRS = 2

MIN_ROWS = 10
MAX_ROWS = 20_000

# Validation tolerances (relative error)
TOL_MATCH = 0.005   # <= 0.5 %  -> match
TOL_CLOSE = 0.02    # <= 2 %    -> close

# label -> (python strftime, excel number format)
DATE_FORMATS = {
    "YYYY-MM-DD": ("%Y-%m-%d", "yyyy-mm-dd"),
    "DD/MM/YYYY": ("%d/%m/%Y", "dd/mm/yyyy"),
    "MM/DD/YYYY": ("%m/%d/%Y", "mm/dd/yyyy"),
}
