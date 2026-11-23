"""Benchmark Dash2Data.

Offline mode (default, no API key):
    python -m benchmark.run_benchmark [--seeds 5] [--rows 50]
  Runs every benchmark/cases/*.spec.json through generation + calibration + validation and reports
  how closely the generated data reproduces the dashboard numbers.

Online mode (measures the LLM extraction too, needs GEMINI_API_KEY):
    python -m benchmark.run_benchmark --images path/to/folder
  Folder layout:  name.png  +  name.truth.json  where truth is a hand-transcribed list of
  {"measure": "Total Sales", "filters": [{"column": "Region", "value": "East"}], "value": 400000}.
  Reports target recall (how many visible numbers were extracted) and value accuracy of extraction.
"""
from __future__ import annotations

import argparse
import glob
import json
import mimetypes
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.pipeline import build_dataset  # noqa: E402
from core.schemas import load_spec  # noqa: E402

HERE = os.path.dirname(__file__)


def run_offline(rows: int, seeds: int) -> list:
    results = []
    for path in sorted(glob.glob(os.path.join(HERE, "cases", "*.spec.json"))):
        name = os.path.basename(path).replace(".spec.json", "")
        with open(path, encoding="utf-8") as f:
            spec, _ = load_spec(json.load(f))
        means, maxes, rates, times, n_rows = [], [], [], [], 0
        for s in range(seeds):
            t = time.time()
            r = build_dataset(spec, num_rows=rows, seed=1000 + s)
            times.append(time.time() - t)
            means.append(r.summary["mean_error_pct"])
            maxes.append(r.summary["max_error_pct"])
            rates.append(r.summary["match_rate"])
            n_rows = r.n_rows
        results.append({
            "case": name, "targets": r.summary["targets"], "rows": n_rows,
            "mean_err": statistics.mean(means), "max_err": max(maxes),
            "match_rate": statistics.mean(rates), "sec": statistics.mean(times),
        })
    return results


def _norm_key(measure: str, filters) -> tuple:
    return (measure.lower(), tuple(sorted((f["column"].lower(), str(f["value"]).lower()) for f in filters or [])))


def run_online(folder: str) -> list:
    from core.extractor import extract_spec
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        sys.exit("Set GEMINI_API_KEY to run the online benchmark.")
    results = []
    for img in sorted(glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jp*g"))):
        base = os.path.splitext(img)[0]
        truth_path = base + ".truth.json"
        if not os.path.exists(truth_path):
            continue
        truth = json.load(open(truth_path, encoding="utf-8"))
        mime = mimetypes.guess_type(img)[0] or "image/png"
        t = time.time()
        spec, _, _ = extract_spec([(open(img, "rb").read(), mime)], key)
        t_llm = time.time() - t
        got = {}
        for tg in spec.targets:
            got[_norm_key(tg.measure, [{"column": f.column, "value": f.value} for f in tg.filters])] = tg.value
        found, accurate = 0, 0
        for item in truth:
            k = _norm_key(item["measure"], item.get("filters"))
            if k in got:
                found += 1
                if abs(got[k] - item["value"]) <= 0.02 * abs(item["value"] or 1):
                    accurate += 1
        r = build_dataset(spec, num_rows=50, seed=1)
        results.append({
            "case": os.path.basename(base), "truth": len(truth), "recall": found / len(truth),
            "value_acc": accurate / len(truth), "gen_mean_err": r.summary["mean_error_pct"], "llm_sec": t_llm,
        })
    return results


def to_markdown_offline(res: list, rows: int, seeds: int) -> str:
    lines = [f"# Benchmark (offline calibration) - {seeds} seeds per case, requested rows = {rows}", "",
             "| Case | Targets | Rows | Mean error % | Worst error % | Targets matched (<=0.5%) | Time (s) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in res:
        lines.append(f"| {r['case']} | {r['targets']} | {r['rows']:,} | {r['mean_err']:.5f} | "
                     f"{r['max_err']:.5f} | {r['match_rate'] * 100:.1f}% | {r['sec']:.2f} |")
    return "\n".join(lines) + "\n"


def to_markdown_online(res: list) -> str:
    lines = ["# Benchmark (online: extraction + generation)", "",
             "| Dashboard | Truth values | Recall | Value accuracy (2%) | Generated mean error % | LLM time (s) |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in res:
        lines.append(f"| {r['case']} | {r['truth']} | {r['recall'] * 100:.0f}% | {r['value_acc'] * 100:.0f}% | "
                     f"{r['gen_mean_err']:.4f} | {r['llm_sec']:.1f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--images", help="folder with screenshots + .truth.json (online mode)")
    ap.add_argument("--out", default=os.path.join(HERE, "results.md"))
    a = ap.parse_args()
    if a.images:
        md = to_markdown_online(run_online(a.images))
    else:
        md = to_markdown_offline(run_offline(a.rows, a.seeds), a.rows, a.seeds)
    print(md)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    main()
