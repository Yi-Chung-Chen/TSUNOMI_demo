"""M13 Step 2: Download, SHA256-verify, and convert selected ADSB.lol days."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time

from scripts.m13_convert_day import convert_day_scenarios, extract_ord_traces, verify_sha256


def download_asset(url: str, dest: Path, expected_size: int, expected_digest: str | None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == expected_size:
        try:
            verify_sha256(dest, expected_digest)
            print(json.dumps({"stage": "download_skip_verified", "file": dest.name}), flush=True)
            return dest
        except ValueError:
            dest.unlink()
    tmp = dest.with_suffix(dest.suffix + ".part")
    t0 = time.monotonic()
    subprocess.run(
        ["curl", "-fsSL", "--retry", "3", "--retry-delay", "5", "-o", str(tmp), url],
        check=True,
    )
    verify_sha256(tmp, expected_digest)
    tmp.replace(dest)
    print(
        json.dumps(
            {
                "stage": "downloaded",
                "file": dest.name,
                "gb": round(expected_size / 1e9, 2),
                "elapsed_s": round(time.monotonic() - t0, 1),
            }
        ),
        flush=True,
    )
    return dest


def process_day(day_item: dict, adsb_root: Path, workers: int) -> dict:
    day = day_item["date"]
    rel = day_item["release"]
    out_dir = adsb_root / "scenarios" / day
    idx_path = out_dir / "index.jsonl"
    if idx_path.exists() and idx_path.stat().st_size > 0:
        n_lines = sum(1 for l in idx_path.read_text().splitlines() if l.strip())
        print(json.dumps({"stage": "day_skip_converted", "date": day, "scenarios": n_lines}), flush=True)
        return {"date": day, "scenarios": n_lines, "skipped": True}

    raw_dir = adsb_root / "raw" / day
    assets = rel["assets"]
    with ThreadPoolExecutor(max_workers=min(2, len(assets))) as dl_ex:
        futs = [
            dl_ex.submit(download_asset, a["url"], raw_dir / a["name"], a["size"], a.get("digest"))
            for a in assets
        ]
        part_paths = [f.result() for f in futs]

    cache_path = adsb_root / "ord_traces" / f"ord_traces_{day}.jsonl.gz"
    traces = extract_ord_traces(part_paths, cache_path, workers=workers)
    entries = convert_day_scenarios(day, traces, out_dir, source_tag=rel["tag"], workers=workers)
    return {"date": day, "scenarios": len(entries), "skipped": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-json", type=Path, default=Path("outputs/density_v2/selected_days.json"))
    parser.add_argument("--adsb-root", type=Path, default=Path("/local/scratch/a/chen5262/sygard_adsb"))
    parser.add_argument("--dates", nargs="*", help="Optional subset of YYYY-MM-DD dates (default: all selected_days)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    payload = json.loads(args.selected_json.read_text())
    pool = {d["date"]: d for d in payload["selected_days"] + payload["backup_east_days"]}
    targets = args.dates if args.dates else [d["date"] for d in payload["selected_days"]]

    t0 = time.monotonic()
    summary = []
    for day in targets:
        if day not in pool:
            raise ValueError(f"Unknown date: {day}")
        summary.append(process_day(pool[day], args.adsb_root, workers=args.workers))
    print(
        json.dumps(
            {
                "stage": "fetch_and_convert_done",
                "days": len(summary),
                "total_scenarios": sum(s["scenarios"] for s in summary),
                "elapsed_s": round(time.monotonic() - t0, 1),
                "per_day": summary,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
