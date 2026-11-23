"""M13 Step 3: Ingest converted extra ADS-B days into data/processed_extra/ and check flow balance."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
import numpy as np

from src.eval.runway_flow import scenario_flow
from src.preprocess.ingest import ingest_scenario, summarize


def _worker(args):
    return str(ingest_scenario(*args))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adsb-root", type=Path, default=Path("/local/scratch/a/chen5262/sygard_adsb"))
    parser.add_argument("--dates", nargs="+", required=True, help="YYYY-MM-DD dates to ingest")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed_extra"))
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/density_extra_manifest.txt"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    t0 = time.monotonic()
    tasks = []
    all_entries = []
    for day in sorted(args.dates):
        day_dir = args.adsb_root / "scenarios" / day
        idx_path = day_dir / "index.jsonl"
        if not idx_path.exists():
            raise FileNotFoundError(f"Missing converted index: {idx_path}")
        for line in idx_path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            all_entries.append(entry)
            tasks.append((day_dir, entry, args.output_dir))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            paths = [Path(p) for p in ex.map(_worker, tasks)]
    else:
        paths = [Path(_worker(t)) for t in tasks]

    stats = summarize(paths, all_entries)
    (args.output_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    names = sorted(e["scenario"] for e in all_entries)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text("\n".join(names) + "\n")

    # Compute flow balance across ingested extra scenarios
    flow_counts = Counter()
    by_day = defaultdict(Counter)
    for p in paths:
        with np.load(p, allow_pickle=False) as d:
            data = {k: d[k] for k in ("pos", "valid", "on_ground", "date")}
        fl = scenario_flow(data)["flow"]
        flow_counts[fl] += 1
        by_day[str(data["date"])][fl] += 1

    total = len(paths)
    east_frac = flow_counts["east"] / total if total else 0.0
    decided = flow_counts["east"] + flow_counts["west"]
    east_decided = flow_counts["east"] / decided if decided else 0.0
    out = {
        "elapsed_s": round(time.monotonic() - t0, 2),
        "scenarios": total,
        "aircraft_tracks": stats["aircraft_tracks"],
        "valid_rows": stats["valid_rows"],
        "flows": dict(flow_counts),
        "east_fraction_all": round(east_frac, 4),
        "east_fraction_decided": round(east_decided, 4),
        "needs_more_east_days": east_frac < 0.40,
        "flows_by_day": {k: dict(by_day[k]) for k in sorted(by_day)},
        "manifest": str(args.manifest),
    }
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
