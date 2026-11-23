"""M13 Step 2 Gate A: verify converter interpolation on all 800 SYGARD scenarios."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
import numpy as np

from src.preprocess.adsb_convert import interpolate_track


def check_one_scenario(args: tuple[Path, dict]) -> dict:
    root, entry = args
    folder = root / entry["path"]
    start_utc_s = float(entry["start_utc_s"])
    duration_s = int(entry["duration_s"])

    obs_by_id: dict[str, list[dict]] = defaultdict(list)
    with (folder / "adsb_observations.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            obs_by_id[str(r["source_object_id"])].append(r)

    ref_by_key: dict[tuple[str, int], dict] = {}
    with (folder / "reference_tracks_3hz.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            ref_by_key[(str(r["source_object_id"]), int(r["frame_index"]))] = r

    gen_by_key: dict[tuple[str, int], dict] = {}
    for aid, obs_list in obs_by_id.items():
        for r in interpolate_track(obs_list, start_utc_s, duration_s=duration_s):
            gen_by_key[(aid, r["frame_index"])] = r

    ref_keys = set(ref_by_key.keys())
    gen_keys = set(gen_by_key.keys())
    common = ref_keys & gen_keys
    extra = gen_keys - ref_keys
    missing = ref_keys - gen_keys

    pos_errs = []
    og_matches = 0
    for k in common:
        gr = gen_by_key[k]
        rr = ref_by_key[k]
        err = float(np.linalg.norm(np.asarray(gr["xyz_enu_m"]) - np.asarray(rr["xyz_enu_m"])))
        pos_errs.append(err)
        if gr["on_ground"] == rr["on_ground"]:
            og_matches += 1

    return {
        "scenario": entry["scenario"],
        "ref_rows": len(ref_keys),
        "gen_rows": len(gen_keys),
        "common_rows": len(common),
        "extra_rows": len(extra),
        "missing_rows": len(missing),
        "exact_row_set": (len(extra) == 0 and len(missing) == 0),
        "og_matches": og_matches,
        "pos_errs": pos_errs,
        "sample_extra": sorted(extra)[:3],
        "sample_missing": sorted(missing)[:3],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/local/scratch/a/chen5262/PythonProject/SYGARD-Dataset/ord_aircraft_training_800_3hz"),
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/density_v2/gate_a.json"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    t0 = time.monotonic()
    entries = [json.loads(line) for line in (args.dataset / "index.jsonl").read_text().splitlines()]
    tasks = [(args.dataset, e) for e in entries]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(check_one_scenario, tasks))

    all_errs = np.concatenate([np.asarray(r["pos_errs"], dtype=np.float64) for r in results if r["pos_errs"]])
    n_scen = len(results)
    n_differ = sum(1 for r in results if not r["exact_row_set"])
    differ_frac = n_differ / n_scen
    total_ref = sum(r["ref_rows"] for r in results)
    total_gen = sum(r["gen_rows"] for r in results)
    total_common = sum(r["common_rows"] for r in results)
    total_extra = sum(r["extra_rows"] for r in results)
    total_missing = sum(r["missing_rows"] for r in results)
    total_og_match = sum(r["og_matches"] for r in results)

    median_err = float(np.median(all_errs))
    p95_err = float(np.percentile(all_errs, 95))
    max_err = float(np.max(all_errs))
    og_agree = total_og_match / total_common if total_common else 0.0

    passed = (differ_frac <= 0.02) and (median_err <= 0.1)
    summary = {
        "gate": "Gate A (interpolation on 800 scenarios)",
        "passed": passed,
        "elapsed_s": round(time.monotonic() - t0, 2),
        "scenarios": n_scen,
        "scenarios_with_row_set_diff": n_differ,
        "scenarios_with_row_set_diff_fraction": differ_frac,
        "total_ref_rows": total_ref,
        "total_gen_rows": total_gen,
        "total_common_rows": total_common,
        "total_extra_rows": total_extra,
        "total_missing_rows": total_missing,
        "median_pos_error_m": median_err,
        "p95_pos_error_m": p95_err,
        "max_pos_error_m": max_err,
        "on_ground_agreement": og_agree,
        "thresholds": {
            "max_scenario_row_diff_fraction": 0.02,
            "max_median_pos_error_m": 0.1,
        },
        "differing_scenarios_sample": [
            {
                "scenario": r["scenario"],
                "extra": r["extra_rows"],
                "missing": r["missing_rows"],
                "sample_extra": r["sample_extra"],
                "sample_missing": r["sample_missing"],
            }
            for r in results
            if not r["exact_row_set"]
        ][:15],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
