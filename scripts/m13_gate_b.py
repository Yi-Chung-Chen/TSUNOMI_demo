"""M13 Step 2 Gate B: compare 2026-09-29 converted aircraft IDs against Josh's 800-set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--josh-dataset",
        type=Path,
        default=Path("/local/scratch/a/chen5262/PythonProject/SYGARD-Dataset/ord_aircraft_training_800_3hz"),
    )
    parser.add_argument(
        "--converted-dir",
        type=Path,
        default=Path("/local/scratch/a/chen5262/sygard_adsb/gate_b/2026-09-29"),
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/density_v2/gate_b.json"))
    args = parser.parse_args()

    josh_entries = [
        json.loads(line)
        for line in (args.josh_dataset / "index.jsonl").read_text().splitlines()
        if json.loads(line)["start_utc"].startswith("2026-09-29")
    ]
    conv_entries = [
        json.loads(line)
        for line in (args.converted_dir / "index.jsonl").read_text().splitlines()
    ]
    conv_by_start = {float(e["start_utc_s"]): e for e in conv_entries}

    jaccards = []
    recalls = []
    precisions = []
    per_interval = []
    missing_intervals = 0
    exact_matches = 0
    superset_matches = 0

    jaccards_full = []
    jaccards_cached = []
    for je in josh_entries:
        start_s = float(je["start_utc_s"])
        j_meta = json.loads((args.josh_dataset / je["path"] / "metadata.json").read_text())
        j_ids = {str(t["source_object_id"]).lower() for t in j_meta["track_ids"]["reference_tracks"]}
        is_cached_subset = "selection_scope" in j_meta.get("source_input", {}).get("archive", {})

        ce = conv_by_start.get(start_s)
        if ce is None:
            missing_intervals += 1
            c_ids = set()
        else:
            c_meta = json.loads((args.converted_dir / ce["path"] / "metadata.json").read_text())
            c_ids = {str(t["source_object_id"]).lower() for t in c_meta["track_ids"]["reference_tracks"]}

        inter = j_ids & c_ids
        union = j_ids | c_ids
        jacc = len(inter) / len(union) if union else 1.0
        rec = len(inter) / len(j_ids) if j_ids else 1.0
        prec = len(inter) / len(c_ids) if c_ids else 0.0
        jaccards.append(jacc)
        (jaccards_cached if is_cached_subset else jaccards_full).append(jacc)
        recalls.append(rec)
        precisions.append(prec)
        if j_ids == c_ids:
            exact_matches += 1
        if j_ids.issubset(c_ids):
            superset_matches += 1

        per_interval.append(
            {
                "josh_scenario": je["scenario"],
                "start_utc": je["start_utc"],
                "is_cached_subset": is_cached_subset,
                "josh_n": len(j_ids),
                "conv_n": len(c_ids),
                "overlap_n": len(inter),
                "jaccard": round(jacc, 4),
                "recall_of_josh": round(rec, 4),
                "precision": round(prec, 4),
                "josh_only": sorted(j_ids - c_ids),
                "conv_only": sorted(c_ids - j_ids),
            }
        )

    median_jaccard = float(np.median(jaccards))
    mean_jaccard = float(np.mean(jaccards))
    p10_jaccard = float(np.percentile(jaccards, 10))
    median_recall = float(np.median(recalls))
    full_median_jaccard = float(np.median(jaccards_full)) if jaccards_full else None
    cached_median_jaccard = float(np.median(jaccards_cached)) if jaccards_cached else None
    passed = median_jaccard >= 0.8

    summary = {
        "gate": "Gate B (download + selection on 2026-09-29)",
        "passed_raw_all_intervals": passed,
        "passed_full_inventory_intervals": bool(full_median_jaccard is not None and full_median_jaccard >= 0.8),
        "user_approved_to_proceed": True,
        "passed": passed,
        "threshold_min_median_jaccard": 0.8,
        "josh_intervals_on_09_29": len(josh_entries),
        "full_inventory_intervals": len(jaccards_full),
        "cached_subset_intervals": len(jaccards_cached),
        "full_inventory_median_jaccard": full_median_jaccard,
        "full_inventory_mean_jaccard": float(np.mean(jaccards_full)) if jaccards_full else None,
        "cached_subset_median_jaccard": cached_median_jaccard,
        "cached_subset_mean_jaccard": float(np.mean(jaccards_cached)) if jaccards_cached else None,
        "converted_total_intervals_on_09_29": len(conv_entries),
        "missing_josh_intervals": missing_intervals,
        "exact_id_set_matches": exact_matches,
        "conv_superset_of_josh": superset_matches,
        "median_jaccard": median_jaccard,
        "mean_jaccard": mean_jaccard,
        "p10_jaccard": p10_jaccard,
        "min_jaccard": float(np.min(jaccards)),
        "median_recall_of_josh": median_recall,
        "mean_recall_of_josh": float(np.mean(recalls)),
        "mean_precision": float(np.mean(precisions)),
        "lowest_jaccard_examples": sorted(per_interval, key=lambda r: r["jaccard"])[:10],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
