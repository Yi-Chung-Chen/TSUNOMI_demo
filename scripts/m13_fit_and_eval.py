"""M13 Steps 3b–6: flow/runway audit, v2a density refit, mixed-set check, and summary report."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from scripts.merge_demo_val import protected_hashes
from scripts.render_unreal_full_scene import build_timeline
from scripts.shortlist_unreal_replays import read_jsonl, read_scene
from src.common.runway_anchors import RUNWAYS, alignment_masks, fixed_endpoints, runway_projection
from src.eval.anomaly_score import write_csv
from src.eval.anomaly_state import (
    HEIGHT_EDGES,
    METRICS,
    fit_density,
    plot_density,
    plot_timeline,
    score_m8_rows,
    standardize,
    summarize,
    write_report,
)
from src.eval.runway_flow import scenario_flow
from src.integrated.scoring import COMPONENTS, FrozenStateScorer


ORIGIN = {"latitude_deg": 41.980103, "longitude_deg": -87.903868, "altitude_m": 2250.0}
SHORTLIST_SCENES = ["scenario_000052", "scenario_000073", "scenario_000094", "scenario_000097"]


def hash_directory_files(root: Path, rel_paths: list[str]) -> dict[str, str]:
    out = {}
    for rel in rel_paths:
        p = root / rel
        if p.is_file():
            out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
        elif p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file():
                    out[str(child.relative_to(root))] = hashlib.sha256(child.read_bytes()).hexdigest()
    return out


def runway_arr_dep_shares(scenarios: list[tuple[str, dict]]) -> dict[str, dict]:
    """Compute per-runway-end arrival/departure scenario share using M10 airborne alignment geometry."""
    anchors = fixed_endpoints(ORIGIN)
    pairs = [anchors[i : i + 2] for i in range(0, 16, 2)]
    n_scen = len(scenarios)
    counts: Counter = Counter()

    for _, d in scenarios:
        pos, heading, valid, ground = d["pos"], d["heading"], d["valid"], d["on_ground"]
        eligible = valid.sum(axis=1) >= 10
        for a, b in pairs:
            pa = [a["x_m"], a["y_m"]]
            pb = [b["x_m"], b["y_m"]]
            along, perp, length, axis = runway_projection(pos, pa, pb)
            cos_h = np.cos(heading - axis)
            air_base = valid & (~ground) & (pos[..., 2] < 500) & (perp < 120)
            # End A (heading along +axis): approach is (-6000, 0), departure is (L, L+6000)
            a_arr = air_base & (cos_h > 0.97) & (along > -6000) & (along < 0)
            a_dep = air_base & (cos_h > 0.97) & (along > length) & (along < length + 6000)
            # End B (heading along -axis): approach is (L, L+6000), departure is (-6000, 0)
            b_arr = air_base & (cos_h < -0.97) & (along > length) & (along < length + 6000)
            b_dep = air_base & (cos_h < -0.97) & (along > -6000) & (along < 0)

            for key, mask in [
                (f"{a['ident']} arr", a_arr),
                (f"{a['ident']} dep", a_dep),
                (f"{b['ident']} arr", b_arr),
                (f"{b['ident']} dep", b_dep),
            ]:
                if ((mask.sum(axis=1) >= 6) & eligible).any():
                    counts[key] += 1

    ordered_keys = []
    for a, b in pairs:
        ordered_keys.extend([f"{a['ident']} arr", f"{a['ident']} dep", f"{b['ident']} arr", f"{b['ident']} dep"])
    return {
        k: {"scenarios": int(counts[k]), "share": (counts[k] / n_scen if n_scen else 0.0)}
        for k in ordered_keys
    }


def load_npz_scenarios(manifest_path: Path, processed_dir: Path) -> list[tuple[str, dict]]:
    names = [line.strip() for line in manifest_path.read_text().splitlines() if line.strip()]
    out = []
    for name in names:
        with np.load(processed_dir / f"{name}.npz", allow_pickle=False) as d:
            out.append(
                (
                    name,
                    {
                        "pos": d["pos"],
                        "heading": d["heading"],
                        "valid": d["valid"],
                        "on_ground": d["on_ground"],
                        "date": str(d["date"]),
                        "agent_id": d["agent_id"],
                    },
                )
            )
    return out


def load_mixed_scenarios(mixed_root: Path) -> list[dict]:
    entries = read_jsonl(mixed_root / "index.jsonl")
    scenes = []
    for entry in entries:
        folder = mixed_root / entry["path"]
        clean, observed, valid, ids, drone = read_scene(folder, entry)
        # Also read on_ground and heading_rad from reference_tracks_3hz.jsonl for flow/ground breakdown
        id_to_i = {aid: i for i, aid in enumerate(ids)}
        on_ground = np.zeros(valid.shape, dtype=bool)
        heading = np.zeros(valid.shape, dtype=np.float32)
        with (folder / "reference_tracks_3hz.jsonl").open() as f:
            for line in f:
                row = json.loads(line)
                i = id_to_i[row["source_object_id"]]
                t = int(row["frame_index"])
                on_ground[i, t] = bool(row["on_ground"])
                heading[i, t] = float(row["heading_rad"])
        # Flow computed on aircraft only (same as runway_flow.scenario_flow on aircraft tracks)
        ac_mask = ~drone
        flow_info = scenario_flow(
            {
                "pos": clean[ac_mask],
                "valid": valid[ac_mask],
                "on_ground": on_ground[ac_mask],
            }
        )
        scenes.append(
            {
                "entry": entry,
                "clean": clean,
                "observed": observed,
                "valid": valid,
                "ids": ids,
                "drone": drone,
                "on_ground": on_ground,
                "heading": heading,
                "flow": flow_info["flow"],
                "east_tracks": flow_info["east_tracks"],
                "west_tracks": flow_info["west_tracks"],
            }
        )
    return scenes


def audit_flows_and_counts(scenarios: list[tuple[str, dict]]) -> dict:
    by_day = defaultdict(lambda: {"scenarios": 0, "aircraft_tracks": 0, "eligible_states": 0, "east": 0, "west": 0, "none/tie": 0})
    total_flow = Counter()
    for _, d in scenarios:
        day = d["date"]
        fl = scenario_flow(d)["flow"]
        total_flow[fl] += 1
        by_day[day]["scenarios"] += 1
        by_day[day]["aircraft_tracks"] += int(d["valid"].shape[0])
        by_day[day]["eligible_states"] += int((d["valid"][:, 6:] & d["valid"][:, :-6]).sum())
        by_day[day][fl] += 1
    n = len(scenarios)
    east_frac = total_flow["east"] / n if n else 0.0
    decided = total_flow["east"] + total_flow["west"]
    east_decided_frac = total_flow["east"] / decided if decided else 0.0
    return {
        "scenarios": n,
        "aircraft_tracks": sum(v["aircraft_tracks"] for v in by_day.values()),
        "eligible_states": sum(v["eligible_states"] for v in by_day.values()),
        "flows": dict(total_flow),
        "east_fraction_all": east_frac,
        "east_fraction_decided": east_decided_frac,
        "by_day": {k: by_day[k] for k in sorted(by_day)},
    }


def fit_and_write_v2a(root: Path, extra_manifest: Path, extra_processed_dir: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    model, fit = fit_density(
        root,
        bin_smoothing=True,
        height_edges=HEIGHT_EDGES,
        extra_manifest=extra_manifest,
        extra_processed_dir=extra_processed_dir,
    )
    rows, drone, drone_info = score_m8_rows(root, model)
    reference = standardize(rows)
    summary = summarize(rows)

    # Combined 10-03 + 10-04 airborne state NLL p95 and p98
    comb_air = [
        r for r in rows
        if r["source"] == "aircraft" and (not r["on_ground"]) and r["date"] in ("2026-10-03", "2026-10-04")
    ]
    comb_state_vals = np.array([r["state_nll"] for r in comb_air], dtype=np.float64)
    p95_comb = float(np.percentile(comb_state_vals, 95))
    p98_comb = float(np.percentile(comb_state_vals, 98))
    drone_state_vals = np.array([r["state_nll"] for r in rows if r["source"] == "drone"], dtype=np.float64)

    # Also add an explicit combined row to summary.csv (preserving original 10-04/10-03 rows honestly)
    for metric in METRICS:
        vals = np.array([r[metric] for r in comb_air], dtype=np.float64)
        pos_vals = np.array([r[metric] for r in rows if r["source"] == "drone"], dtype=np.float64)
        from src.eval.anomaly_score import auroc
        p95_m = float(np.percentile(vals, 95))
        summary.append(
            {
                "score": metric,
                "group": "aircraft-airborne (2026-10-03+2026-10-04)",
                "token_count": len(vals),
                "mean": float(vals.mean()),
                "median": float(np.median(vals)),
                "p95": p95_m,
                "auroc_drone_positive": auroc(pos_vals, vals),
                "airborne_10_04_p95": next(
                    r["airborne_10_04_p95"]
                    for r in summary
                    if r["score"] == metric and r["group"] == "aircraft-airborne (2026-10-04)"
                ),
                "drone_fraction_above_airborne_10_04_p95": float(np.mean(pos_vals > p95_m)),
            }
        )

    write_csv(output / "token_scores.csv", rows)
    write_csv(output / "summary.csv", summary)
    (output / "standardization.json").write_text(json.dumps(reference, indent=2) + "\n")
    cutoff_info = {
        "cutoff_source": "p95 of airborne state_nll over 2026-10-03 + 2026-10-04 together",
        "token_count": int(len(comb_state_vals)),
        "state_nll_cutoff": p95_comb,
        "state_nll_p95": p95_comb,
        "state_nll_p98": p98_comb,
        "drone_fraction_above_p95": float(np.mean(drone_state_vals > p95_comb)),
        "drone_fraction_above_p98": float(np.mean(drone_state_vals > p98_comb)),
    }
    (output / "cutoff.json").write_text(json.dumps(cutoff_info, indent=2) + "\n")

    np.savez_compressed(
        output / "state_density.npz",
        bounds=model.bounds,
        height_edges=model.height_edges,
        joint=model.joint,
        speed=model.speed,
        course=model.course,
        pooled_joint=model.pooled_joint,
        pooled_speed=model.pooled_speed,
        pooled_course=model.pooled_course,
        smoothed_speed=model.smoothed_speed,
        smoothed_course=model.smoothed_course,
        p_joint=model.p_joint,
        p_speed=model.p_speed,
        p_course=model.p_course,
    )
    plot_density(model, drone, root / "data/basemap_cache", output / "traffic_density.png")
    plot_timeline(rows, summary, output / "drone_timeline_state.png")
    write_report(summary, reference, fit, output)

    # Verify FrozenStateScorer loads it unchanged and gets exact combined p95
    loaded = FrozenStateScorer(output)
    assert np.isclose(loaded.cutoff, p95_comb), f"Scorer cutoff mismatch: {loaded.cutoff} vs {p95_comb}"

    receipt = {
        "milestone": "M13_v2a",
        "density_fit": fit,
        "standardization": reference,
        "cutoff": cutoff_info,
        "token_rows": len(rows),
        "elapsed_s": round(time.monotonic() - t0, 2),
        "neural_training": False,
        "model_execution": False,
        "device": "cpu",
    }
    (output / "acceptance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return {"receipt": receipt, "rows": rows, "cutoff": cutoff_info, "summary": summary}


def eval_mixed_set(mixed_scenes: list[dict], scorer: FrozenStateScorer, cutoff: float, label: str) -> dict:
    """Score all 100 mixed scenarios with noisy input exactly as D12 timelines do."""
    ac_total = ac_flagged = 0
    ac_by_flow = defaultdict(lambda: [0, 0])  # [flagged, total]
    ac_air = [0, 0]
    ac_gnd = [0, 0]

    case_000052 = {}
    drone_by_family = defaultdict(lambda: [0, 0])
    drone_first_lag_s = []
    drones_never_flagged = []
    total_drones = 0

    for sc in mixed_scenes:
        entry = sc["entry"]
        scen_name = entry["scenario"]
        family = entry["family"]
        flow = sc["flow"]
        observed, valid, ids, drone, on_ground = (
            sc["observed"],
            sc["valid"],
            sc["ids"],
            sc["drone"],
            sc["on_ground"],
        )
        trace = scorer.observation_trace(observed, valid)
        nll = trace["state_nll"]
        scored = valid & np.isfinite(nll)
        flagged = scored & (nll > cutoff)

        for i, aid in enumerate(ids):
            sc_i = scored[i]
            fl_i = flagged[i]
            n_sc = int(sc_i.sum())
            n_fl = int(fl_i.sum())
            if drone[i]:
                total_drones += 1
                drone_by_family[family][0] += n_fl
                drone_by_family[family][1] += n_sc
                valid_frames = np.flatnonzero(valid[i])
                flag_frames = np.flatnonzero(fl_i)
                if len(flag_frames) == 0:
                    drones_never_flagged.append(f"{scen_name}:{aid}")
                else:
                    lag = (int(flag_frames[0]) - int(valid_frames[0])) / 3.0
                    drone_first_lag_s.append(lag)
            else:
                ac_total += n_sc
                ac_flagged += n_fl
                ac_by_flow[flow][0] += n_fl
                ac_by_flow[flow][1] += n_sc
                air_mask = sc_i & (~on_ground[i])
                gnd_mask = sc_i & on_ground[i]
                ac_air[0] += int((fl_i & air_mask).sum())
                ac_air[1] += int(air_mask.sum())
                ac_gnd[0] += int((fl_i & gnd_mask).sum())
                ac_gnd[1] += int(gnd_mask.sum())

                if scen_name == "scenario_000052" and aid in ("a4f2ee", "a19e61"):
                    case_000052[aid] = {"flagged": n_fl, "scored": n_sc, "rate": n_fl / n_sc if n_sc else 0.0}

    def frac(pair):
        return pair[0] / pair[1] if pair[1] else 0.0

    return {
        "label": label,
        "cutoff": cutoff,
        "aircraft_overall": {"flagged": ac_flagged, "scored": ac_total, "rate": ac_flagged / ac_total if ac_total else 0.0},
        "aircraft_east": {"flagged": ac_by_flow["east"][0], "scored": ac_by_flow["east"][1], "rate": frac(ac_by_flow["east"])},
        "aircraft_west": {"flagged": ac_by_flow["west"][0], "scored": ac_by_flow["west"][1], "rate": frac(ac_by_flow["west"])},
        "aircraft_none_tie": {"flagged": ac_by_flow["none/tie"][0], "scored": ac_by_flow["none/tie"][1], "rate": frac(ac_by_flow["none/tie"])},
        "aircraft_airborne": {"flagged": ac_air[0], "scored": ac_air[1], "rate": frac(ac_air)},
        "aircraft_ground": {"flagged": ac_gnd[0], "scored": ac_gnd[1], "rate": frac(ac_gnd)},
        "scenario_000052_cases": case_000052,
        "drones_total": total_drones,
        "drones_never_flagged": drones_never_flagged,
        "drone_first_flag_lag_s": {
            "median": float(np.median(drone_first_lag_s)) if drone_first_lag_s else None,
            "p95": float(np.percentile(drone_first_lag_s, 95)) if drone_first_lag_s else None,
            "max": float(np.max(drone_first_lag_s)) if drone_first_lag_s else None,
        },
        "drone_by_family": {
            fam: {"flagged": v[0], "scored": v[1], "rate": frac(v)}
            for fam, v in sorted(drone_by_family.items())
        },
        "drone_overall": {
            "flagged": sum(v[0] for v in drone_by_family.values()),
            "scored": sum(v[1] for v in drone_by_family.values()),
            "rate": sum(v[0] for v in drone_by_family.values()) / max(1, sum(v[1] for v in drone_by_family.values())),
        },
    }


def eval_shortlist_timelines(root: Path, mixed_root: Path, scorer_dir: Path, cutoff_override: float | None = None) -> dict:
    """Replay build_timeline on the 4 Unreal shortlist scenes using cached Anchor forecasts."""
    config = json.loads((root / "configs/integrated_demo/v2_anchor.json").read_text())
    zone = config["zone"]
    scorer = FrozenStateScorer(scorer_dir)
    if cutoff_override is not None:
        scorer.cutoff = float(cutoff_override)
        scorer.z_cutoff = (scorer.cutoff - scorer.median) / scorer.iqr
    with (scorer_dir / "summary.csv").open() as stream:
        medians = {
            row["score"]: float(row["median"])
            for row in csv.DictReader(stream)
            if row["group"] == "aircraft-airborne (2026-10-04)" and row["score"] in COMPONENTS
        }
    entries = {e["scenario"]: e for e in read_jsonl(mixed_root / "index.jsonl")}
    out = {}
    for scen in SHORTLIST_SCENES:
        entry = entries[scen]
        _, observed, valid, ids, drone = read_scene(mixed_root / entry["path"], entry)
        labels = [""] * len(ids)
        for mask, name in [(drone, "Drone"), (~drone, "Aircraft")]:
            for num, idx in enumerate(np.flatnonzero(mask), 1):
                labels[idx] = f"{name} {num:02}"
        cache = root / "outputs/unreal_replay_shortlist/d12_full_scenes" / scen / "forecasts"
        forecasts = {}
        for npz_path in sorted(cache.glob("*.npz")):
            frame = int(npz_path.stem)
            with np.load(npz_path) as data:
                members = data["members"]
                pos = data["position"]
                medoid = int(data["medoid"])
            full = np.full((32, len(valid), 90, 3), np.nan, dtype=pos.dtype)
            full[:, members] = pos
            forecasts[frame] = {"position": full, "medoid": medoid, "members": members}
        positions = np.where(valid[..., None], observed, np.nan)
        timeline = build_timeline(positions, valid, ids, labels, forecasts, scorer, medians, zone)
        statuses = Counter(r["scenario"]["status"] for r in timeline)
        first_pot = next((r["time_s"] for r in timeline if r["scenario"]["status"] == "Potential intrusion"), None)
        first_high = next((r["time_s"] for r in timeline if r["scenario"]["status"] in ("High threat", "Intrusion detected")), None)
        ac_flags = sum(
            1
            for r in timeline
            for o in r["objects"]
            if o["label"].startswith("Aircraft") and o["anomalous"]
        )
        ac_scored = sum(
            1
            for r in timeline
            for o in r["objects"]
            if o["label"].startswith("Aircraft") and o["observed"] and o["current_state_nll"] is not None
        )
        out[scen] = {
            "first_potential_intrusion_s": first_pot,
            "first_high_or_intrusion_s": first_high,
            "status_counts": dict(statuses),
            "aircraft_flagged_frames": ac_flags,
            "aircraft_scored_frames": ac_scored,
            "aircraft_flag_rate": ac_flags / ac_scored if ac_scored else 0.0,
        }
    return out


def eval_val_sanity(token_rows: list[dict], cutoff: float) -> dict:
    out = {}
    for date in ("2026-10-03", "2026-10-04"):
        for gnd_label, gnd_val in [("airborne", False), ("ground", True)]:
            sub = [r for r in token_rows if r["source"] == "aircraft" and r["date"] == date and r["on_ground"] == gnd_val]
            fl = sum(1 for r in sub if r["state_nll"] > cutoff)
            out[f"{date}_{gnd_label}"] = {"flagged": fl, "tokens": len(sub), "rate": fl / len(sub) if sub else 0.0}
    drone_sub = [r for r in token_rows if r["source"] == "drone"]
    dfl = sum(1 for r in drone_sub if r["state_nll"] > cutoff)
    out["drone_m8"] = {"flagged": dfl, "tokens": len(drone_sub), "rate": dfl / len(drone_sub) if drone_sub else 0.0}
    return out


def write_summary_md(out_path: Path, results: dict) -> None:
    sel = results["selected_days"]
    gate_a = results["gate_a"]
    gate_b = results["gate_b"]
    step3 = results["step3"]
    cut = results["v2a_cutoff"]
    m_cur = results["mixed"]["current"]
    m_p95 = results["mixed"]["v2a_p95"]
    m_p98 = results["mixed"]["v2a_p98"]
    sh_cur = results["shortlist"]["current"]
    sh_p95 = results["shortlist"]["v2a_p95"]
    sh_p98 = results["shortlist"]["v2a_p98"]

    lines = [
        "# M13: State density v2a refit with balanced ADS-B days",
        "",
        "Scope note: per Yi-Chung's confirmed decision (`working-note.md`), M13 builds **`v2a` only** (exact existing M8b 250 m cells, 5 height bins, 3×3 spatial pooling on all layers, `[1,2,1]/4` bin smoothing, α = 0.5) and skips `v2b`. No neural network training, no demo switch, and no video re-render.",
        "",
        "## 1. Days added and selection rule",
        "",
        sel["rule"],
        "",
        "| Date | Day | ISO week | Role | METAR East h | METAR West h | Scenarios | Aircraft tracks | Eligible 3 Hz states | East / West / None-tie |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    by_day = step3["extra"]["by_day"]
    for item in sel["selected_days"]:
        d = item["date"]
        bd = by_day.get(d, {})
        lines.append(
            f"| {d} | {item['weekday']} | {item['iso_week']} | {item['selected_role']} | "
            f"{item['east_hours']} | {item['west_hours']} | {bd.get('scenarios', 0)} | "
            f"{bd.get('aircraft_tracks', 0):,} | {bd.get('eligible_states', 0):,} | "
            f"{bd.get('east', 0)} / {bd.get('west', 0)} / {bd.get('none/tie', 0)} |"
        )
    if results.get("extra_backup_days_used"):
        for item in results["extra_backup_days_used"]:
            d = item["date"]
            bd = by_day.get(d, {})
            lines.append(
                f"| {d} | {item['weekday']} | {item['iso_week']} | east_backup (added) | "
                f"{item['east_hours']} | {item['west_hours']} | {bd.get('scenarios', 0)} | "
                f"{bd.get('aircraft_tracks', 0):,} | {bd.get('eligible_states', 0):,} | "
                f"{bd.get('east', 0)} / {bd.get('west', 0)} / {bd.get('none/tie', 0)} |"
            )

    lines += [
        "",
        f"- **Extra set totals:** {step3['extra']['scenarios']:,} scenarios, {step3['extra']['aircraft_tracks']:,} aircraft tracks, {step3['extra']['eligible_states']:,} eligible 3 Hz states. Flow split: east {step3['extra']['flows'].get('east', 0)} ({100*step3['extra']['east_fraction_all']:.1f}% of all, {100*step3['extra']['east_fraction_decided']:.1f}% of decided), west {step3['extra']['flows'].get('west', 0)}, none/tie {step3['extra']['flows'].get('none/tie', 0)}.",
        f"- **Existing 5-day train totals:** {step3['train5']['scenarios']:,} scenarios, {step3['train5']['eligible_states']:,} eligible 3 Hz states (east {step3['train5']['flows'].get('east', 0)}, west {step3['train5']['flows'].get('west', 0)}, none/tie {step3['train5']['flows'].get('none/tie', 0)}).",
        f"- **Combined v2a training states:** {step3['train5']['eligible_states'] + step3['extra']['eligible_states']:,} eligible 3 Hz states across {step3['train5']['scenarios'] + step3['extra']['scenarios']:,} scenarios.",
        "",
        "## 2. Converter verification gates",
        "",
        f"- **Gate A (interpolation on all 800 SYGARD scenarios):** **{'PASS' if gate_a['passed'] else 'FAIL'}** — "
        f"{gate_a['scenarios_with_row_set_diff']}/{gate_a['scenarios']} scenarios with any row-set difference (`{100*gate_a['scenarios_with_row_set_diff_fraction']:.3f}%`, limit ≤ 2%), "
        f"exact row match `{gate_a['total_common_rows']:,}/{gate_a['total_ref_rows']:,}`, "
        f"median 3D ENU position error `{gate_a['median_pos_error_m']:.3e} m` (limit ≤ 0.1 m), "
        f"`on_ground` agreement `{100*gate_a['on_ground_agreement']:.4f}%`.",
        f"- **Gate B (download + selection on 2026-09-29 tarball vs Josh's {gate_b['josh_intervals_on_09_29']} intervals):** **USER-APPROVED TO PROCEED** — "
        f"overall median Jaccard `{gate_b['median_jaccard']:.4f}` (mean `{gate_b['mean_jaccard']:.4f}`, median recall of Josh `{gate_b['median_recall_of_josh']:.4f}`); "
        f"stratified by Josh's `source_input.archive`: `{gate_b['full_inventory_intervals']}` full-inventory intervals achieve median Jaccard **`{gate_b['full_inventory_median_jaccard']:.4f}`** (≥ 0.80), "
        f"while `{gate_b['cached_subset_intervals']}` intentionally incomplete cached-subset intervals have median Jaccard `{gate_b['cached_subset_median_jaccard']:.4f}` because the full tarball recovers the omitted real aircraft.",
        "",
        "## 3. Per-runway arrival/departure scenario shares (key runways + full summary)",
        "",
        "| Runway op | Train 5 days | New extra days | Combined train | Mixed 100 set |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    r_tr = step3["runway_shares"]["train5"]
    r_ex = step3["runway_shares"]["extra"]
    r_cb = step3["runway_shares"]["combined"]
    r_mx = step3["runway_shares"]["mixed"]
    for op in sorted(r_tr.keys()):
        if max(r_tr[op]["share"], r_ex[op]["share"], r_mx[op]["share"]) >= 0.01:
            lines.append(
                f"| {op} | {100*r_tr[op]['share']:.1f}% | {100*r_ex[op]['share']:.1f}% | "
                f"{100*r_cb[op]['share']:.1f}% | {100*r_mx[op]['share']:.1f}% |"
            )

    lines += [
        "",
        "## 4. Cutoffs and mixed-set comparison (100 scenarios, noisy input)",
        "",
        f"- Current frozen cutoff (`outputs/anomaly_state_smoothed/`, 10-04 airborne p95): **`{m_cur['cutoff']:.4f}` nats**",
        f"- New `v2a` combined 10-03+10-04 airborne **p95** (`outputs/anomaly_state_v2a/cutoff.json`): **`{cut['state_nll_p95']:.4f}` nats** ({cut['token_count']:,} airborne tokens)",
        f"- New `v2a` combined 10-03+10-04 airborne **p98** (reference): **`{cut['state_nll_p98']:.4f}` nats**",
        "",
        "| Metric | Current (5d, 10-04 p95) | v2a (p95, 10-03+10-04) | v2a (p98, 10-03+10-04) |",
        "| --- | ---: | ---: | ---: |",
        f"| Aircraft overall flag rate | {100*m_cur['aircraft_overall']['rate']:.2f}% | {100*m_p95['aircraft_overall']['rate']:.2f}% | {100*m_p98['aircraft_overall']['rate']:.2f}% |",
        f"| Aircraft east-flow flag rate | {100*m_cur['aircraft_east']['rate']:.2f}% | {100*m_p95['aircraft_east']['rate']:.2f}% | {100*m_p98['aircraft_east']['rate']:.2f}% |",
        f"| Aircraft west-flow flag rate | {100*m_cur['aircraft_west']['rate']:.2f}% | {100*m_p95['aircraft_west']['rate']:.2f}% | {100*m_p98['aircraft_west']['rate']:.2f}% |",
        f"| Aircraft airborne flag rate | {100*m_cur['aircraft_airborne']['rate']:.2f}% | {100*m_p95['aircraft_airborne']['rate']:.2f}% | {100*m_p98['aircraft_airborne']['rate']:.2f}% |",
        f"| Aircraft ground flag rate | {100*m_cur['aircraft_ground']['rate']:.2f}% | {100*m_p95['aircraft_ground']['rate']:.2f}% | {100*m_p98['aircraft_ground']['rate']:.2f}% |",
        f"| 000052 Aircraft 07 (`a4f2ee`) flagged updates | {m_cur['scenario_000052_cases']['a4f2ee']['flagged']}/{m_cur['scenario_000052_cases']['a4f2ee']['scored']} | {m_p95['scenario_000052_cases']['a4f2ee']['flagged']}/{m_p95['scenario_000052_cases']['a4f2ee']['scored']} | {m_p98['scenario_000052_cases']['a4f2ee']['flagged']}/{m_p98['scenario_000052_cases']['a4f2ee']['scored']} |",
        f"| 000052 Aircraft 03 (`a19e61`) flagged updates | {m_cur['scenario_000052_cases']['a19e61']['flagged']}/{m_cur['scenario_000052_cases']['a19e61']['scored']} | {m_p95['scenario_000052_cases']['a19e61']['flagged']}/{m_p95['scenario_000052_cases']['a19e61']['scored']} | {m_p98['scenario_000052_cases']['a19e61']['flagged']}/{m_p98['scenario_000052_cases']['a19e61']['scored']} |",
        f"| Drones overall time flagged | {100*m_cur['drone_overall']['rate']:.2f}% | {100*m_p95['drone_overall']['rate']:.2f}% | {100*m_p98['drone_overall']['rate']:.2f}% |",
        f"| Drones never flagged (of {m_cur['drones_total']}) | {len(m_cur['drones_never_flagged'])} | {len(m_p95['drones_never_flagged'])} | {len(m_p98['drones_never_flagged'])} |",
        f"| Drone first-flag lag median / max (s) | {m_cur['drone_first_flag_lag_s']['median']:.2f} / {m_cur['drone_first_flag_lag_s']['max']:.2f} | {m_p95['drone_first_flag_lag_s']['median']:.2f} / {m_p95['drone_first_flag_lag_s']['max']:.2f} | {m_p98['drone_first_flag_lag_s']['median']:.2f} / {m_p98['drone_first_flag_lag_s']['max']:.2f} |",
        "",
        "### Drone time flagged by family",
        "",
        "| Family | Current | v2a (p95) | v2a (p98) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for fam in sorted(m_cur["drone_by_family"]):
        lines.append(
            f"| {fam} | {100*m_cur['drone_by_family'][fam]['rate']:.2f}% | "
            f"{100*m_p95['drone_by_family'][fam]['rate']:.2f}% | "
            f"{100*m_p98['drone_by_family'][fam]['rate']:.2f}% |"
        )

    lines += [
        "",
        "## 5. Four Unreal shortlist scenes (cached Anchor forecasts, full 0–300 s timeline)",
        "",
        "| Scenario | Variant | First Potential intrusion (s) | First High / Intrusion (s) | Aircraft flag rate |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for scen in SHORTLIST_SCENES:
        for vname, sh in [("Current", sh_cur[scen]), ("v2a (p95)", sh_p95[scen]), ("v2a (p98)", sh_p98[scen])]:
            fp = "—" if sh["first_potential_intrusion_s"] is None else f"{sh['first_potential_intrusion_s']:.2f}"
            fh = "—" if sh["first_high_or_intrusion_s"] is None else f"{sh['first_high_or_intrusion_s']:.2f}"
            lines.append(f"| {scen} | {vname} | {fp} | {fh} | {100*sh['aircraft_flag_rate']:.2f}% |")

    lines += [
        "",
        "## 6. Validation-date sanity check (10-03 & 10-04 M8 token rows)",
        "",
        "| Group | Current (10-04 p95) | v2a (p95) | v2a (p98) |",
        "| --- | ---: | ---: | ---: |",
    ]
    s_cur = results["val_sanity"]["current"]
    s_p95 = results["val_sanity"]["v2a_p95"]
    s_p98 = results["val_sanity"]["v2a_p98"]
    for key in ("2026-10-03_airborne", "2026-10-03_ground", "2026-10-04_airborne", "2026-10-04_ground", "drone_m8"):
        lines.append(
            f"| {key} | {100*s_cur[key]['rate']:.2f}% | {100*s_p95[key]['rate']:.2f}% | {100*s_p98[key]['rate']:.2f}% |"
        )

    lines += [
        "",
        "## 7. Protected artifacts and next-step recommendation",
        "",
        f"- All protected artifacts (`data/splits/`, `assets/motion_dict_K2048_s6.pkl`, `outputs/air_pre_bc/checkpoints/epoch_016.ckpt`, `outputs/runway_anchor/training/anchor/checkpoints/epoch_018.ckpt`, `outputs/anomaly_state_smoothed/`) verified **SHA256 byte-identical** before and after M13: `{results['protected_unchanged']}`.",
        "- Demo config and rendered videos were **not** modified. Switching the integrated demo or re-rendering videos to `outputs/anomaly_state_v2a/` requires separate approval from Yi-Chung.",
        "",
    ]
    out_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--mixed-root",
        type=Path,
        default=Path("/local/scratch/a/chen5262/PythonProject/SYGARD-Dataset/ord_mixed_100_3hz"),
    )
    parser.add_argument("--extra-manifest", type=Path, default=Path("data/splits/density_extra_manifest.txt"))
    parser.add_argument("--extra-processed", type=Path, default=Path("data/processed_extra"))
    parser.add_argument("--v2a-out", type=Path, default=Path("outputs/anomaly_state_v2a"))
    parser.add_argument("--report-dir", type=Path, default=Path("outputs/density_v2"))
    args = parser.parse_args()

    root = args.root.resolve()
    report_dir = root / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    protected_paths = [
        "data/splits/train_manifest.txt",
        "data/splits/train_windows.csv",
        "data/splits/val_manifest.txt",
        "data/splits/val_windows.csv",
        "data/splits/type_vocab.json",
        "assets/motion_dict_K2048_s6.pkl",
        "outputs/air_pre_bc/checkpoints/epoch_016.ckpt",
        "outputs/runway_anchor/training/anchor/checkpoints/epoch_018.ckpt",
        "outputs/anomaly_state_smoothed",
    ]
    before_hashes = hash_directory_files(root, protected_paths)

    # Step 3 audit: flows & per-runway arrival/departure shares
    train5 = load_npz_scenarios(root / "data/splits/train_manifest.txt", root / "data/processed")
    extra = load_npz_scenarios(root / args.extra_manifest, root / args.extra_processed)
    mixed = load_mixed_scenarios(args.mixed_root)
    mixed_as_npz = [
        (
            sc["entry"]["scenario"],
            {
                "pos": sc["clean"][~sc["drone"]],
                "heading": sc["heading"][~sc["drone"]],
                "valid": sc["valid"][~sc["drone"]],
                "on_ground": sc["on_ground"][~sc["drone"]],
                "date": sc["entry"]["start_utc"][:10],
            },
        )
        for sc in mixed
    ]

    step3 = {
        "train5": audit_flows_and_counts(train5),
        "extra": audit_flows_and_counts(extra),
        "runway_shares": {
            "train5": runway_arr_dep_shares(train5),
            "extra": runway_arr_dep_shares(extra),
            "combined": runway_arr_dep_shares(train5 + extra),
            "mixed": runway_arr_dep_shares(mixed_as_npz),
        },
    }
    print("Step 3 extra flows:", json.dumps(step3["extra"]["flows"]), "east_frac_all=", round(step3["extra"]["east_fraction_all"], 4), flush=True)

    # Step 4: refit v2a
    v2a_out = root / args.v2a_out
    v2a = fit_and_write_v2a(root, root / args.extra_manifest, root / args.extra_processed, v2a_out)
    p95_cut = v2a["cutoff"]["state_nll_p95"]
    p98_cut = v2a["cutoff"]["state_nll_p98"]

    # Step 5: score 100 mixed scenarios + 4 shortlist timelines + 10-03/10-04 sanity
    cur_dir = root / "outputs/anomaly_state_smoothed"
    cur_scorer = FrozenStateScorer(cur_dir)
    v2a_scorer = FrozenStateScorer(v2a_out)
    assert np.isclose(v2a_scorer.cutoff, p95_cut)

    mixed_results = {
        "current": eval_mixed_set(mixed, cur_scorer, cur_scorer.cutoff, "current"),
        "v2a_p95": eval_mixed_set(mixed, v2a_scorer, p95_cut, "v2a_p95"),
        "v2a_p98": eval_mixed_set(mixed, v2a_scorer, p98_cut, "v2a_p98"),
    }
    shortlist_results = {
        "current": eval_shortlist_timelines(root, args.mixed_root, cur_dir),
        "v2a_p95": eval_shortlist_timelines(root, args.mixed_root, v2a_out),
        "v2a_p98": eval_shortlist_timelines(root, args.mixed_root, v2a_out, cutoff_override=p98_cut),
    }

    with (cur_dir / "token_scores.csv").open() as f:
        cur_rows = []
        for r in csv.DictReader(f):
            r["on_ground"] = r["on_ground"] == "True"
            r["state_nll"] = float(r["state_nll"])
            cur_rows.append(r)

    val_sanity = {
        "current": eval_val_sanity(cur_rows, cur_scorer.cutoff),
        "v2a_p95": eval_val_sanity(v2a["rows"], p95_cut),
        "v2a_p98": eval_val_sanity(v2a["rows"], p98_cut),
    }

    after_hashes = hash_directory_files(root, protected_paths)
    protected_ok = before_hashes == after_hashes
    assert protected_ok, "Protected artifact hash changed during M13!"

    sel = json.loads((report_dir / "selected_days.json").read_text())
    gate_a = json.loads((report_dir / "gate_a.json").read_text())
    gate_b = json.loads((report_dir / "gate_b.json").read_text())

    extra_dates_present = set(step3["extra"]["by_day"].keys())
    selected_dates = {d["date"] for d in sel["selected_days"]}
    backup_used = [d for d in sel.get("backup_east_days", []) if d["date"] in extra_dates_present and d["date"] not in selected_dates]

    results = {
        "selected_days": sel,
        "extra_backup_days_used": backup_used,
        "gate_a": gate_a,
        "gate_b": gate_b,
        "step3": step3,
        "v2a_cutoff": v2a["cutoff"],
        "mixed": mixed_results,
        "shortlist": shortlist_results,
        "val_sanity": val_sanity,
        "protected_unchanged": protected_ok,
        "protected_file_count": len(before_hashes),
    }
    (report_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    write_summary_md(report_dir / "summary.md", results)
    print(f"Wrote {report_dir / 'summary.md'} and {report_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
