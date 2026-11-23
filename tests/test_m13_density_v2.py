"""Unit tests for M13 ADS-B converter, extra-manifest density fit, and FrozenStateScorer."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.anomaly_state import fit_density
from src.integrated.scoring import FrozenStateScorer
from src.preprocess.adsb_convert import (
    BOUNDS,
    FALLBACK_ALT_M,
    interpolate_track,
    qualify_aircraft_in_window,
)


def _make_obs(utc_s, lat, lon, alt_m, on_ground=False, speed_mps=70.0, heading_deg=90.0, flags=0):
    return {
        "utc_s": float(utc_s),
        "latitude": float(lat),
        "longitude": float(lon),
        "altitude_m": None if alt_m is None else float(alt_m),
        "altitude_source": "terrain_collision_or_documented_fallback" if alt_m is None else "reported_geometric",
        "on_ground": bool(on_ground),
        "speed_mps": float(speed_mps),
        "heading_deg": float(heading_deg),
        "vertical_speed_mps": 0.0,
        "roll_deg": 0.0,
        "callsign": "TEST1",
        "source": "adsb_icao",
        "flags": int(flags),
        "source_object_id": "abc123",
        "vehicle_type": "aircraft",
        "adsb_reported_type": "B738",
    }


def test_converter_roundtrip_gap_bounds_fallback_and_selection():
    t0 = 1790553600.0
    # 1) Normal segment 0..30s inside ORD bounds (moving east ~2000 m), height 300 m -> qualifies
    obs = []
    for k in range(31):
        lat = 41.980
        lon = -87.920 + k * 0.0008  # ~66 m/s east
        obs.append(_make_obs(t0 + k, lat, lon, alt_m=466.0 + k * 2.0, on_ground=False, speed_mps=66.0))
    # 2) Gap of 40 s (> 30 s) -> frames between t0+30 and t0+70 must be dropped
    obs.append(_make_obs(t0 + 70.0, 41.980, -87.890, alt_m=None, on_ground=True, speed_mps=5.0))
    obs.append(_make_obs(t0 + 75.0, 41.980, -87.889, alt_m=None, on_ground=True, speed_mps=5.0))
    # 3) Point outside bounds at t0+80 -> out-of-bounds interpolated frames dropped
    obs.append(_make_obs(t0 + 80.0, 42.100, -87.889, alt_m=500.0, on_ground=False, speed_mps=70.0))

    rows = interpolate_track(obs, start_utc_s=t0, duration_s=300)
    frames = {r["frame_index"]: r for r in rows}

    # Exact 3 Hz grid in 0..30 s (frames 0..90 inclusive)
    for f in range(0, 91):
        assert f in frames
        assert np.isclose(frames[f]["elapsed_s"], f / 3.0)
        assert not frames[f]["altitude_fallback_used"]

    # Gap > 30 s produces NO frames in (90, 210)
    for f in range(91, 210):
        assert f not in frames

    # Fallback altitude 166.8 m used on ground frames 210..225
    assert 210 in frames and 225 in frames
    assert frames[210]["altitude_fallback_used"] is True
    assert np.isclose(frames[210]["altitude_m"], FALLBACK_ALT_M)
    assert frames[210]["on_ground"] is True

    # Out-of-bounds frames dropped
    assert all(
        BOUNDS["south"] <= r["latitude"] <= BOUNDS["north"]
        and BOUNDS["west"] <= r["longitude"] <= BOUNDS["east"]
        for r in rows
    )

    # Selection qualifies on the 0..30 s airborne segment
    event = qualify_aircraft_in_window(obs, start_utc_s=t0, duration_s=300)
    assert event is not None
    assert event["icao24"] == "abc123"
    assert event["duration_s"] >= 25
    assert event["traveled_m"] >= 1200


def test_fit_density_extra_manifest_preserves_default_and_blocks_excluded_dates(tmp_path):
    processed = tmp_path / "data/processed"
    extra = tmp_path / "data/processed_extra"
    splits = tmp_path / "data/splits"
    for d in (processed, extra, splits):
        d.mkdir(parents=True)

    bounds_json = np.asarray(json.dumps(BOUNDS))
    p = np.zeros((1, 20, 3), dtype=np.float32)
    p[0, :, 0] = np.arange(20) * 10.0
    p[0, :, 1] = 200.0
    p[0, :, 2] = 150.0
    valid = np.ones((1, 20), dtype=bool)

    np.savez(processed / "s1.npz", pos=p, valid=valid, date=np.asarray("2026-09-28"), bounds_json=bounds_json)
    np.savez(extra / "e1.npz", pos=p + 500.0, valid=valid, date=np.asarray("2026-09-06"), bounds_json=bounds_json)
    np.savez(extra / "bad.npz", pos=p, valid=valid, date=np.asarray("2026-10-04"), bounds_json=bounds_json)

    (splits / "train_manifest.txt").write_text("s1\n")
    (splits / "extra_ok.txt").write_text("e1\n")
    (splits / "extra_bad.txt").write_text("bad\n")

    m_base, r_base = fit_density(tmp_path)
    m_base2, _ = fit_density(tmp_path, extra_manifest=None)
    np.testing.assert_array_equal(m_base.p_joint, m_base2.p_joint)
    assert r_base["raw_eligible_states"] == 14

    m_ext, r_ext = fit_density(tmp_path, extra_manifest=splits / "extra_ok.txt", extra_processed_dir=extra)
    assert r_ext["raw_eligible_states"] == 28
    assert r_ext["extra_scenarios"] == 1
    assert r_ext["extra_dates"] == {"2026-09-06": 1}
    assert np.isclose(m_ext.p_joint.sum(), 1.0)

    with pytest.raises(ValueError, match="Excluded date"):
        fit_density(tmp_path, extra_manifest=splits / "extra_bad.txt", extra_processed_dir=extra)


def test_frozen_state_scorer_backward_compatible_cutoff(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    legacy_dir = repo_root / "outputs/anomaly_state_smoothed"
    if not (legacy_dir / "state_density.npz").exists():
        pytest.skip("Local outputs/anomaly_state_smoothed not present")

    legacy = FrozenStateScorer(legacy_dir)
    assert np.isclose(legacy.cutoff, 11.387990034013754)

    # Copy minimal layout into tmp_path and add cutoff.json for v2 combined cutoff
    v2_dir = tmp_path / "anomaly_state_v2a"
    v2_dir.mkdir()
    for fname in ("state_density.npz", "standardization.json", "summary.csv"):
        (v2_dir / fname).write_bytes((legacy_dir / fname).read_bytes())
    (v2_dir / "cutoff.json").write_text(json.dumps({"state_nll_cutoff": 12.123456}) + "\n")

    v2 = FrozenStateScorer(v2_dir)
    assert np.isclose(v2.cutoff, 12.123456)
    assert np.isclose(v2.z_cutoff, (12.123456 - v2.median) / v2.iqr)
