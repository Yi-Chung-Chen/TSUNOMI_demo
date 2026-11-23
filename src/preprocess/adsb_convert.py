"""Convert ADSB.lol/readsb traces to SYGARD 800-set 3 Hz scenario format."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import numpy as np

ORIGIN = {
    "latitude_deg": 41.980103,
    "longitude_deg": -87.903868,
    "altitude_m": 2250.0,
}
BOUNDS = {
    "south": 41.94,
    "north": 42.015,
    "west": -87.95,
    "east": -87.845,
}
MAX_INTERP_GAP_S = 30.0
FALLBACK_ALT_M = 166.8
REF_GROUND_ALT_M = 166.0
FRAME_HZ = 3
DURATION_S = 300
N_FRAMES = DURATION_S * FRAME_HZ + 1

FT_TO_M = 0.3048
KT_TO_MPS = 1852.0 / 3600.0
FPM_TO_MPS = FT_TO_M / 60.0

SELECTION_RULES = {
    "area": "ORD airport bounds",
    "min_airborne_duration_s": 25.0,
    "min_speed_mps": 30.0,
    "min_traveled_m": 1200.0,
    "min_displacement_m": 800.0,
    "min_source_observations": 4,
    "min_height_above_reference_ground_m": 25.0,
    "max_height_above_reference_ground_m": 1100.0,
    "reference_ground_altitude_m": REF_GROUND_ALT_M,
}

# Precompute WGS84 ECEF->ENU constants for ORIGIN
_WGS84_A = 6378137.0
_WGS84_E2 = 0.0066943799901413165


def _ecef_origin(origin: dict = ORIGIN) -> tuple[np.ndarray, np.ndarray]:
    lat0 = math.radians(origin["latitude_deg"])
    lon0 = math.radians(origin["longitude_deg"])
    alt0 = float(origin["altitude_m"])
    slat, clat = math.sin(lat0), math.cos(lat0)
    slon, clon = math.sin(lon0), math.cos(lon0)
    n0 = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * slat * slat)
    ecef0 = np.array(
        [
            (n0 + alt0) * clat * clon,
            (n0 + alt0) * clat * slon,
            (n0 * (1.0 - _WGS84_E2) + alt0) * slat,
        ],
        dtype=np.float64,
    )
    rot = np.array(
        [
            [-slon, clon, 0.0],
            [-slat * clon, -slat * slon, clat],
            [clat * clon, clat * slon, slat],
        ],
        dtype=np.float64,
    )
    return ecef0, rot


_ECEF0, _ROT0 = _ecef_origin(ORIGIN)


def geodetic_to_enu_batch(
    lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray, origin: dict = ORIGIN
) -> np.ndarray:
    """Vectorized WGS84 geodetic -> local ENU (metres), matching runway_anchors.geodetic_to_enu."""
    ecef0, rot = (_ECEF0, _ROT0) if origin is ORIGIN else _ecef_origin(origin)
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    alt = np.asarray(alt_m, dtype=np.float64)
    slat = np.sin(lat)
    clat = np.cos(lat)
    slon = np.sin(lon)
    clon = np.cos(lon)
    n = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * slat * slat)
    dx = (n + alt) * clat * clon - ecef0[0]
    dy = (n + alt) * clat * slon - ecef0[1]
    dz = (n * (1.0 - _WGS84_E2) + alt) * slat - ecef0[2]
    diff = np.stack((dx, dy, dz), axis=-1)
    return diff @ rot.T


def compass_deg_to_heading_rad(deg: float) -> float:
    """Compass degrees CW from north -> radians CCW from east in (-pi, pi]."""
    rad = math.radians(90.0 - deg)
    return (rad + math.pi) % (2.0 * math.pi) - math.pi


def heading_rad_to_compass_deg(rad: float) -> float:
    return (90.0 - math.degrees(rad)) % 360.0


def in_bounds(lat: float, lon: float, bounds: dict = BOUNDS) -> bool:
    return (
        bounds["south"] <= lat <= bounds["north"]
        and bounds["west"] <= lon <= bounds["east"]
    )


def _euler_to_quat_xyzw(yaw: float, pitch: float, roll: float) -> list[float]:
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def interpolate_track(
    observations: list[dict],
    start_utc_s: float,
    duration_s: int = DURATION_S,
    hz: int = FRAME_HZ,
    bounds: dict = BOUNDS,
    origin: dict = ORIGIN,
    max_gap_s: float = MAX_INTERP_GAP_S,
) -> list[dict]:
    """Interpolate one aircraft's irregular ADS-B observations onto the 3 Hz grid.

    Follows SYGARD 800-set README conventions:
    - Grid frames 0..duration_s*hz inclusive; elapsed_s = frame_index / hz.
    - No extrapolation before first or after last observation.
    - Linear lat/lon/altitude interpolation; missing altitude uses 166.8 m fallback.
    - Gaps > 30 s, flagged airborne discontinuities (flags & 2), or outside-bounds
      interpolated positions produce no row.
    - on_ground from nearer endpoint; heading toward next waypoint with reported fallback.
    """
    if not observations:
        return []

    # Sort by utc_s and collapse exact duplicate timestamps (keep last)
    sorted_obs = sorted(observations, key=lambda o: o["utc_s"])
    dedup: list[dict] = []
    for o in sorted_obs:
        if dedup and o["utc_s"] == dedup[-1]["utc_s"]:
            dedup[-1] = o
        else:
            dedup.append(o)
    obs = dedup

    ts = np.array([o["utc_s"] for o in obs], dtype=np.float64)
    lats = np.array([o["latitude"] for o in obs], dtype=np.float64)
    lons = np.array([o["longitude"] for o in obs], dtype=np.float64)
    alts = np.array(
        [FALLBACK_ALT_M if o["altitude_m"] is None else float(o["altitude_m"]) for o in obs],
        dtype=np.float64,
    )
    alt_missing = np.array([o["altitude_m"] is None for o in obs], dtype=bool)
    on_ground = np.array([bool(o["on_ground"]) for o in obs], dtype=bool)
    flags = np.array([int(o.get("flags") or 0) for o in obs], dtype=np.int64)
    obs_enu = geodetic_to_enu_batch(lats, lons, alts, origin=origin)

    n_frames = duration_s * hz + 1
    frames = np.arange(n_frames, dtype=np.int64)
    grid_t = start_utc_s + frames / float(hz)

    in_span = (grid_t >= ts[0]) & (grid_t <= ts[-1])
    if not in_span.any():
        return []

    cand_frames = frames[in_span]
    cand_t = grid_t[in_span]
    # idx_right is first index where ts[idx_right] >= cand_t
    idx_right = np.searchsorted(ts, cand_t, side="left")
    exact = (idx_right < len(ts)) & (ts[idx_right] == cand_t)
    idx_left = np.where(exact, idx_right, idx_right - 1)

    t0 = ts[idx_left]
    t1 = ts[idx_right]
    dt = t1 - t0

    # Reject gaps > max_gap_s and flagged discontinuities (readsb bit 1 = new leg / discontinuity)
    # Note: bit 1 (flags & 2) on o1 marks a discontinuity between o0 and o1; check airborne or general.
    discontinuity = (~exact) & (((flags[idx_right] & 2) != 0) & (~on_ground[idx_left] | ~on_ground[idx_right]))
    ok = (dt <= max_gap_s) & (~discontinuity)
    if not ok.any():
        return []

    cand_frames = cand_frames[ok]
    cand_t = cand_t[ok]
    idx_left = idx_left[ok]
    idx_right = idx_right[ok]
    exact = exact[ok]
    t0 = t0[ok]
    t1 = t1[ok]
    dt = dt[ok]

    frac = np.where(exact, 0.0, (cand_t - t0) / np.where(exact, 1.0, dt))
    lat_i = lats[idx_left] + frac * (lats[idx_right] - lats[idx_left])
    lon_i = lons[idx_left] + frac * (lons[idx_right] - lons[idx_left])
    alt_i = alts[idx_left] + frac * (alts[idx_right] - alts[idx_left])

    inside = (
        (lat_i >= bounds["south"])
        & (lat_i <= bounds["north"])
        & (lon_i >= bounds["west"])
        & (lon_i <= bounds["east"])
    )
    if not inside.any():
        return []

    cand_frames = cand_frames[inside]
    cand_t = cand_t[inside]
    idx_left = idx_left[inside]
    idx_right = idx_right[inside]
    frac = frac[inside]
    t0 = t0[inside]
    t1 = t1[inside]
    lat_i = lat_i[inside]
    lon_i = lon_i[inside]
    alt_i = alt_i[inside]

    enu_i = geodetic_to_enu_batch(lat_i, lon_i, alt_i, origin=origin)
    near_right = frac > 0.5
    near_idx = np.where(near_right, idx_right, idx_left)
    og_i = on_ground[near_idx]
    fb_i = alt_missing[idx_left] | alt_missing[idx_right]

    # Segment heading toward next waypoint (idx_right) or next observation if exact
    rows: list[dict] = []
    source_object_id = str(obs[0]["source_object_id"])
    vehicle_type = obs[0].get("vehicle_type", "aircraft")
    reported_type = obs[0].get("adsb_reported_type")

    prev_frame: int | None = None
    prev_enu: np.ndarray | None = None

    for j in range(len(cand_frames)):
        f_idx = int(cand_frames[j])
        il = int(idx_left[j])
        ir = int(idx_right[j])
        ni = int(near_idx[j])
        cur_enu = enu_i[j]

        # Heading toward next route waypoint, fallback to reported heading
        if ir > il:
            d_wp = obs_enu[ir] - obs_enu[il]
        elif ir + 1 < len(obs_enu) and (ts[ir + 1] - ts[ir]) <= max_gap_s:
            d_wp = obs_enu[ir + 1] - obs_enu[ir]
        elif il > 0 and (ts[il] - ts[il - 1]) <= max_gap_s:
            d_wp = obs_enu[il] - obs_enu[il - 1]
        else:
            d_wp = np.zeros(3, dtype=np.float64)

        horiz_dist = math.hypot(float(d_wp[0]), float(d_wp[1]))
        if horiz_dist > 1e-3:
            hdg_rad = math.atan2(float(d_wp[1]), float(d_wp[0]))
            pitch_rad = math.atan2(float(d_wp[2]), horiz_dist)
        else:
            rep_deg = obs[ni].get("heading_deg")
            if rep_deg is not None and math.isfinite(rep_deg):
                hdg_rad = compass_deg_to_heading_rad(float(rep_deg))
            else:
                hdg_rad = 0.0
            pitch_rad = 0.0

        roll_deg = obs[ni].get("roll_deg")
        roll_rad = math.radians(float(roll_deg)) if roll_deg is not None and math.isfinite(roll_deg) else 0.0
        quat = _euler_to_quat_xyzw(hdg_rad, pitch_rad, roll_rad)

        if prev_frame is not None and f_idx == prev_frame + 1 and prev_enu is not None:
            vel = ((cur_enu - prev_enu) * float(hz)).tolist()
        else:
            vel = None

        alt_src = obs[ni].get("altitude_source") or (
            "terrain_collision_or_documented_fallback" if fb_i[j] else "reported_geometric"
        )
        rows.append(
            {
                "vehicle_type": vehicle_type,
                "frame_index": f_idx,
                "elapsed_s": f_idx / float(hz),
                "utc_s": float(cand_t[j]),
                "xyz_enu_m": [float(cur_enu[0]), float(cur_enu[1]), float(cur_enu[2])],
                "velocity_enu_mps": vel,
                "quaternion_xyzw_enu": quat,
                "heading_deg": heading_rad_to_compass_deg(hdg_rad),
                "latitude": float(lat_i[j]),
                "longitude": float(lon_i[j]),
                "altitude_m": float(alt_i[j]),
                "altitude_source": alt_src,
                "altitude_fallback_used": bool(fb_i[j]),
                "on_ground": bool(og_i[j]),
                "source_times_utc_s": [float(t0[j]), float(t1[j])],
                "interpolation_fraction": float(frac[j]),
                "source_object_id": source_object_id,
                "adsb_reported_type": reported_type,
                "heading_rad": float(hdg_rad),
            }
        )
        prev_frame = f_idx
        prev_enu = cur_enu

    return rows


def parse_readsb_trace(data: dict, bounds: dict = BOUNDS, margin_deg: float = 0.15) -> tuple[str, dict, list[dict]] | None:
    """Parse one readsb trace_full_<icao>.json object.

    Returns (icao24, aircraft_meta, observations) if any point falls within
    bounds + margin_deg, else None.
    """
    trace = data.get("trace")
    base_ts = data.get("timestamp")
    if not trace or base_ts is None:
        return None
    base_ts = float(base_ts)

    s_lo = bounds["south"] - margin_deg
    s_hi = bounds["north"] + margin_deg
    w_lo = bounds["west"] - margin_deg
    w_hi = bounds["east"] + margin_deg

    # Fast bounding-box check before converting rows
    any_near = False
    for pt in trace:
        if len(pt) < 3:
            continue
        lat, lon = pt[1], pt[2]
        if lat is not None and lon is not None and s_lo <= lat <= s_hi and w_lo <= lon <= w_hi:
            any_near = True
            break
    if not any_near:
        return None

    icao = str(data.get("icao", "")).strip().lower()
    reg = data.get("r")
    ac_type = data.get("t")
    if ac_type is not None:
        ac_type = str(ac_type).strip() or None

    obs: list[dict] = []
    for pt in trace:
        if len(pt) < 4:
            continue
        lat, lon = pt[1], pt[2]
        if lat is None or lon is None:
            continue
        lat = float(lat)
        lon = float(lon)
        if not (math.isfinite(lat) and math.isfinite(lon)):
            continue

        t_off = float(pt[0])
        utc_s = base_ts + t_off
        alt_baro = pt[3]
        gs_kt = pt[4] if len(pt) > 4 else None
        track_deg = pt[5] if len(pt) > 5 else None
        flags = int(pt[6]) if len(pt) > 6 and pt[6] is not None else 0
        vrate_fpm = pt[7] if len(pt) > 7 else None
        ac_dict = pt[8] if len(pt) > 8 and isinstance(pt[8], dict) else None
        source = pt[9] if len(pt) > 9 and pt[9] is not None else "adsb_icao"
        alt_geom_ft = pt[10] if len(pt) > 10 else None

        # readsb flags bit 0 (1) = stale position; drop if stale? Keep check via Gate B!
        on_ground = alt_baro == "ground"
        if on_ground:
            alt_m = None
            alt_src = "terrain_collision_or_documented_fallback"
        elif alt_geom_ft is not None and isinstance(alt_geom_ft, (int, float)) and math.isfinite(alt_geom_ft):
            alt_m = float(alt_geom_ft) * FT_TO_M
            alt_src = "reported_geometric"
        elif isinstance(alt_baro, (int, float)) and math.isfinite(alt_baro):
            alt_m = float(alt_baro) * FT_TO_M
            alt_src = "barometric_uncorrected"
        else:
            alt_m = None
            alt_src = "terrain_collision_or_documented_fallback"

        speed_mps = float(gs_kt) * KT_TO_MPS if isinstance(gs_kt, (int, float)) and math.isfinite(gs_kt) else None
        heading_deg = float(track_deg) if isinstance(track_deg, (int, float)) and math.isfinite(track_deg) else None
        vspeed_mps = float(vrate_fpm) * FPM_TO_MPS if isinstance(vrate_fpm, (int, float)) and math.isfinite(vrate_fpm) else None

        roll_deg = None
        callsign = ""
        if ac_dict:
            r_val = ac_dict.get("roll")
            if isinstance(r_val, (int, float)) and math.isfinite(r_val):
                roll_deg = float(r_val)
            cs = ac_dict.get("flight")
            if isinstance(cs, str):
                callsign = cs.strip()
            if ac_type is None and ac_dict.get("t"):
                ac_type = str(ac_dict["t"]).strip() or None

        obs.append(
            {
                "utc_s": utc_s,
                "latitude": lat,
                "longitude": lon,
                "altitude_m": alt_m,
                "altitude_source": alt_src,
                "on_ground": on_ground,
                "speed_mps": speed_mps,
                "heading_deg": heading_deg,
                "vertical_speed_mps": vspeed_mps,
                "roll_deg": roll_deg,
                "callsign": callsign,
                "source": str(source),
                "flags": flags,
                "source_object_id": icao,
                "vehicle_type": "aircraft",
                "adsb_reported_type": ac_type,
            }
        )

    if not obs:
        return None
    meta = {
        "source_object_id": icao,
        "registration": reg,
        "adsb_reported_type": ac_type,
    }
    return icao, meta, obs


def qualify_aircraft_in_window(
    observations: list[dict],
    start_utc_s: float,
    duration_s: int = DURATION_S,
    bounds: dict = BOUNDS,
    origin: dict = ORIGIN,
) -> dict | None:
    """Apply Josh's selection.rules to one aircraft in [start_utc_s, start_utc_s + duration_s].

    An aircraft qualifies if it has a contiguous airborne segment inside ORD bounds with:
    - height above 166 m in [25, 1100] m
    - reported speed >= 30 m/s
    - duration >= 25 s
    - traveled distance >= 1200 m
    - net displacement >= 800 m
    - source observations >= 4
    """
    end_utc_s = start_utc_s + duration_s
    # Filter observations inside the time window
    win = [o for o in observations if start_utc_s <= o["utc_s"] <= end_utc_s]
    if len(win) < SELECTION_RULES["min_source_observations"]:
        return None

    # Identify qualifying airborne observations inside bounds/height/speed
    min_h = SELECTION_RULES["min_height_above_reference_ground_m"]
    max_h = SELECTION_RULES["max_height_above_reference_ground_m"]
    min_spd = SELECTION_RULES["min_speed_mps"]

    # Partition into contiguous segments (breaking on >MAX_INTERP_GAP_S, flags&2, or non-qualifying obs)
    segments: list[list[dict]] = []
    cur: list[dict] = []
    for o in win:
        alt = o["altitude_m"]
        h = (alt - REF_GROUND_ALT_M) if alt is not None else -999.0
        spd = o["speed_mps"]
        qual_pt = (
            (not o["on_ground"])
            and in_bounds(o["latitude"], o["longitude"], bounds)
            and (min_h <= h <= max_h)
            and (spd is not None and spd >= min_spd)
        )
        if qual_pt:
            if cur and ((o["utc_s"] - cur[-1]["utc_s"]) > MAX_INTERP_GAP_S or ((int(o.get("flags") or 0) & 2) != 0)):
                segments.append(cur)
                cur = [o]
            else:
                cur.append(o)
        else:
            if cur:
                segments.append(cur)
                cur = []
    if cur:
        segments.append(cur)

    best_event = None
    for seg in segments:
        if len(seg) < SELECTION_RULES["min_source_observations"]:
            continue
        dur = seg[-1]["utc_s"] - seg[0]["utc_s"]
        if dur < SELECTION_RULES["min_airborne_duration_s"]:
            continue
        lats = np.array([o["latitude"] for o in seg], dtype=np.float64)
        lons = np.array([o["longitude"] for o in seg], dtype=np.float64)
        alts = np.array([o["altitude_m"] for o in seg], dtype=np.float64)
        enu = geodetic_to_enu_batch(lats, lons, alts, origin=origin)
        step_d = np.linalg.norm(np.diff(enu[:, :2], axis=0), axis=1)
        traveled = float(step_d.sum())
        disp = float(np.linalg.norm(enu[-1, :2] - enu[0, :2]))
        if traveled < SELECTION_RULES["min_traveled_m"] or disp < SELECTION_RULES["min_displacement_m"]:
            continue
        alt_change = float(alts[-1] - alts[0])
        category = "climbing" if alt_change > 50.0 else ("descending" if alt_change < -50.0 else "level")
        ev = {
            "icao24": str(seg[0]["source_object_id"]),
            "vehicle_type": "aircraft",
            "category": category,
            "start_s": round(seg[0]["utc_s"] - start_utc_s),
            "end_s": round(seg[-1]["utc_s"] - start_utc_s),
            "duration_s": round(dur),
            "traveled_m": round(traveled),
            "displacement_m": round(disp),
            "source_observations": len(seg),
            "altitude_change_m": round(alt_change),
        }
        if best_event is None or ev["duration_s"] > best_event["duration_s"]:
            best_event = ev
    return best_event


def slice_observations_for_window(
    observations: list[dict], start_utc_s: float, duration_s: int = DURATION_S, pad_s: float = MAX_INTERP_GAP_S
) -> list[dict]:
    """Keep all observations in [start_utc_s, end_utc_s] plus up to one bracketing obs on each side."""
    end_utc_s = start_utc_s + duration_s
    # Find indices inside [start_utc_s, end_utc_s]
    lo = None
    hi = None
    for i, o in enumerate(observations):
        t = o["utc_s"]
        if t >= start_utc_s and lo is None:
            lo = i
        if t <= end_utc_s:
            hi = i
    if lo is None or hi is None or lo > hi:
        # Still check if a pair brackets the window within pad_s
        ts = [o["utc_s"] for o in observations]
        # Simple fallback: keep obs within [start_utc_s - pad_s, end_utc_s + pad_s]
        return [o for o in observations if (start_utc_s - pad_s) <= o["utc_s"] <= (end_utc_s + pad_s)]
    if lo > 0 and (start_utc_s - observations[lo - 1]["utc_s"]) <= pad_s:
        lo -= 1
    if hi + 1 < len(observations) and (observations[hi + 1]["utc_s"] - end_utc_s) <= pad_s:
        hi += 1
    return observations[lo : hi + 1]


def write_scenario(
    out_dir: Path,
    scenario_name: str,
    start_utc_s: float,
    qualifying_tracks: list[tuple[dict, list[dict], list[dict], dict]],
    source_tag: str,
    duration_s: int = DURATION_S,
) -> dict:
    """Write one scenario folder in exact 800-set schema and return its index.jsonl entry.

    qualifying_tracks: list of (ac_meta, obs_rows, ref_rows, selection_event)
    """
    folder = out_dir / scenario_name
    folder.mkdir(parents=True, exist_ok=True)

    # Order reference_tracks_3hz.jsonl by (frame_index, source_object_id)
    all_ref: list[dict] = []
    all_obs: list[dict] = []
    track_meta: list[dict] = []
    events: list[dict] = []

    for ac_meta, obs_rows, ref_rows, event in sorted(qualifying_tracks, key=lambda x: x[0]["source_object_id"]):
        all_obs.extend(obs_rows)
        all_ref.extend(ref_rows)
        track_meta.append(
            {
                "vehicle_type": "aircraft",
                "registration": ac_meta.get("registration"),
                "adsb_reported_type": ac_meta.get("adsb_reported_type"),
                "trajectory_source": "adsb_route",
                "source_observations": len(obs_rows),
                "reference_samples": len(ref_rows),
                "source_object_id": ac_meta["source_object_id"],
            }
        )
        events.append(event)

    all_ref.sort(key=lambda r: (r["frame_index"], r["source_object_id"]))

    with (folder / "reference_tracks_3hz.jsonl").open("w") as f:
        for r in all_ref:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")

    with (folder / "adsb_observations.jsonl").open("w") as f:
        for o in all_obs:
            f.write(json.dumps(o, separators=(",", ":")) + "\n")

    start_iso = datetime.fromtimestamp(start_utc_s, tz=timezone.utc).isoformat()
    end_iso = datetime.fromtimestamp(start_utc_s + duration_s, tz=timezone.utc).isoformat()

    metadata = {
        "schema_version": 2,
        "instance": scenario_name,
        "dataset_kind": "route_trajectories_without_rendering",
        "start_utc_s": float(start_utc_s),
        "duration_s": int(duration_s),
        "interval_utc": [start_iso, end_iso],
        "coordinate_frame": {
            "axes": "local ENU, metres: x east, y north, z up",
            "origin_wgs84": dict(ORIGIN),
        },
        "sampling": {
            "reference_rate_hz": FRAME_HZ,
            "grid_timestamps": duration_s * FRAME_HZ + 1,
            "source_observations": "Original route observations and timestamps; no resampling",
            "full_interval": True,
            "window_segmentation": False,
            "nominal_rate_hz": FRAME_HZ,
        },
        "generation": {
            "position_source": "Calculated from ADSB.lol globe_history_2026 traces (M13 converter)",
            "interpolation": "Linear latitude, longitude and altitude",
            "max_interpolation_gap_s": int(MAX_INTERP_GAP_S),
            "bounds": dict(BOUNDS),
            "missing_altitude_fallback_m": FALLBACK_ALT_M,
            "terrain_queries_performed": False,
            "unreal_used": False,
            "cameras_used": False,
        },
        "selection": {
            "rules": dict(SELECTION_RULES),
            "qualifying_aircraft": len(track_meta),
            "events": events,
        },
        "source_input": {
            "archive": {
                "name": "ADSB.lol/readsb",
                "license": "ODbL-1.0",
                "release_tag": source_tag,
            }
        },
        "counts": {
            "reference_track_rows": len(all_ref),
            "source_observation_rows": len(all_obs),
            "objects_with_tracks": len(track_meta),
            "source_objects": len(track_meta),
        },
        "files": {
            "reference_tracks": "reference_tracks_3hz.jsonl",
            "source_observations": "adsb_observations.jsonl",
        },
        "track_ids": {
            "reference_tracks": track_meta,
        },
    }
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    return {
        "scenario": scenario_name,
        "path": scenario_name,
        "start_utc": start_iso,
        "end_utc": end_iso,
        "start_utc_s": float(start_utc_s),
        "duration_s": int(duration_s),
        "aircraft": len(track_meta),
        "reference_samples": len(all_ref),
        "source_observations": len(all_obs),
    }
