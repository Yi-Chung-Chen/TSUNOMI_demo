"""Stream an ADSB.lol daily split tarball, extract ORD traces, and convert 5-min scenarios."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time

from src.preprocess.adsb_convert import (
    BOUNDS,
    DURATION_S,
    interpolate_track,
    parse_readsb_trace,
    qualify_aircraft_in_window,
    slice_observations_for_window,
    write_scenario,
)


class ConcatReader(io.RawIOBase):
    """Read-only sequential stream over split tarball parts (.tar.aa, .tar.ab, ...)."""

    def __init__(self, paths: list[Path]):
        self._paths = list(paths)
        self._idx = 0
        self._fh = self._paths[0].open("rb") if self._paths else None

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        if self._fh is None:
            return b""
        if size is None or size < 0:
            chunks = [self._fh.read()]
            self._fh.close()
            self._fh = None
            for p in self._paths[self._idx + 1 :]:
                chunks.append(p.read_bytes())
            self._idx = len(self._paths)
            return b"".join(chunks)

        out = bytearray()
        while len(out) < size and self._fh is not None:
            chunk = self._fh.read(size - len(out))
            if chunk:
                out.extend(chunk)
            else:
                self._fh.close()
                self._idx += 1
                if self._idx < len(self._paths):
                    self._fh = self._paths[self._idx].open("rb")
                else:
                    self._fh = None
        return bytes(out)

    def readinto(self, b: bytearray | memoryview) -> int:
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        super().close()


def verify_sha256(path: Path, expected_digest: str | None) -> None:
    if not expected_digest:
        return
    expected = expected_digest.split(":", 1)[-1].lower()
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(8 * 1024 * 1024)
            if not buf:
                break
            h.update(buf)
    actual = h.hexdigest().lower()
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {path.name}: expected {expected}, got {actual}")


def _parse_batch(raw_items: list[bytes]) -> list[tuple[str, dict, list[dict]]]:
    kept = []
    for raw in raw_items:
        if len(raw) >= 2 and raw[0] == 0x1F and raw[1] == 0x8B:
            raw = gzip.decompress(raw)
        data = json.loads(raw)
        parsed = parse_readsb_trace(data, bounds=BOUNDS, margin_deg=0.15)
        if parsed is not None:
            kept.append(parsed)
    return kept


def extract_ord_traces(part_paths: list[Path], cache_path: Path, workers: int = 8) -> list[tuple[str, dict, list[dict]]]:
    """Stream split tarball once and cache all aircraft traces entering ORD ± 0.15 deg."""
    if cache_path.exists():
        traces = []
        with gzip.open(cache_path, "rt") as f:
            for line in f:
                rec = json.loads(line)
                traces.append((rec["icao"], rec["meta"], rec["obs"]))
        return traces

    t0 = time.monotonic()
    traces: list[tuple[str, dict, list[dict]]] = []
    total_files = 0
    batch: list[bytes] = []
    batch_size = 512

    reader = io.BufferedReader(ConcatReader(part_paths), buffer_size=4 * 1024 * 1024)
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = []
            with tarfile.open(fileobj=reader, mode="r|*") as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    name = member.name
                    if "trace_full_" not in name or not name.endswith(".json"):
                        continue
                    fh = tf.extractfile(member)
                    if fh is None:
                        continue
                    raw = fh.read()
                    total_files += 1
                    batch.append(raw)
                    if len(batch) >= batch_size:
                        futures.append(ex.submit(_parse_batch, batch))
                        batch = []
                        if len(futures) >= workers * 4:
                            traces.extend(futures.pop(0).result())
            if batch:
                futures.append(ex.submit(_parse_batch, batch))
            for fut in futures:
                traces.extend(fut.result())
    finally:
        reader.close()

    # Deduplicate by icao if any duplicate member exists
    by_icao = {}
    for icao, meta, obs in traces:
        by_icao[icao] = (icao, meta, obs)
    traces = [by_icao[k] for k in sorted(by_icao)]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp.gz")
    with gzip.open(tmp, "wt") as f:
        for icao, meta, obs in traces:
            f.write(json.dumps({"icao": icao, "meta": meta, "obs": obs}, separators=(",", ":")) + "\n")
    tmp.replace(cache_path)

    print(
        json.dumps(
            {
                "stage": "extract_ord_traces",
                "cache": str(cache_path),
                "total_trace_files": total_files,
                "ord_aircraft": len(traces),
                "elapsed_s": round(time.monotonic() - t0, 2),
            }
        ),
        flush=True,
    )
    return traces


_CONVERT_CTX = None


def _convert_slot(slot: int) -> dict | None:
    day_start, day_tag, spans, out_dir, source_tag = _CONVERT_CTX
    start_utc_s = float(day_start + slot * DURATION_S)
    end_utc_s = start_utc_s + DURATION_S

    qualifying = []
    for t_min, t_max, icao, meta, obs_sorted in spans:
        if t_max < start_utc_s or t_min > end_utc_s:
            continue
        event = qualify_aircraft_in_window(obs_sorted, start_utc_s, duration_s=DURATION_S)
        if event is None:
            continue
        obs_win = slice_observations_for_window(obs_sorted, start_utc_s, duration_s=DURATION_S)
        ref_rows = interpolate_track(obs_win, start_utc_s, duration_s=DURATION_S)
        if not ref_rows:
            continue
        qualifying.append((meta, obs_win, ref_rows, event))

    if not qualifying:
        return None
    scen_name = f"scenario_{day_tag}_{slot:03d}_aircraft"
    return write_scenario(
        out_dir=out_dir,
        scenario_name=scen_name,
        start_utc_s=start_utc_s,
        qualifying_tracks=qualifying,
        source_tag=source_tag,
        duration_s=DURATION_S,
    )


def convert_day_scenarios(
    day_str: str,
    traces: list[tuple[str, dict, list[dict]]],
    out_dir: Path,
    source_tag: str,
    only_starts: set[float] | None = None,
    workers: int = 8,
) -> list[dict]:
    """Convert all 5-minute intervals of day_str (or only_starts) with >=1 qualifying aircraft."""
    global _CONVERT_CTX
    from multiprocessing import get_context

    t0 = time.monotonic()
    day_start = int(datetime.strptime(day_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())

    # Pre-index each aircraft's [min_t, max_t] for fast interval lookup
    spans = []
    for icao, meta, obs in traces:
        if not obs:
            continue
        obs_sorted = sorted(obs, key=lambda o: o["utc_s"])
        spans.append((obs_sorted[0]["utc_s"], obs_sorted[-1]["utc_s"], icao, meta, obs_sorted))

    out_dir.mkdir(parents=True, exist_ok=True)
    day_tag = day_str.replace("-", "")
    slots = [
        slot
        for slot in range(288)
        if only_starts is None or float(day_start + slot * DURATION_S) in only_starts
    ]

    _CONVERT_CTX = (day_start, day_tag, spans, out_dir, source_tag)
    if workers > 1 and len(slots) > 1:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("fork")) as ex:
            results = list(ex.map(_convert_slot, slots))
    else:
        results = [_convert_slot(s) for s in slots]
    entries = [r for r in results if r is not None]

    index_path = out_dir / "index.jsonl"
    with index_path.open("w") as f:
        for e in entries:
            f.write(json.dumps(e, separators=(",", ":")) + "\n")

    print(
        json.dumps(
            {
                "stage": "convert_day_scenarios",
                "date": day_str,
                "scenarios": len(entries),
                "total_aircraft": sum(e["aircraft"] for e in entries),
                "total_ref_samples": sum(e["reference_samples"] for e in entries),
                "elapsed_s": round(time.monotonic() - t0, 2),
            }
        ),
        flush=True,
    )
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--adsb-root", type=Path, default=Path("/local/scratch/a/chen5262/sygard_adsb"))
    parser.add_argument("--selected-json", type=Path, default=Path("outputs/density_v2/selected_days.json"))
    parser.add_argument("--out-subdir", default="scenarios", help="Subdir under adsb-root")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-sha", action="store_true")
    args = parser.parse_args()

    meta_all = json.loads(args.selected_json.read_text())
    rel = None
    if args.date == "2026-09-29" and meta_all.get("gate_b_release"):
        rel = meta_all["gate_b_release"]
    else:
        for item in meta_all["selected_days"] + meta_all["backup_east_days"]:
            if item["date"] == args.date:
                rel = item["release"]
                break
    if rel is None:
        raise ValueError(f"No release metadata found for {args.date}")

    raw_dir = args.adsb_root / "raw" / args.date
    part_paths = [raw_dir / a["name"] for a in rel["assets"]]
    for p, a in zip(part_paths, rel["assets"]):
        if not p.exists():
            raise FileNotFoundError(f"Missing tarball part: {p}")
        if not args.skip_sha:
            verify_sha256(p, a.get("digest"))

    cache_path = args.adsb_root / "ord_traces" / f"ord_traces_{args.date}.jsonl.gz"
    traces = extract_ord_traces(part_paths, cache_path, workers=args.workers)
    out_dir = args.adsb_root / args.out_subdir / args.date
    convert_day_scenarios(args.date, traces, out_dir, source_tag=rel["tag"])


if __name__ == "__main__":
    main()
