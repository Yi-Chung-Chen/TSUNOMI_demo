"""M13 Step 1: Select balanced east/west-flow candidate days from KORD hourly METAR."""
import argparse
import csv
from collections import Counter, defaultdict
from datetime import date
import json
from pathlib import Path
import re


START_DATE = date(2026, 8, 15)
END_DATE = date(2026, 11, 20)
EXCLUDED_RANGES = [
    (date(2026, 9, 28), date(2026, 10, 4)),   # 800-scenario train/val set
    (date(2026, 10, 6), date(2026, 10, 14)),  # 100-scenario mixed dev set
]
TAG_RE = re.compile(r"^v(\d{4})\.(\d{2})\.(\d{2})-planes-readsb-(prod|staging)-0$")


def is_excluded(day: date) -> bool:
    return any(lo <= day <= hi for lo, hi in EXCLUDED_RANGES)


def load_releases(metar_dir: Path) -> dict[str, dict]:
    """Return prod-0 release metadata keyed by YYYY-MM-DD."""
    by_date: dict[str, dict] = {}
    for path in sorted(metar_dir.glob("releases_p*.json")):
        items = json.loads(path.read_text())
        for rel in items:
            m = TAG_RE.match(rel.get("tag_name", ""))
            if not m:
                continue
            y, mo, d, kind = m.groups()
            day_str = f"{y}-{mo}-{d}"
            assets = [
                {
                    "name": a["name"],
                    "size": a["size"],
                    "digest": a.get("digest"),
                    "url": a["browser_download_url"],
                }
                for a in sorted(rel.get("assets", []), key=lambda a: a["name"])
                if ".tar" in a["name"]
            ]
            if not assets:
                continue
            # Prefer prod-0 over staging-0; keep complete releases (>=2 parts or >=1GB)
            total_bytes = sum(a["size"] for a in assets)
            prev = by_date.get(day_str)
            if prev is None or (kind == "prod" and prev["kind"] != "prod"):
                by_date[day_str] = {
                    "tag": rel["tag_name"],
                    "kind": kind,
                    "total_bytes": total_bytes,
                    "assets": assets,
                }
    return by_date


def parse_metar(csv_path: Path, releases: dict[str, dict]) -> list[dict]:
    """Aggregate routine hourly METAR wind reports by UTC date."""
    per_day_hour: dict[tuple[str, str], tuple[float | None, float | None]] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            valid = row["valid"].strip()
            day_str, hhmm = valid.split(" ")
            hour = hhmm.split(":")[0]
            drct = None if row["drct"] in ("", "M") else float(row["drct"])
            sknt = None if row["sknt"] in ("", "M") else float(row["sknt"])
            # Keep first routine report per UTC hour (deduplicates any duplicate lines)
            key = (day_str, hour)
            if key not in per_day_hour:
                per_day_hour[key] = (drct, sknt)

    grouped: dict[str, list[tuple[float | None, float | None]]] = defaultdict(list)
    for (day_str, _), obs in sorted(per_day_hour.items()):
        grouped[day_str].append(obs)

    rows = []
    for day_str in sorted(grouped):
        d = date.fromisoformat(day_str)
        if not (START_DATE <= d <= END_DATE):
            continue
        obs = grouped[day_str]
        east = 0
        west = 0
        other_ge5 = 0
        calm_or_lt5 = 0
        missing = 0
        for drct, sknt in obs:
            if drct is None or sknt is None:
                missing += 1
            elif sknt < 5.0:
                calm_or_lt5 += 1
            elif 20.0 <= drct <= 160.0:
                east += 1
            elif 200.0 <= drct <= 340.0:
                west += 1
            else:
                other_ge5 += 1
        iso_year, iso_week, iso_dow = d.isocalendar()
        rel = releases.get(day_str)
        rows.append(
            {
                "date": day_str,
                "weekday": d.strftime("%a"),
                "is_weekend": iso_dow >= 6,
                "iso_week": f"{iso_year}-W{iso_week:02d}",
                "excluded": is_excluded(d),
                "has_prod_release": bool(rel and rel["kind"] == "prod" and rel["total_bytes"] >= 1_500_000_000),
                "release_tag": rel["tag"] if rel else None,
                "release_bytes": rel["total_bytes"] if rel else 0,
                "n_hours": len(obs),
                "east_hours": east,
                "west_hours": west,
                "other_ge5_hours": other_ge5,
                "calm_lt5_hours": calm_or_lt5,
                "missing_hours": missing,
                "net_east": east - west,
            }
        )
    return rows


def select_days(rows: list[dict], n_east: int = 5, n_west: int = 3, max_per_week: int = 2) -> tuple[list[dict], list[dict]]:
    """Deterministic fixed rule:
    - Eligible: not excluded, has_prod_release, n_hours >= 22.
    - East-likely pool: east_hours > west_hours, ranked by (east_hours - west_hours, east_hours, date) desc.
    - West-likely pool: west_hours > east_hours, ranked by (west_hours - east_hours, west_hours, date) desc.
    - Enforce <= max_per_week per ISO week, >= 2 weekend days total (at least 1 east weekend + 1 west weekend if available,
      otherwise filling weekend quota from strongest weekend days first), >= 4 distinct ISO weeks.
    """
    eligible = [r for r in rows if not r["excluded"] and r["has_prod_release"] and r["n_hours"] >= 22]
    east_pool = sorted(
        [r for r in eligible if r["east_hours"] > r["west_hours"]],
        key=lambda r: (r["east_hours"] - r["west_hours"], r["east_hours"], r["date"]),
        reverse=True,
    )
    west_pool = sorted(
        [r for r in eligible if r["west_hours"] > r["east_hours"]],
        key=lambda r: (r["west_hours"] - r["east_hours"], r["west_hours"], r["date"]),
        reverse=True,
    )

    selected: list[dict] = []
    week_counts: Counter = Counter()
    used_dates: set[str] = set()

    def try_add(row: dict, role: str) -> bool:
        if row["date"] in used_dates:
            return False
        if week_counts[row["iso_week"]] >= max_per_week:
            return False
        item = dict(row, selected_role=role)
        selected.append(item)
        week_counts[row["iso_week"]] += 1
        used_dates.add(row["date"])
        return True

    # First ensure 1 strongest east weekend day and 1 strongest west weekend day
    for r in east_pool:
        if r["is_weekend"] and try_add(r, "east_weekend"):
            break
    for r in west_pool:
        if r["is_weekend"] and try_add(r, "west_weekend"):
            break

    # Fill remaining east slots
    east_picked = sum(1 for r in selected if r["selected_role"].startswith("east"))
    for r in east_pool:
        if east_picked >= n_east:
            break
        if try_add(r, "east"):
            east_picked += 1

    # Fill remaining west slots
    west_picked = sum(1 for r in selected if r["selected_role"].startswith("west"))
    for r in west_pool:
        if west_picked >= n_west:
            break
        if try_add(r, "west"):
            west_picked += 1

    # Verify M13 Step 1 constraints
    assert len(selected) == n_east + n_west
    assert sum(1 for r in selected if r["selected_role"].startswith("east")) >= 4
    assert sum(1 for r in selected if r["selected_role"].startswith("west")) >= 3
    assert sum(1 for r in selected if r["is_weekend"]) >= 2
    assert len({r["iso_week"] for r in selected}) >= 4

    # Backup east-likely days (up to 4) respecting max_per_week
    backups: list[dict] = []
    backup_weeks = Counter(week_counts)
    for r in east_pool:
        if len(backups) >= 4:
            break
        if r["date"] in used_dates or backup_weeks[r["iso_week"]] >= max_per_week:
            continue
        backups.append(dict(r, selected_role="east_backup"))
        backup_weeks[r["iso_week"]] += 1

    selected.sort(key=lambda r: r["date"])
    return selected, backups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adsb-root", type=Path, default=Path("/local/scratch/a/chen5262/sygard_adsb"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/density_v2"))
    args = parser.parse_args()

    metar_dir = args.adsb_root / "metar"
    releases = load_releases(metar_dir)
    rows = parse_metar(metar_dir / "kord_metar_20260815_20261120.csv", releases)
    selected, backups = select_days(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "metar_candidates.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "rule": (
            "Candidate dates 2026-08-15..2026-11-20 excluding 09-28..10-04 (800 set) and 10-06..10-14 (mixed set), "
            "requiring >=22 hourly KORD METAR reports and an available adsblol/globe_history_2026 prod-0 tarball. "
            "East-likely: wind >=5 kt from 020-160 deg; west-likely: wind >=5 kt from 200-340 deg. "
            "Pick 5 east-likely + 3 west-likely days, starting with the strongest east weekend day and strongest west weekend day, "
            "then filling remaining slots by (dominant_hours - opposite_hours, dominant_hours, date) descending, "
            "with at most 2 days per ISO calendar week."
        ),
        "gate_b_release": releases.get("2026-09-29"),
        "selected_days": [
            {**r, "release": releases[r["date"]]} for r in selected
        ],
        "backup_east_days": [
            {**r, "release": releases[r["date"]]} for r in backups
        ],
        "summary": {
            "total_candidate_days": len(rows),
            "eligible_days": sum(1 for r in rows if not r["excluded"] and r["has_prod_release"] and r["n_hours"] >= 22),
            "selected_count": len(selected),
            "east_count": sum(1 for r in selected if r["selected_role"].startswith("east")),
            "west_count": sum(1 for r in selected if r["selected_role"].startswith("west")),
            "weekend_count": sum(1 for r in selected if r["is_weekend"]),
            "distinct_weeks": len({r["iso_week"] for r in selected}),
            "total_selected_tarball_gb": round(sum(r["release_bytes"] for r in selected) / 1e9, 2),
        },
    }
    (args.out_dir / "selected_days.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))
    print("\nSelected 8 days:")
    for r in selected:
        print(
            f"  {r['date']} ({r['weekday']}, {r['iso_week']}) role={r['selected_role']:12s} "
            f"east={r['east_hours']:2d} west={r['west_hours']:2d} calm={r['calm_lt5_hours']:2d} "
            f"size={r['release_bytes']/1e9:.2f}GB tag={r['release_tag']}"
        )
    print("\nBackup east days:")
    for r in backups:
        print(
            f"  {r['date']} ({r['weekday']}, {r['iso_week']}) east={r['east_hours']:2d} west={r['west_hours']:2d} "
            f"size={r['release_bytes']/1e9:.2f}GB tag={r['release_tag']}"
        )
    print("\nGate B (2026-09-29):", payload["gate_b_release"]["tag"] if payload["gate_b_release"] else "MISSING")


if __name__ == "__main__":
    main()
