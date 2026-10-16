"""Fail when any source Python file is missing from the Git index."""
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0"))
source = {p.relative_to(root).as_posix() for p in (root / "src").rglob("*.py")}
missing = sorted(source - tracked)
print(f"Tracked source Python files: {len(source - set(missing))}/{len(source)}")
if missing:
    raise SystemExit("Untracked source files: " + ", ".join(missing))
