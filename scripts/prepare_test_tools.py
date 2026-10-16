"""Run with base Python only if catk lacks pytest (no environment changes)."""
from pathlib import Path
import importlib.util
import shutil

out = Path(__file__).resolve().parents[1] / ".cache/test-tools"
out.mkdir(parents=True, exist_ok=True)
for name in ("pytest", "_pytest", "pluggy", "iniconfig", "py"):
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise SystemExit(f"Base Python is missing {name}; install pytest in catk instead")
    source = Path(spec.origin)
    if source.name == "__init__.py":
        shutil.copytree(source.parent, out / name, dirs_exist_ok=True)
    else:
        shutil.copy2(source, out / source.name)
print(out)
