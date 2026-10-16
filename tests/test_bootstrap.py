from pathlib import Path
import yaml

def test_paths_and_gpu_defaults():
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs/paths/default.yaml").read_text())
    assert Path(cfg["sygard_root"]).is_dir()
    assert "CUDA_VISIBLE_DEVICES=1,3" in (root / "scripts/catk.sh").read_text()
