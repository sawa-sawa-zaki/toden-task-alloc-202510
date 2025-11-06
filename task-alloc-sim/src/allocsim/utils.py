import hashlib, json, os, time, random
from dataclasses import dataclass
from pydantic import BaseModel


def make_run_dir(base_dir: str, cfg: BaseModel) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    h = hashlib.sha1(json.dumps(cfg.model_dump(), sort_keys=True).encode()).hexdigest()[:8]
    out = os.path.join(base_dir, f"{ts}_{h}")
    os.makedirs(out, exist_ok=True)
    return out


def commit_config_snapshot(run_dir: str, cfg: BaseModel):
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg.model_dump(), f, indent=2, ensure_ascii=False)


def seed_all(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
