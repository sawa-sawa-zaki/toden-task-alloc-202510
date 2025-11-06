import argparse, json, os, time
from pathlib import Path
import yaml
from src.allocsim.config import SimConfig, load_config_with_includes
from src.allocsim.runners import run_repeats
from src.allocsim.utils import make_run_dir, commit_config_snapshot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/runs/rsd_then_ttc_baseline.yaml")
    args = ap.parse_args()

    cfg_dict = load_config_with_includes(args.config)
    cfg = SimConfig.model_validate(cfg_dict)

    run_dir = make_run_dir(cfg.output_dir, cfg)
    commit_config_snapshot(run_dir, cfg)

    t0 = time.time()
    run_repeats(cfg, run_dir)
    dt = (time.time() - t0) * 1000
    print(f"✅ Done: {run_dir} ({dt:.1f} ms)")


if __name__ == "__main__":
    main()
