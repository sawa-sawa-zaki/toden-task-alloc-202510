# scripts/run_t3_modeA.py

import os, sys, subprocess, random
from datetime import datetime

def make_unique_outdir(base="results", prefix="global_T3_A"):
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    rnd = random.randint(10000, 99999)
    outdir = os.path.join(base, f"{prefix}_{ts}_{rnd}")
    os.makedirs(outdir, exist_ok=True)
    return outdir

def main():
    A = 2
    M = 2
    T = 3
    q = ["1","1"]
    barS = ["1","1","1"]
    buffer = ["0","0","0"]
    window = "2"
    p = "0.2"
    samples = "300"
    seed = "0"

    # ★ 追加：T=3 用の rank_order（6マスぶんを高→低で列挙）
    rank_order = "(0,1),(1,1),(0,2),(1,2),(0,3),(1,3)"
    ties = ""  # 同価なしなら空でOK

    outdir = make_unique_outdir()

    cmd = [
        sys.executable, "scripts/ic_check_global.py",
        "--A", str(A), "--M", str(M), "--T", str(T),
        "--q", *q,
        "--barS", *barS,
        "--buffer", *buffer,
        "--window", window,
        "--p", p,
        "--samples", samples,
        "--seed", seed,
        "--out_dir", outdir,
        # ★ 追加：rank_order / ties を明示
        "--rank_order", rank_order,
        "--ties", ties,
    ]

    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(rc)
    print("\nDone.")
    print("Output dir :", outdir)

if __name__ == "__main__":
    main()
