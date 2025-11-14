# src/allocsim/julia_bridge.py
import json, os, subprocess, shutil

class JuliaNotFound(Exception): ...
class JuliaFailed(Exception): ...

def _ensure_julia():
    if not shutil.which("julia"):
        raise JuliaNotFound("julia not found in PATH")

def _to_py_int_list(x):
    # x が np.ndarray / list / tuple のとき、再帰的に Python int/list に落とす
    if isinstance(x, (list, tuple)):
        return [_to_py_int_list(v) for v in x]
    try:
        # numpy型を想定（np.int64 など）
        import numpy as np  # 局所importで依存を軽く
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, np.ndarray):
            return _to_py_int_list(x.tolist())
    except Exception:
        pass
    # 素の int / float / None / str はそのまま
    return x

def _reports_to_py_lists(reports_linear):
    """
    reports_linear: list[list[(m, t1)]]
    → JSON 互換の list[list[[m, t1]]]（タプル→リスト、要素は Python int）
    """
    out = []
    for one in reports_linear:
        out.append([[int(m), int(t1)] for (m, t1) in one])
    return out

def call_julia_trial(cfg, values, reports_linear, rsd_order, shocks, julia_entry="julia/run_trial.jl"):
    """
    cfg: Config(A,M,T,q,barS,buffer,window,shock_p)
    values: np.ndarray (A,M,T)  float
    reports_linear: list[list[(m, t1)]]  # 各iの線形順序（レベル展開済み）
    rsd_order: list[int]  # 1-based
    shocks: list[int]
    """
    _ensure_julia()

    # すべて純Python型に変換
    req = {
        "A": int(cfg.A),
        "M": int(cfg.M),
        "T": int(cfg.T),
        "q": _to_py_int_list(cfg.q),
        "barS": _to_py_int_list(cfg.barS),
        "buffer": _to_py_int_list(cfg.buffer),
        "window": int(cfg.window),
        "p": float(getattr(cfg, "shock_p", 0.0)),
        "values": values.tolist(),                       # 既にPythonのlistに変換
        "reports": _reports_to_py_lists(reports_linear), # [[ [m,t1], ... ], ...]
        "rsd_order": [int(x) for x in rsd_order],
        "shocks": [int(x) for x in shocks],
    }

    env = os.environ.copy()
    cmd = ["julia", "--project=.", julia_entry]
    try:
        p = subprocess.run(
            cmd,
            input=json.dumps(req).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise JuliaFailed(e.stderr.decode("utf-8"))

    out = json.loads(p.stdout.decode("utf-8"))
    return out  # {"utility":[...], "alloc":[...], "residual":[...]}

def linearize_levels(report):
    """
    report: 弱順序のレベル構造 [[(m,t1), ...], [(m,t1), ...], 'OUT', ...]
    'OUT' は無視し、(m,t1) をフラットな線形順序に変換。
    返り値は JSON 直列化しやすいよう **タプルではなく list** にする。
    """
    flat = []
    for level in report:
        if isinstance(level, list):
            for cell in level:
                if isinstance(cell, (list, tuple)) and len(cell) == 2:
                    m, t1 = int(cell[0]), int(cell[1])
                    flat.append([m, t1])  # ← tuple ではなく list
        # 'OUT' は無視
    return flat
