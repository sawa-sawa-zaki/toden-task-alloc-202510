#!/usr/bin/env bash
set -euo pipefail

OUT="mre_$(date +%Y%m%d_%H%M%S).tar.gz"
LIST="$(mktemp)"

# ここに“優先的に集めたい”ファイル/ディレクトリを列挙
read -r -d '' CANDIDATES <<'EOF'
pyproject.toml
requirements.txt
requirements-dev.txt
README.md
scripts/ic_check_global.py
scripts/ic_check_time_monotone.py
scripts/run_t3_modeA.py
src/allocsim/__init__.py
src/allocsim/online_rsd_ttc.py
src/allocsim/julia_bridge.py
src/allocsim/report_from_values.py
src/allocsim/strategy/__init__.py
src/allocsim/strategy/time_monotone.py
julia/run_trial.jl
julia/Project.toml
EOF

echo "[INFO] collecting files for MRE..."

# あるものだけ拾う（無いものは警告だけ出してスキップ）
while IFS= read -r rel; do
  [[ -z "$rel" ]] && continue
  if [[ -e "$rel" ]]; then
    echo "$rel" >> "$LIST"
  else
    echo "[WARN] skip (not found): $rel"
  fi
done <<< "$CANDIDATES"

# 念のため src/scripts/julia の存在だけは最低限チェック
if [[ ! -s "$LIST" ]]; then
  echo "[ERROR] nothing to pack (no candidates found)"; exit 1
fi

# アーカイブ作成
tar --exclude-vcs \
    --exclude='**/__pycache__' \
    --exclude='**/.mypy_cache' \
    --exclude='**/.pytest_cache' \
    --exclude='**/.ipynb_checkpoints' \
    --exclude='**/.DS_Store' \
    -czf "$OUT" -T "$LIST"

rm -f "$LIST"
echo "[OK] Created $OUT"
