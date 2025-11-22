import os
import sys
import pandas as pd
import numpy as np
import argparse

# srcフォルダからインポート
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.config import Config
from src.generator import (
    generate_time_monotone_prefs, 
    generate_all_permutations, 
    generate_all_weak_orders
)

def ranks_to_string(ranks, T=3, M=2):
    """
    ランクのタプルを人間が読める文字列 (A1 > B1 = A2...) に変換
    ranks: (rA1, rB1, rA2, rB2, rA3, rB3)
    """
    # スロットとランクのペアリスト作成
    slots = []
    idx = 0
    machines = ["A", "B"] # M=2前提
    for t in range(T):
        for m in range(M):
            slot_name = f"{machines[m]}{t}" # A0, B0, ...
            rank = ranks[idx]
            slots.append((rank, slot_name))
            idx += 1
    
    # ランク順にソート
    slots.sort(key=lambda x: x[0])
    
    # 文字列構築
    res = ""
    for i, (r, name) in enumerate(slots):
        if i > 0:
            prev_r = slots[i-1][0]
            if r == prev_r:
                res += " = "
            else:
                res += " > "
        res += name
    return res

def analyze_strategy_change(true_ranks, lie_ranks):
    """
    真のランクと嘘のランクを比較して、どのような戦略的変更があったかを言語化する
    """
    analysis = []
    
    # 1. トップのすげ替え確認
    true_top = [i for i, r in enumerate(true_ranks) if r == min(true_ranks)]
    lie_top = [i for i, r in enumerate(lie_ranks) if r == min(lie_ranks)]
    
    if true_top != lie_top:
        analysis.append("【トップの変更】一番欲しいスロットを偽りました。")
        
    # 2. 時間選好の変化 (安全志向か？)
    # 前半(T0, T1)と後半(T2)の平均ランクを比較
    # ランクは値が小さいほど偉いので、平均値が上がれば「評価を下げた」ことになる
    # index: 0,1(T0), 2,3(T1), 4,5(T2)
    
    def get_avg_rank(r_tuple, indices):
        return sum(r_tuple[i] for i in indices) / len(indices)
    
    true_early = get_avg_rank(true_ranks, [0, 1])
    lie_early = get_avg_rank(lie_ranks, [0, 1])
    
    true_late = get_avg_rank(true_ranks, [4, 5])
    lie_late = get_avg_rank(lie_ranks, [4, 5])
    
    if lie_early > true_early + 0.5 and lie_late < true_late - 0.5:
        analysis.append("【リスク回避】早い時間を嫌い、遅い時間を好むふりをしました（安全資産への退避）。")
    elif lie_early < true_early - 0.5 and lie_late > true_late + 0.5:
        analysis.append("【リスク愛好】遅い時間を嫌い、早い時間を好むふりをしました（強気な確保）。")

    # 3. マシンのこだわり
    # A(偶数idx) と B(奇数idx) の評価差
    true_a_pref = sum(true_ranks[i] for i in [0, 2, 4])
    true_b_pref = sum(true_ranks[i] for i in [1, 3, 5])
    
    lie_a_pref = sum(lie_ranks[i] for i in [0, 2, 4])
    lie_b_pref = sum(lie_ranks[i] for i in [1, 3, 5])
    
    # 値が小さい方が好き
    true_likes_A = true_a_pref < true_b_pref
    lie_likes_A = lie_a_pref < lie_b_pref
    
    if true_likes_A != lie_likes_A:
        analysis.append("【マシンの偽装】好みの計算機(A/B)を逆にして申告しました。")

    # 4. 無差別の偽装
    true_unique = len(set(true_ranks))
    lie_unique = len(set(lie_ranks))
    
    if lie_unique < true_unique:
        analysis.append("【無差別の装い】実際より「どっちでもいい」スロットを増やしました（柔軟性の演出）。")
    elif lie_unique > true_unique:
        analysis.append("【こだわりの演出】実際はどっちでもいいのに、順位に差をつけました（厳格化）。")

    if not analysis:
        analysis.append("【微調整】全体的な順序を少し入れ替えましたが、大きな特性変化は見られません（微妙な調整で確率を操作）。")
        
    return "\n".join(analysis)

def main():
    # 最新の結果フォルダを探す
    base_dir = "results"
    if not os.path.exists(base_dir):
        print("Results directory not found.")
        return

    # フォルダを日付順にソートして最新を取得
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not dirs:
        print("No result data found.")
        return
    dirs.sort(reverse=True)
    latest_dir = os.path.join(base_dir, dirs[0])
    
    target_file = os.path.join(latest_dir, "ic_violations_only.csv")
    if not os.path.exists(target_file):
        print(f"No violations file found in {latest_dir}")
        return

    print(f"📂 Analyzing: {target_file}")
    df = pd.read_csv(target_file)
    
    if len(df) == 0:
        print("✅ No violations found in the CSV. (Strategy-Proofness holds!)")
        return

    # Configの読み込み（戦略空間の特定のため）
    # ログファイルから簡易的に読み取るか、デフォルトConfigを使う
    # ここでは簡易的に、Generatorを再実行してインデックスを復元する
    cfg = Config()
    
    # ログからSTRATEGY_DOMAINを推定（または固定）
    # 本当はconfig_log.txtを読むべきだが、ここでは直近のmain.pyの設定と仮定
    # もしずれる場合は手動指定してください
    
    # 1. 選好リストの再生成（インデックス復元用）
    print("Generating preference maps...")
    true_types = generate_time_monotone_prefs(cfg.M, cfg.T)
    
    # Strategy Domainの判定（簡易ロジック: インデックスが108を超えていればUnrestricted）
    max_idx = df["best_lie_idx"].max()
    if max_idx >= len(true_types):
        print("Detected UNRESTRICTED domain.")
        strategy_types = generate_all_permutations(cfg.M, cfg.T)
    else:
        print("Assuming RESTRICTED domain.")
        strategy_types = true_types

    print(f"Found {len(df)} violation cases. Showing top 5 distinct patterns...\n")
    
    # Regret（利得差）が大きい順にソート
    df = df.sort_values("regret", ascending=False)
    
    count = 0
    for _, row in df.iterrows():
        if count >= 5: break # 上位5件のみ表示
        
        agent_id = row["agent"] # "A" or "B" or 0 or 1
        true_idx = int(row["true_type_A"] if agent_id == "A" or agent_id == 0 else row["true_type_B"])
        lie_idx = int(row["best_lie_idx"])
        regret = row["regret"]
        
        # 配列外参照ガード
        if true_idx >= len(true_types) or lie_idx >= len(strategy_types):
            continue

        true_rank = true_types[true_idx]['ranks']
        lie_rank = strategy_types[lie_idx]['ranks']
        
        print("="*60)
        print(f"🚨 VIOLATION CASE #{count+1} (Gain: {regret:.4f})")
        print(f"   Agent: {agent_id}")
        print("-" * 60)
        print(f"💖 [True Preference] (Type {true_idx})")
        print(f"   {ranks_to_string(true_rank)}")
        print("-" * 60)
        print(f"🤥 [Strategic Lie]   (Type {lie_idx})")
        print(f"   {ranks_to_string(lie_rank)}")
        print("-" * 60)
        print("🧐 [Analysis]")
        print(analyze_strategy_change(true_rank, lie_rank))
        print("\n")
        
        count += 1

    print("Analysis Complete.")

if __name__ == "__main__":
    main()