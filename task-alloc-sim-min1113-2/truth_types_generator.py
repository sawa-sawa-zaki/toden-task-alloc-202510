# ==============================================
# truth_types_generator_csv.py
# 108種類の truth types を生成して CSV 保存
# ==============================================

import csv
import itertools

def generate_truth_types():
    types = []

    # スロット：6個（A,B）×（1,2,3）
    slots = [
        ("A", 1), ("B", 1),
        ("A", 2), ("B", 2),
        ("A", 3), ("B", 3),
    ]

    # 機械バイアス：A>B, A=B, B>A
    machine_bias_options = ["A>B", "A=B", "B>A"]
    machine_bias_list = list(itertools.product(
        machine_bias_options,
        repeat=3  # t=1,2,3
    ))

    # 時間帯無差別グループのパターン
    time_group_patterns = [
        [[1], [2], [3]],        # 完全順序
        [[1], [2, 3]],          # 2=3
        [[1, 2], [3]],          # 1=2
        [[1, 2, 3]],            # 全無差別
    ]

    truth_id = 1
    for mbias in machine_bias_list:
        for time_group in time_group_patterns:

            ranking = []
            for group in time_group:
                group_slots = []

                for t in group:
                    sA = ("A", t)
                    sB = ("B", t)
                    bias = mbias[t-1]

                    if bias == "A>B":
                        group_slots.extend([sA, sB])
                    elif bias == "B>A":
                        group_slots.extend([sB, sA])
                    else:
                        group_slots.extend([sA, sB])

                ranking.append(group_slots)

            types.append({
                "id": truth_id,
                "machine_bias": mbias,
                "time_groups": time_group,
                "ranking": ranking
            })
            truth_id += 1

    return types


def save_truth_types_csv(filename="truth_types.csv"):
    types = generate_truth_types()

    # CSV 形式で保存
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        # ヘッダ
        writer.writerow([
            "id",
            "machine_bias",
            "time_groups",
            "ranking",
            "flat_rank"  # ランキングを線形に展開
        ])

        for t in types:
            # ランキングを線形に展開
            flat = []
            for group in t["ranking"]:
                for slot in group:
                    flat.append(f"{slot[0]}{slot[1]}")  # 例: A1,B1,A2,...

            writer.writerow([
                t["id"],
                "|".join(t["machine_bias"]),
                str(t["time_groups"]),
                str(t["ranking"]),
                ",".join(flat)
            ])

    print(f"Saved to {filename} (total {len(types)} types)")


if __name__ == "__main__":
    save_truth_types_csv()
