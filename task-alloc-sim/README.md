# Task Allocation Simulator (Multi-Unit, RSD→TTC by timeslot)

- マルチユニット前提（各エージェントは合計 `units_demand` 個の席を希望）
- 各時間帯 t で **RSD** を実行 → 直後に供給変動を想定し **TTC型チェーン**で再調整
- 効用は (machine, timeslot) ごとの **単価の線形結合**。無差別（同率）を許容
- 出力は run ごとのフォルダへ `config.json`, `metrics.csv`, 図（任意）
