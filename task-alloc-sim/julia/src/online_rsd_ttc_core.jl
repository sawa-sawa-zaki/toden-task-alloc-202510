# julia/src/online_rsd_ttc_core.jl
module OnlineRSDTTCCore
using Random

# --- 型 ---
struct Config
    A::Int; M::Int; T::Int
    q::Vector{Int}         # 各エージェントの需要（合計本数）
    barS::Vector{Int}      # 期待供給 (t=1..T)
    buffer::Vector{Int}    # バッファ (t=1..T)
    window::Int            # タイムウィンドウ幅
    p::Float64             # （未使用でも保持）ショック強度のメタ情報
end

# reports[i] は (m,t1) の線形順序（レベルはPython側で展開済み）
# values[i,m,t] は効用単価
function run_one_trial!(
    cfg::Config,
    values::Array{Float64,3},              # (A,M,T)
    reports::Vector{Vector{Tuple{Int,Int}}},# 各iの線形順序（(m, t1)）
    rsd_order::Vector{Int},                 # RSDの順番（エージェント番号 1..A）
    shocks::Vector{Int},                    # 各tのショック（-1/0/1など）
    alloc::Array{Int,3},                    # (A,M,T) 出力：割当 0/1
    residual::Vector{Int}                   # (t) 出力：各tの残余供給
)::Nothing
    A,M,T = cfg.A, cfg.M, cfg.T
    @inbounds for t in 1:T
        residual[t] = cfg.barS[t]
    end
    fill!(alloc, 0)

    # --- RSD：順に欲しい (m,t1) を取りに行く。window=2 などの詳細は
    #  現仕様では Python 側で報告順序に折り込む前提（= window内の候補を先に並べる）
    demand_left = copy(cfg.q)
    @inbounds for i in rsd_order
        for (m,t1) in reports[i]
            if demand_left[i] <= 0; break; end
            if residual[t1] > 0
                alloc[i,m,t1] = 1
                residual[t1] -= 1
                demand_left[i] -= 1
            end
        end
    end

    # --- 供給実現 & TTC（簡易版） ---
    # 実現供給 = barS[t] + shocks[t]
    # まず割当数と比較して差分だけを処理（+なら繰上げ、-なら追い出し）
    @inbounds for t in 1:T
        realized = cfg.barS[t] + shocks[t]
        assigned = 0
        for i in 1:A, m in 1:M
            assigned += alloc[i,m,t]
        end
        diff = realized - assigned

        if diff > 0
            # 繰上げ diff 本ぶん：そのtで未割当の中から values 最大の（i,m）を付与
            for _ in 1:diff
                best_i = 0; best_m = 0; best_v = -Inf
                for i in 1:A
                    has = false
                    @inbounds for m in 1:M
                        if alloc[i,m,t] == 1; has = true; break; end
                    end
                    if !has
                        @inbounds for m in 1:M
                            v = values[i,m,t]
                            if v > best_v
                                best_v = v; best_i = i; best_m = m
                            end
                        end
                    end
                end
                if best_i != 0
                    alloc[best_i,best_m,t] = 1
                end
            end
        elseif diff < 0
            # 追い出し -diff 本ぶん：そのtの現割当から values 最小の（i,m）を外す
            for _ in 1:(-diff)
                worst_i = 0; worst_m = 0; worst_v = Inf
                @inbounds for i in 1:A, m in 1:M
                    if alloc[i,m,t] == 1
                        v = values[i,m,t]
                        if v < worst_v
                            worst_v = v; worst_i = i; worst_m = m
                        end
                    end
                end
                if worst_i != 0
                    alloc[worst_i,worst_m,t] = 0
                end
            end
        end
    end
    return nothing
end

function utility(values::Array{Float64,3}, alloc::Array{Int,3})
    A,M,T = size(values)
    u = zeros(Float64, A)
    @inbounds for i in 1:A, m in 1:M, t in 1:T
        if alloc[i,m,t] == 1
            u[i] += values[i,m,t]
        end
    end
    return u
end

end # module
