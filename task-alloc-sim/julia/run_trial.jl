#!/usr/bin/env julia
using JSON3
include("src/online_rsd_ttc_core.jl")
using .OnlineRSDTTCCore

function main()
    req = JSON3.read(read(stdin, String))
    cfg = OnlineRSDTTCCore.Config(
        Int(req["A"]), Int(req["M"]), Int(req["T"]),
        Vector{Int}(req["q"]),
        Vector{Int}(req["barS"]),
        Vector{Int}(req["buffer"]),
        Int(req["window"]),
        Float64(get(req, "p", 0.0))
    )
    A,M,T = cfg.A, cfg.M, cfg.T

    # values (A,M,T)
    values = Array{Float64,3}(undef, A,M,T)
    for i in 1:A, m in 1:M, t in 1:T
        values[i,m,t] = Float64(req["values"][i][m][t])
    end

    # reports: 各iの線形順序 [(m,t1),(m,t1),...]
    reports = Vector{Vector{Tuple{Int,Int}}}(undef, A)
    for i in 1:A
        lst = Vector{Tuple{Int,Int}}()
        for cell in req["reports"][i]
            push!(lst, (Int(cell[1]), Int(cell[2])))
        end
        reports[i] = lst
    end

    rsd_order = Vector{Int}(req["rsd_order"])
    shocks    = Vector{Int}(req["shocks"])

    alloc    = zeros(Int, A,M,T)
    residual = zeros(Int, T)

    OnlineRSDTTCCore.run_one_trial!(cfg, values, reports, rsd_order, shocks, alloc, residual)
    u = OnlineRSDTTCCore.utility(values, alloc)

    out = Dict(
        "utility" => u,
        "alloc"   => alloc,
        "residual"=> residual
    )
    print(JSON3.write(out))
end

main()
