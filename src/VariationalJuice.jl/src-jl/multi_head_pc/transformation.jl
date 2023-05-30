using StatsBase: percentile
using Random
using Distributions: Normal
using StatsFuns
using ProbabilisticCircuits: PlainInputNode

import ProbabilisticCircuits: foreach, num_parameters # import

function foreach(f::Function, pcs::Vector{T}, ::Nothing = nothing) where T <: ProbCircuit
    foreach(f, pcs, Dict{ProbCircuit,Nothing}())
end

function foreach(f::Function, pcs::Vector{T}, seen::Dict{ProbCircuit,Nothing}) where T <: ProbCircuit
    for pc in pcs
        foreach(f, pc, seen)
    end
end

function foreach(f::Function, pc::ProbCircuit, seen::Dict{ProbCircuit,Nothing})
    get!(seen, pc) do
        if PCs.isinner(pc)
            for c in PCs.children(pc)
                foreach(f, c, seen)
            end
        end
        f(pc)
        nothing
    end
    nothing
end

function foreach_down(f::Function, pcs::Vector{T}) where T <: ProbCircuit
    lin = Vector{ProbCircuit}()
    foreach(pcs) do n
        push!(lin, n)
    end
    foreach(f, Iterators.reverse(lin))
end

import ProbabilisticCircuits: copy # extend

function copy(n::PlainInputNode)
    v = randvar(n)
    d = dist(n)
    if d isa Categorical
        num_cats = PCs.num_categories(d)
        logps = zeros(Float32, num_cats)
        logps .= d.logps
        new_d = Categorical(logps)
    else
        error("Unsupported distribution type.")
    end
    PlainInputNode(v, new_d)
end

function num_parameters(pcs::Vector{T}) where T <: ProbCircuit
    num_params = 0

    foreach(pcs) do n
        if issum(n)
            num_params += length(n.params) - 1
        elseif isinput(n)
            d = dist(n)
            if d isa Categorical
                num_params += length(d.logps) - 1
            else
                error("Unknown input distribution.")
            end
        end
    end

    num_params
end

function perturb(n::PlainInputNode; sigma::AbstractFloat)
    d = dist(n)
    if d isa Categorical
        d.logps .+= log.(clamp.(rand(Normal(1.0, sigma), PCs.num_categories(d)), 0.1, 2.0))
        d.logps .-= StatsFuns.logsumexp(d.logps)
    else
        error("Unsupported distribution type.")
    end
    nothing
end

function perturb(params::Vector{Float32}; sigma::AbstractFloat)
    params = deepcopy(params)
    params .+= log.(clamp.(rand(Normal(1.0, sigma), length(params)), 0.1, 10.0))
    params .-= StatsFuns.logsumexp(params)

    params
end

function grow_heads_by_flows(pcs::Vector{T}, data::CuMatrix, head_mask::CuMatrix; 
                             sigma::AbstractFloat = 0.2, cross_term_discount::AbstractFloat = 0.2,
                             node_selection_method = "percentage", node_selection_args = Dict(),
                             batch_size, soft_reg = 0.0, soft_reg_width = 3, mine = 2, maxe = 32,
                             mhbpc = nothing, mars_mem = nothing, flows_mem = nothing, 
                             aggr_flows_mem = nothing) where T <: ProbCircuit

    if mhbpc === nothing
        mhbpc = CuMultiHeadBitsProbCircuit(pcs)
    end

    n_examples = size(data, 1)
    n_nodes = length(mhbpc.bpc.nodes)
    mars = PCs.prep_memory(mars_mem, (batch_size, n_nodes), (false, true))
    flows = PCs.prep_memory(flows_mem, (batch_size, n_nodes), (false, true))

    aggr_flows_gpu = PCs.prep_memory(aggr_flows_mem, (n_nodes,))
    aggr_flows_gpu .= zero(Float32)

    for batch_start = 1 : batch_size : n_examples
        batch_end = min(batch_start + batch_size - 1, n_examples)
        batch = batch_start : batch_end
        num_batch_examples = batch_end - batch_start + 1

        multi_head_probs_flows_circuit(flows, mars, nothing, mhbpc, data, head_mask, batch; mine, maxe, soft_reg, soft_reg_width)

        @views aggr_flows_gpu .+= sum(flows[1:num_batch_examples,:]; dims = 1)[1,:]
    end

    aggr_flows = Array(aggr_flows_gpu) ./ n_examples
    if node_selection_method == "percentage"
        threshold = max(percentile(aggr_flows, (1.0 - node_selection_args["grow_frac"]) * 100.0), 1e-8)
        node_grow_flag = (aggr_flows .>= threshold)
    else
        error("Unknown node selection method $(node_selection_method).")
    end

    node2id = Dict{ProbCircuit,Int}()
    for idx = 1 : length(mhbpc.bpc.nodes_map)
        node2id[mhbpc.bpc.nodes_map[idx]] = idx
    end

    # Root nodes to split
    split_roots = findall(Array(sum(head_mask; dims = 1) .> 0))

    has_grow_pars = Dict{ProbCircuit,Bool}()
    foreach_down(pcs) do n
        if isinner(n) && ((n in keys(node2id) && node_grow_flag[node2id[n]]) || (n in keys(has_grow_pars) && has_grow_pars[n]))
            for c in children(n)
                has_grow_pars[c] = true
            end
        end
    end

    C = Union{ProbCircuit,Nothing}
    old2new = Dict{ProbCircuit,Tuple{C,C}}()
    foreach(pcs) do n
        get!(old2new, n) do
            if n in keys(node2id) && node_grow_flag[node2id[n]]
                if isinput(n)
                    n1, n2 = copy(n), copy(n)
                    perturb(n1; sigma)
                    perturb(n2; sigma)
                    (n1, n2)
                else
                    ch_ns = Vector{Tuple{C,C}}(undef, num_children(n))
                    map!(c -> old2new[c]::Tuple{C,C}, ch_ns, children(n))
                    if issum(n)
                        chs = Vector{ProbCircuit}()
                        params1, params2 = Vector{Float32}(), Vector{Float32}()
                        for (i, cs) in enumerate(ch_ns)
                            if cs[2] === nothing
                                push!(chs, cs[1])
                                push!(params1, n.params[i])
                                push!(params2, n.params[i])
                            else
                                push!(chs, cs[1], cs[2])
                                push!(params1, n.params[i], n.params[i] .+ log(cross_term_discount))
                                push!(params2, n.params[i] .+ log(cross_term_discount), n.params[i])
                            end
                        end
                        params1 .-= StatsFuns.logsumexp(params1)
                        params2 .-= StatsFuns.logsumexp(params1)
                        n1, n2 = summate(chs...), summate(chs...)
                        n1.params .= perturb(params1; sigma)
                        n2.params .= perturb(params2; sigma)
                        (n1, n2)
                    else
                        @assert ismul(n)
                        chs1, chs2 = Vector{ProbCircuit}(), Vector{ProbCircuit}()
                        for cs in ch_ns
                            push!(chs1, cs[1])
                            if cs[2] === nothing
                                push!(chs2, cs[1])
                            else
                                push!(chs2, cs[2])
                            end
                        end
                        n1, n2 = multiply(chs1...), multiply(chs2...)
                        (n1, n2)
                    end
                end
            else
                if isinput(n)
                    (n, nothing)
                else
                    ch_ns = Vector{Tuple{C,C}}(undef, num_children(n))
                    map!(c -> old2new[c]::Tuple{C,C}, ch_ns, children(n))
                    if issum(n)
                        if all(map(cs -> cs[2] === nothing, ch_ns))
                            n1 = summate(children(n))
                            n1.params .= n.params
                            (n1, nothing)
                        else
                            chs = Vector{ProbCircuit}()
                            if get(has_grow_pars, n, false)
                                params1, params2 = Vector{Float32}(), Vector{Float32}()
                                for (i, cs) in enumerate(ch_ns)
                                    if cs[2] === nothing
                                        push!(chs, cs[1])
                                        push!(params1, n.params[i])
                                        push!(params2, n.params[i])
                                    else
                                        push!(chs, cs[1], cs[2])
                                        push!(params1, n.params[i], n.params[i] .+ log(cross_term_discount))
                                        push!(params2, n.params[i] .+ log(cross_term_discount), n.params[i])
                                    end
                                end
                                params1 .-= StatsFuns.logsumexp(params1)
                                params2 .-= StatsFuns.logsumexp(params2)

                                n1, n2 = summate(chs...), summate(chs...)
                                n1.params .= perturb(params1; sigma)
                                n2.params .= perturb(params2; sigma)
                                (n1, n2)
                            else
                                params = Vector{Float32}()
                                for (i, cs) in enumerate(ch_ns)
                                    if cs[2] === nothing
                                        push!(chs, cs[1])
                                        push!(params, n.params[i])
                                    else
                                        push!(chs, cs[1], cs[2])
                                        rnd = clamp(rand(), 0.1, 0.9)
                                        push!(params, n.params[i] .+ log(rnd), n.params[i] .+ log(1.0 - rnd))
                                    end
                                end
                                params .-= StatsFuns.logsumexp(params)

                                n1 = summate(chs...)
                                n1.params .= perturb(params; sigma)
                                (n1, nothing)
                            end
                        end
                    else
                        @assert ismul(n)
                        chs1, chs2 = Vector{ProbCircuit}(), Vector{ProbCircuit}()
                        for cs in ch_ns
                            push!(chs1, cs[1])
                            if cs[2] === nothing
                                push!(chs2, cs[1])
                            else
                                push!(chs2, cs[2])
                            end
                        end
                        n1, n2 = multiply(chs1...), multiply(chs2...)
                        (n1, n2)
                    end
                end
            end
        end
    end

    PCs.cleanup_memory((mars, mars_mem), (flows, flows_mem), (aggr_flows_gpu, aggr_flows_mem))

    new_pcs = Vector{ProbCircuit}()
    grown_pcs = Vector{ProbCircuit}()
    for pc in pcs
        push!(new_pcs, old2new[pc][1])
        if old2new[pc][2] !== nothing
            push!(grown_pcs, old2new[pc][2])
        end
    end
    vcat(new_pcs, grown_pcs)
end

import ProbabilisticCircuits: init_parameters # extend

function init_parameters(pcs::Vector{T}; perturbation = 0.0) where T <: ProbCircuit
    pc = summate(pcs...)
    init_parameters(pc; perturbation)
    nothing
end