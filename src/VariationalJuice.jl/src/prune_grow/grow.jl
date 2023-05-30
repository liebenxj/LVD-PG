using Distributions: Uniform
using Random
using StatsFuns


function grow_pc(pc::ProbCircuit; eps::AbstractFloat = 0.2, no_grow_depth_indices::Vector{Int} = [])

    node_depth_cache = get_node_depth(pc)

    f_i(n) = begin
        if node_depth_cache[n] in no_grow_depth_indices
            (n, n)
        else
            v = randvar(n)
            d = dist(n)
            if d isa Categorical
                num_cats = length(d.logps)
                new_d = Categorical(num_cats)
                new_d.logps .= d.logps
                # add noise
                d.logps .+= rand(Uniform(-eps / num_cats, eps / num_cats), (num_cats,))
                new_d.logps .+= rand(Uniform(-eps / num_cats, eps / num_cats), (num_cats,))
                # renormalize
                d.logps .-= StatsFuns.logsumexp(d.logps)
                new_d.logps .-= StatsFuns.logsumexp(new_d.logps)
            else
                error("Unknown input distribution type $(typeof(d))")
            end
            n1 = PlainInputNode(v, d)
            n2 = PlainInputNode(v, new_d)
            (n1, n2)
        end
    end
    f_m(n, ins) = begin
        n1 = multiply(first.(ins))
        n2 = multiply(last.(ins))
        (n1, n2)
    end
    f_s(n, ins) = begin
        if node_depth_cache[n] in no_grow_depth_indices
            orig_params = deepcopy(n.params)
            nparams = length(orig_params)
            chs = cat(first.(ins), last.(ins); dims = 1)
            new_n = summate(chs...)
            new_n.params[1:nparams] .= n.params
            new_n.params[nparams+1:end] .= n.params
            # add noise
            new_n.params .+= rand(Uniform(-eps / nparams, eps / nparams), (2 * nparams,))
            # renormalize
            new_n.params .-= StatsFuns.logsumexp(new_n.params)
            (new_n, new_n)
        else
            orig_params = deepcopy(n.params)
            nparams = length(orig_params)
            chs = cat(first.(ins), last.(ins); dims = 1)
            n1 = summate(chs...)
            n2 = summate(chs...)
            n1.params[1:nparams] .= orig_params
            n1.params[nparams+1:end] .= orig_params
            n2.params[1:nparams] .= orig_params
            n2.params[nparams+1:end] .= orig_params
            # add noise
            n1.params .+= rand(Uniform(-eps / nparams, eps / nparams), (2 * nparams,))
            n2.params .+= rand(Uniform(-eps / nparams, eps / nparams), (2 * nparams,))
            # renormalize
            n1.params .-= StatsFuns.logsumexp(n1.params)
            n2.params .-= StatsFuns.logsumexp(n2.params)
            (n1, n2)
        end
    end
    pc1, pc2 = foldup_aggregate(pc, f_i, f_m, f_s, Tuple{ProbCircuit, ProbCircuit})

    if issum(pc1)
        chs = cat(inputs(pc1), inputs(pc2); dims = 1)
        pc = summate(chs...)
        pc.params[1:num_inputs(pc1)] .= pc1.params
        pc.params[num_inputs(pc1)+1:end] .= pc2.params
    else
        error("Not implemented")
    end

    pc
end

function per_node_grow_pc(pc::ProbCircuit; eps::AbstractFloat = 0.2, nodes_not_to_grow::Set{ProbCircuit} = Set{ProbCircuit}())

    f_i(n) = begin
        if n in nodes_not_to_grow
            (n, n)
        else
            v = randvar(n)
            d = dist(n)
            if d isa Categorical
                num_cats = length(d.logps)
                new_d = Categorical(num_cats)
                new_d.logps .= d.logps
                # add noise
                d.logps .+= rand(Uniform(-eps / num_cats, eps / num_cats), (num_cats,))
                new_d.logps .+= rand(Uniform(-eps / num_cats, eps / num_cats), (num_cats,))
                # renormalize
                d.logps .-= StatsFuns.logsumexp(d.logps)
                new_d.logps .-= StatsFuns.logsumexp(new_d.logps)
            else
                error("Unknown input distribution type $(typeof(d))")
            end
            n1 = PlainInputNode(v, d)
            n2 = PlainInputNode(v, new_d)
            (n1, n2)
        end
    end
    f_m(n, ins) = begin
        n1 = multiply(first.(ins))
        n2 = multiply(last.(ins))
        (n1, n2)
    end
    f_s(n, ins) = begin
        if n in nodes_not_to_grow
            orig_params = deepcopy(n.params)
            nparams = length(orig_params)
            chs = cat(first.(ins), last.(ins); dims = 1)
            new_n = summate(chs...)
            new_n.params[1:nparams] .= n.params
            new_n.params[nparams+1:end] .= n.params
            # add noise
            new_n.params .+= rand(Uniform(-eps / nparams, eps / nparams), (2 * nparams,))
            # renormalize
            new_n.params .-= StatsFuns.logsumexp(new_n.params)
            (new_n, new_n)
        else
            orig_params = deepcopy(n.params)
            nparams = length(orig_params)
            chs = cat(first.(ins), last.(ins); dims = 1)
            n1 = summate(chs...)
            n2 = summate(chs...)
            n1.params[1:nparams] .= orig_params
            n1.params[nparams+1:end] .= orig_params
            n2.params[1:nparams] .= orig_params
            n2.params[nparams+1:end] .= orig_params
            # add noise
            n1.params .+= rand(Uniform(-eps / nparams, eps / nparams), (2 * nparams,))
            n2.params .+= rand(Uniform(-eps / nparams, eps / nparams), (2 * nparams,))
            # renormalize
            n1.params .-= StatsFuns.logsumexp(n1.params)
            n2.params .-= StatsFuns.logsumexp(n2.params)
            (n1, n2)
        end
    end
    pc1, pc2 = foldup_aggregate(pc, f_i, f_m, f_s, Tuple{ProbCircuit, ProbCircuit})

    if issum(pc1)
        chs = cat(inputs(pc1), inputs(pc2); dims = 1)
        pc = summate(chs...)
        pc.params[1:num_inputs(pc1)] .= pc1.params
        pc.params[num_inputs(pc1)+1:end] .= pc2.params
    else
        error("Not implemented")
    end

    pc
end

function get_pc_depth(pc::ProbCircuit)
    max_depth = 1

    f_i(_) = 1
    f_m(_, ins) = begin
        depth = maximum(ins)
        if depth > max_depth
            max_depth = depth
        end
        depth
    end
    f_s(_, ins) = begin
        depth = maximum(ins) + 1
        if depth > max_depth
            max_depth = depth
        end
        depth
    end

    foldup_aggregate(pc, f_i, f_m, f_s, Int)

    max_depth
end

function get_node_depth(pc::ProbCircuit)
    cache = Dict{ProbCircuit,Int}()

    f_i(_) = 1
    f_m(_, ins) = maximum(ins)
    f_s(_, ins) = maximum(ins) + 1

    foldup_aggregate(pc, f_i, f_m, f_s, Int, cache)

    cache
end