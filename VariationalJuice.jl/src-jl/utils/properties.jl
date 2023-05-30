
function get_randvars(pc::ProbCircuit)
    cache = Dict{ProbCircuit,BitSet}()
    f_i(n) = BitSet([randvar(n)])
    f_m(_, ins) = union(ins...)
    f_s(_, ins) = union(ins...)
    foldup_aggregate(pc, f_i, f_m, f_s, BitSet, cache)
    cache
end

function issmooth(pc::ProbCircuit)
    flag = true
    cache = get_randvars(pc)
    foreach(pc) do n
        if issum(n)
            for i = 2 : num_inputs(n) 
                if !issetequal(cache[n.inputs[i]], cache[n.inputs[1]])
                    flag = false
                end
            end
        end
    end
    flag
end

function isdecomposable(pc::ProbCircuit)
    flag = true
    cache = get_randvars(pc)
    foreach(pc) do n
        if ismul(n)
            for i = 2 : num_inputs(n) 
                if length(intersect(cache[n.inputs[i]], cache[n.inputs[1]])) >= 1
                    flag = false
                end
            end
        end
    end
    flag
end

function isvalid(pc::ProbCircuit)
    flag = true
    foreach(pc) do n
        if issum(n)
            if abs(sum(exp.(n.params)) - 1.0) > 1e-4
                flag = false
            end
        elseif isinput(n)
            d = dist(n)
            if d isa Categorical
                if abs(sum(exp.(d.logps)) - 1.0) > 1e-4
                    flag = false
                end
            end
        end
    end
    flag
end