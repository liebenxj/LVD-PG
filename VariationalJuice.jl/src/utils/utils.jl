

function select_gpu(idx)
    device!(collect(devices())[idx+1])
end

function num_categories(mbpc::CuMetaBitsProbCircuit)
    nodes = Array(mbpc.bpc.nodes)
    num_cats = 0
    for n in nodes
        if n isa BitsInput
            ncats = dist(n).num_cats
            if ncats > num_cats
                num_cats = ncats
            end
        end
    end
    num_cats
end
function num_categories(pc::ProbCircuit)
    num_cats = 0
    foreach(pc) do n
        if n isa PlainInputNode
            ncats = num_categories(dist(n))
            if ncats > num_cats
                num_cats = ncats
            end
        end
    end
    num_cats
end

function leaf_params(mbpc::CuMetaBitsProbCircuit, num_vars, n_hiddens; num_cats = 256)
    pc_cat_params = zeros(Float32, num_vars, n_hiddens, num_cats)
    node_idx = ones(Int32, num_vars)
    nodes = Array(mbpc.bpc.nodes)
    heap = Array(mbpc.bpc.heap)
    for node in nodes
        if node isa BitsInput
            v = node.variable
            heap_start = node.dist.heap_start
            pc_cat_params[v, node_idx[v], :] .= heap[heap_start:heap_start+num_cats-1]
            node_idx[v] += 1
        end
    end
    pc_cat_params
end

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