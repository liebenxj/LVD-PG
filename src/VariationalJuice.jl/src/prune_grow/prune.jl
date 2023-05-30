

function prune_pc(pc::ProbCircuit, data::CuArray; batch_size, keep_frac,
                  mars_mem = nothing, flows_mem = nothing, edge_aggr_mem = nothing)
    bpc = CuBitsProbCircuit(pc)

    num_examples = size(data, 1)
    num_nodes = length(bpc.nodes)
    num_edges = length(bpc.edge_layers_down.vectors)
    num_batches = num_examples ÷ batch_size

    mars = prep_memory(mars_mem, (batch_size, num_nodes), (false, true))
    flows = prep_memory(flows_mem, (batch_size, num_nodes), (false, true))
    edge_aggr = prep_memory(edge_aggr_mem, (num_edges,))
    edge_aggr .= zero(Float32)

    for batch_id = 1 : num_batches
        batch = (batch_id - 1) * batch_size + 1 : batch_id * batch_size

        probs_flows_circuit_with_reg(flows, mars, edge_aggr, bpc, data, batch; 
                                     mine = 2, maxe = 32, soft_reg = 0.0, soft_reg_width = 3)
    end

    CUDA.unsafe_free!(mars)
    CUDA.unsafe_free!(flows)

    num_kept_edges = Int(round(num_edges * keep_frac))

    edge_aggr = Array(edge_aggr)
    edge_keep_indices = partialsortperm(edge_aggr, 1:num_kept_edges, rev = true)
    edges_up = Array(bpc.edge_layers_up.vectors)
    down2upedge = Array(bpc.down2upedge)
    nodes = Array(bpc.nodes)
    nodes_map = bpc.nodes_map

    get_edge_count(edge_id) = begin
        edge_count = 1
        while !isfirst(edges_up[edge_id].tag)
            edge_count += 1
            edge_id -= 1
        end
        edge_count
    end

    kept_node_map = Dict{ProbCircuit,BitSet}()
    for edge_id in edge_keep_indices
        up_edge_id = down2upedge[edge_id]
        edge = edges_up[up_edge_id]
        edge_count = get_edge_count(up_edge_id)
        node_id = edge.parent_id
        node = nodes_map[node_id]
        if haskey(kept_node_map, node)
            push!(kept_node_map[node], edge_count)
        else
            kept_node_map[node] = BitSet(edge_count)
        end
    end

    f_i(n) = PlainInputNode(randvar(n), dist(n))
    f_m(_, ins) = multiply(ins...)
    f_s(n, ins) = begin
        chs = Vector()
        params = Vector{Float32}()
        if haskey(kept_node_map, n)
            for ch_idx = 1 : length(ins)
                if ch_idx in kept_node_map[n]
                    push!(chs, ins[ch_idx])
                    push!(params, n.params[ch_idx])
                end
            end
        end
        if length(chs) == 0
            ch_idx = argmax(n.params)
            push!(chs, ins[ch_idx])
            push!(params, n.params[ch_idx])
        end
        params .-= StatsFuns.logsumexp(params)
        new_n = summate(chs...)
        new_n.params .= params
        new_n
    end
    foldup_aggregate(pc, f_i, f_m, f_s, ProbCircuit)
end