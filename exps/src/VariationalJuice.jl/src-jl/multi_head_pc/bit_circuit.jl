using ProbabilisticCircuits: BitsNode, tag_at


struct CuMultiHeadBitsProbCircuit{BitsNodes <: BitsNode}

    # the original BitPC
    bpc::CuBitsProbCircuit{BitsNodes}

    # ids of the root nodes
    root_ids::CuVector{UInt32}

end

function CuMultiHeadBitsProbCircuit(pcs::Vector{<:ProbCircuit})
    # add a fake root node to maximally reuse existing BitsPC code
    fake_pc = summate(pcs...) 

    bpc = BitsProbCircuit(fake_pc)
    BitsNodes = mapreduce(typeof, (x, y) -> Union{x, y}, bpc.nodes)

    # add a tag to all edges that should be skipped
    fake_root_id = length(bpc.nodes)
    down_edges = bpc.edge_layers_down.vectors
    for edge_id = 1 : length(down_edges)
        edge = down_edges[edge_id]
        if edge isa SumEdge && edge.parent_id == fake_root_id
            tag = tag_at(edge.tag, 4)
            new_edge = SumEdge(
                edge.parent_id, edge.prime_id, edge.sub_id,
                edge.logp, tag
            )
            down_edges[edge_id] = new_edge
        end
    end

    root_ids = zeros(UInt32, length(pcs))
    root_pc_set = Set(pcs)
    filled_count = 0
    for node_id = length(bpc.nodes_map) : -1 : 1
        n = bpc.nodes_map[node_id]
        if n in root_pc_set
            root_id = findall(x->x===n, pcs)[1]
            root_ids[root_id] = node_id
            filled_count += 1
        end
        if filled_count >= length(pcs)
            break
        end
    end
    @assert filled_count == length(pcs)
    
    bpc = cu(bpc)
    root_ids = cu(root_ids)

    CuMultiHeadBitsProbCircuit{BitsNodes}(bpc, root_ids)
end

import ProbabilisticCircuits: update_parameters # extend

function update_parameters(mhbpc::CuMultiHeadBitsProbCircuit)
    bpc = mhbpc.bpc
    nodemap = bpc.nodes_map
    
    # copy parameters from sum nodes
    edges = Vector(bpc.edge_layers_up.vectors)
    i = 1
    while i <= length(edges)
        @assert isfirst(edges[i].tag)
        par_id = edges[i].parent_id
        parent = nodemap[par_id]
        if issum(parent)
            ni = num_inputs(parent)
            if par_id !== length(nodemap)
                params(parent) .= map(e -> e.logp, edges[i:i+ni-1])
            end
        else # parent is a product node
            ni = 1
            while !isfirst(edges[i+ni].tag)
                ni += 1
            end
        end
        i += ni
    end
    
    # copy parameters from input nodes
    nodes = Vector(bpc.nodes)
    input_ids = Vector(bpc.input_node_ids)
    heap = Vector(bpc.heap)
    for i in input_ids
        PCs.update_dist(nodemap[i], nodes[i], heap)
    end
    nothing
end