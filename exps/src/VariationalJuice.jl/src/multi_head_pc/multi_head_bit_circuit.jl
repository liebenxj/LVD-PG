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
    for node_id = 1 : length(bpc.nodes_map)
        n = bpc.nodes_map[node_id]
        if n in root_pc_set
            root_id = findall(x->x===n, pcs)[1]
            root_ids[root_id] = node_id
        end
    end

    bpc = cu(bpc)
    root_ids = cu(root_ids)

    CuMultiHeadBitsProbCircuit{BitsNodes}(bpc, root_ids)
end