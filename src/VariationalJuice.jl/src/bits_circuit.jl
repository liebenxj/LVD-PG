using ProbabilisticCircuits: BitsNode, islast, dist, SumEdge
using ProbabilisticCircuits: CuBitsProbCircuit


struct CuMetaBitsProbCircuit{BitsNodes <: BitsNode}

    # the original BitPC
    bpc::CuBitsProbCircuit{BitsNodes}

    # mapping from edge id (w.r.t. upward edges) to param id
    edge2param::CuVector{UInt32}

    # mapping from parent id to edge id (w.r.t. upward edges)
    param2edge::CuVector{UInt32}

    # map params to their respective normalized groups
    param2group::CuVector{UInt32}

    # size of each parameter group
    pargroup_sizes::CuVector{UInt32}

    # number of parameter groups
    num_param_groups::UInt32

    # input node id to its number of cumulative parameters
    innode2ncumparam::CuVector{UInt32}

    # total number of input parameters
    num_input_params::UInt32

    # mapping from upward edge to downward edge
    up2downedge::CuVector{UInt32}

    # node id to its input node id (set to 0 for inner nodes)
    node2inputid::CuVector{UInt32}

end

function CuMetaBitsProbCircuit(pc::ProbCircuit)
    bpc = BitsProbCircuit(pc)
    BitsNodes = mapreduce(typeof, (x, y) -> Union{x, y}, bpc.nodes)
    num_edges = size(bpc.edge_layers_up)[1]

    # mappings between edge id and param id
    edge2param = Vector{UInt32}(undef, num_edges)
    param2edge = UInt32[]
    param2group = UInt32[]
    pargroup_sizes = UInt32[]
    param_id = 1
    param_group = 1
    ch_num = 0
    for edge_id = 1 : num_edges
        edge = bpc.edge_layers_up[edge_id]
        if edge isa SumEdge
            if isfirst(edge.tag)
                ch_num = 0
            end
            edge2param[edge_id] = param_id
            push!(param2edge, edge_id)
            push!(param2group, param_group)
            param_id += 1
            ch_num += 1
            if islast(edge.tag)
                param_group += 1
                push!(pargroup_sizes, ch_num)
            end
        else
            edge2param[edge_id] = zero(UInt32)
        end
    end
    num_param_groups = param_group - 1

    @assert length(pargroup_sizes) == num_param_groups

    # input nodes
    num_input_nodes = length(bpc.input_node_ids)
    innode2nparam = Vector{UInt32}(undef, num_input_nodes)
    for idx = 1 : num_input_nodes
        node_id = bpc.input_node_ids[idx]
        node = bpc.nodes[node_id]
        innode2nparam[idx] = num_parameters(dist(node))
    end
    num_input_params = sum(innode2nparam)
    innode2ncumparam = cumsum(innode2nparam)

    # edge mapping
    up2downedge = Vector{UInt32}(undef, num_edges)
    for downedge_id = 1 : num_edges
        upedge_id = bpc.down2upedge[downedge_id]
        up2downedge[upedge_id] = downedge_id
    end

    # node2inputid
    num_nodes = length(bpc.nodes)
    node2inputid = Vector{UInt32}(undef, num_nodes)
    @inbounds node2inputid .= zero(UInt32)
    for (idx, input_idx) in enumerate(bpc.input_node_ids)
        @inbounds node2inputid[input_idx] = idx
    end

    # move to GPU
    bpc = cu(bpc)
    edge2param = cu(edge2param)
    param2edge = cu(param2edge)
    param2group = cu(param2group)
    pargroup_sizes = cu(pargroup_sizes)
    innode2ncumparam = cu(innode2ncumparam)
    up2downedge = cu(up2downedge)
    node2inputid = cu(node2inputid)

    CuMetaBitsProbCircuit{BitsNodes}(bpc, edge2param, param2edge, param2group, pargroup_sizes, num_param_groups,
                                     innode2ncumparam, num_input_params, up2downedge, node2inputid)
end

############
## Methods
############

num_parameters(mbpc::CuMetaBitsProbCircuit) = 
    length(mbpc.param2edge) + mbpc.num_input_params

import ProbabilisticCircuits: num_nodes, num_edges # extend

num_nodes(mbpc::CuMetaBitsProbCircuit) = length(mbpc.bpc.nodes)

num_edges(mbpc::CuMetaBitsProbCircuit) = length(mbpc.bpc.edge_layers_up.vectors)

num_input_params(mbpc::CuMetaBitsProbCircuit) = mbpc.num_input_params

num_inner_params(mbpc::CuMetaBitsProbCircuit) = 
    num_parameters(mbpc) - num_input_params(mbpc)

function mark_nodes(mbpc::CuMetaBitsProbCircuit, marked_pcs::Matrix{<:ProbCircuit})
    nodes_map = mbpc.bpc.nodes_map

    pc_dict = Dict{ProbCircuit,Tuple{Int,Int}}()
    for i = 1 : size(marked_pcs, 1)
        for j = 1 : size(marked_pcs, 2)
            n = marked_pcs[i,j]
            if issum(n) && num_inputs(n) == 1
                n = n.inputs[1] # the original node will not be materialized
            end
            pc_dict[n] = (i, j)
        end
    end

    marked_pc_idx1 = zeros(UInt32, length(nodes_map))
    marked_pc_idx2 = zeros(UInt32, length(nodes_map))
    count = 0
    for i = 1 : length(nodes_map)
        n = nodes_map[i]
        if haskey(pc_dict, n)
            count += 1
            marked_pc_idx1[i] = pc_dict[n][1]
            marked_pc_idx2[i] = pc_dict[n][2]
        end
    end

    if count < size(marked_pcs, 1) * size(marked_pcs, 2)
        println("Some nodes not materialized by the PC")
    end

    marked_pc_idx1, marked_pc_idx2
end