using ProbabilisticCircuits: isfirst, islast


"Copy parameters from `param2` (w.r.t. mbpc2) to `param1` (w.r.t. mbpc1)"
function map_pc_parameters(mbpc1::CuMetaBitsProbCircuit, mbpc2::CuMetaBitsProbCircuit, params2; 
                           param_mapping = nothing, to_cpu = true, pc2topc1 = nothing)
    params1 = zeros(Float32, size(params2, 1), num_parameters(mbpc1))
    params1 = cu(params1)
    params2 = cu(params2)
    map_pc_parameters(mbpc1, mbpc2, params1, params2; param_mapping, to_cpu, pc2topc1)
end
function map_pc_parameters(mbpc1::CuMetaBitsProbCircuit, mbpc2::CuMetaBitsProbCircuit, params1, params2; 
                           param_mapping = nothing, to_cpu = true, pc2topc1 = nothing)
    # compute the mapping once (on CPU)
    if param_mapping === nothing
        matchparamids1, matchparamids2 = compute_mbpc_mapping(mbpc1, mbpc2, pc2topc1)
    else
        matchparamids1, matchparamids2 = param_mapping
    end

    if matchparamids1 isa Vector
        matchparamids1 = cu(matchparamids1)
        matchparamids2 = cu(matchparamids2)
    end

    # apply the mapping

    num_examples = size(params1, 1)

    dummy_args = (params1, params2, matchparamids1, matchparamids2, num_examples, Int32(1), Int32(1))
    kernel = @cuda name="map_pc_parameters" launch=false map_pc_parameters_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)
    
    threads, blocks, num_example_threads, param_work = 
        balance_threads(length(matchparamids1), num_examples, config; mine = 2, maxe = 32)
    
    args = (params1, params2, matchparamids1, matchparamids2, num_examples, Int32(num_example_threads), Int32(param_work))
    kernel(args...; threads, blocks)
    
    if to_cpu
        Array(params1)
    else
        nothing
    end
end

function compute_mbpc_mapping(mbpc1::CuMetaBitsProbCircuit, mbpc2::CuMetaBitsProbCircuit, pc2topc1)
    @assert pc2topc1 !== nothing

    nodesmap1 = mbpc1.bpc.nodes_map
    nodesdict1 = Dict{ProbCircuit,Int}()
    for idx = 1 : length(nodesmap1)
        nodesdict1[nodesmap1[idx]] = idx
    end
    nodesmap2 = mbpc2.bpc.nodes_map
    edges1 = Array(mbpc1.bpc.edge_layers_up.vectors)
    edges2 = Array(mbpc2.bpc.edge_layers_up.vectors)
    edge2param1 = Array(mbpc1.edge2param)
    edge2param2 = Array(mbpc2.edge2param)

    matchnodeidx1 = Vector{UInt32}()
    matchnodeidx2 = Vector{UInt32}()
    for idx = 1 : length(nodesmap2)
        node2 = nodesmap2[idx]
        if issum(node2) && haskey(pc2topc1, node2)
            node1 = pc2topc1[node2]
            if haskey(nodesdict1, node1)
                push!(matchnodeidx1, nodesdict1[node1])
                push!(matchnodeidx2, idx)
            end
        end
    end
    
    matchnodedict1 = Dict{Int,Int}()
    for idx = 1 : length(matchnodeidx1)
        matchnodedict1[matchnodeidx1[idx]] = idx
    end
    edge_start_ids1 = zeros(UInt32, length(matchnodeidx1))
    for edge_idx = 1 : length(edges1)
        edge = edges1[edge_idx]
        if edge isa SumEdge
            pid = edge.parent_id
            tag = edge.tag
            if isfirst(tag) && haskey(matchnodedict1, pid)
                edge_start_ids1[matchnodedict1[pid]] = edge_idx
            end
        end
    end
    
    matchnodedict2 = Dict{Int,Int}()
    for idx = 1 : length(matchnodeidx2)
        matchnodedict2[matchnodeidx2[idx]] = idx
    end
    edge_start_ids2 = zeros(UInt32, length(matchnodeidx2))
    for edge_idx = 1 : length(edges2)
        edge = edges2[edge_idx]
        if edge isa SumEdge
            pid = edge.parent_id
            tag = edge.tag
            if isfirst(tag) && haskey(matchnodedict2, pid)
                edge_start_ids2[matchnodedict2[pid]] = edge_idx
            end
        end
    end
    
    matchparamids1 = Vector{UInt32}()
    matchparamids2 = Vector{UInt32}()
    for idx = 1 : length(matchnodeidx1)
        edge_id1 = edge_start_ids1[idx]
        edge_id2 = edge_start_ids2[idx]
        while true
            edge1 = edges1[edge_id1]
            param_id1 = edge2param1[edge_id1]
            param_id2 = edge2param2[edge_id2]
            push!(matchparamids1, param_id1)
            push!(matchparamids2, param_id2)
            if islast(edge1.tag)
                break
            end
            edge_id1 += 1
            edge_id2 += 1
        end
    end

    matchparamids1, matchparamids2
end

function map_pc_parameters_kernel(params1, params2, matchparamids1, matchparamids2, num_examples, num_ex_threads::Int32, param_work::Int32)

    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    param_batch, ex_id = fldmod1(threadid, num_ex_threads)

    param_start = one(Int32) + (param_batch - one(Int32)) * param_work
    param_end = min(param_start + param_work - one(Int32), length(matchparamids1))

    @inbounds if ex_id <= num_examples
        for idx = param_start : param_end
            par_id1 = matchparamids1[idx]
            par_id2 = matchparamids2[idx]
            params1[ex_id, par_id1] = params2[ex_id, par_id2]
        end
    end

    nothing
end