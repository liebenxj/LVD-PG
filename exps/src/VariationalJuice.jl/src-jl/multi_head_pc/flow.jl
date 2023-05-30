using ProbabilisticCircuits: tagged_at

##################################################################################
# Downward pass
##################################################################################

function multi_head_layer_down_kernel(flows, edge_aggr, edges, _mars, example_ids,
                                      num_ex_threads::Int32, num_examples::Int32, 
                                      layer_start::Int32, edge_work::Int32, layer_end::Int32, weights)

    mars = Base.Experimental.Const(_mars)
        
    threadid_block = threadIdx().x
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadid_block 

    edge_batch, ex_id = fldmod1(threadid, num_ex_threads)
    orig_ex_id = if ex_id <= num_examples
        example_ids[ex_id]
    else
        1
    end

    edge_start = layer_start + (edge_batch-one(Int32))*edge_work 
    edge_end = min(edge_start + edge_work - one(Int32), layer_end) 
   
    warp_lane = mod1(threadid_block, warpsize())

    local acc::Float32    
    local prime_mar::Float32

    owned_node::Bool = false
    
    for edge_id = edge_start:edge_end

        edge = edges[edge_id]

        parent_id = edge.parent_id
        prime_id = edge.prime_id
        sub_id = edge.sub_id

        tag = edge.tag
        firstedge = isfirst(tag)
        lastedge = islast(tag)
        issum = edge isa SumEdge
        active = (ex_id <= num_examples) && !tagged_at(tag, 4)
        
        if firstedge
            partial = ispartial(tag)
            owned_node = !partial
        end

        if active
            
            edge_flow = flows[ex_id, parent_id]

            if issum
                parent_mar = mars[ex_id, parent_id]
                child_prob = mars[ex_id, prime_id] + edge.logp
                if sub_id != 0
                    child_prob += mars[ex_id, sub_id]
                end
                edge_flow = edge_flow * exp(child_prob - parent_mar)
            end

            if sub_id != 0 
                if isonlysubedge(tag)
                    flows[ex_id, sub_id] = edge_flow
                else
                    CUDA.@atomic flows[ex_id, sub_id] += edge_flow
                end            
            end
        end
        
        # make sure this is run on all warp threads, regardless of `active`
        if !isnothing(edge_aggr)
            !active && (edge_flow = zero(Float32))
            if !isnothing(weights)
                acc_edge_flow = edge_flow * weights[orig_ex_id]
            end
            edge_flow_warp = CUDA.reduce_warp(+, edge_flow)
            if warp_lane == 1
                CUDA.@atomic edge_aggr[edge_id] += edge_flow_warp
            end
        end

        if active

            # accumulate flows from parents
            if firstedge || (edge_id == edge_start)  
                acc = edge_flow
            else
                acc += edge_flow
            end

            # write to global memory
            if lastedge || (edge_id == edge_end)   
                if lastedge && owned_node
                    # no one else is writing to this global memory
                    flows[ex_id, prime_id] = acc
                else
                    CUDA.@atomic flows[ex_id, prime_id] += acc
                end
            end
        end
        
    end

    nothing
end

function multi_head_layer_down(flows, edge_aggr, mhbpc, head_mask, mars, 
                               layer_start, layer_end, example_ids, weights; 
                               mine, maxe, debug=false)
    bpc = mhbpc.bpc
    edges = bpc.edge_layers_down.vectors
    num_edges = layer_end-layer_start+1
    num_examples = length(example_ids)
    dummy_args = (flows, edge_aggr, edges, mars, example_ids,
                  Int32(32), Int32(num_examples), 
                  Int32(1), Int32(1), Int32(2), weights)
    kernel = @cuda name="multi_head_layer_down_kernel" launch=false multi_head_layer_down_kernel(dummy_args...) 
    config = launch_configuration(kernel.fun)

    # configure thread/block balancing
    threads, blocks, num_example_threads, edge_work = 
        balance_threads(num_edges, num_examples, config; mine, maxe, contiguous_warps=true)
    
    args = (flows, edge_aggr, edges, mars, example_ids,
            Int32(num_example_threads), Int32(num_examples), 
            Int32(layer_start), Int32(edge_work), Int32(layer_end), weights)
    if debug
        println("Layer $layer_start:$layer_end")
        @show threads blocks num_example_threads edge_work, num_edges num_examples
        CUDA.@time kernel(args...; threads, blocks)
    else
        kernel(args...; threads, blocks)
    end
    nothing
end

function multi_head_init_flows_kernel(flows, root_ids, head_mask, num_ex_threads, example_ids, node_work)
    threadid_block = threadIdx().x
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadid_block 

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = 1 + (node_batch - one(Int32)) * node_work 
    node_end = min(node_start + node_work - one(Int32), size(head_mask, 2))
    
    if ex_id <= length(example_ids)
        orig_ex_id = example_ids[ex_id]
        for idx = node_start:node_end
            root_node_id = root_ids[idx]
            flows[ex_id, root_node_id] = head_mask[orig_ex_id, idx]
        end
    end
    nothing
end

function multi_head_init_flows(flows, root_ids, head_mask, example_ids; mine = 2, maxe = 32)
    num_nodes = size(flows, 2)
    num_root_nodes = size(head_mask, 2)

    @inbounds @views flows .= zero(Float32)

    dummy_args = (flows, root_ids, head_mask, Int32(32), example_ids, Int32(1))
    kernel = @cuda name="multi_head_init_flows" launch=false multi_head_init_flows_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_ex_threads, node_work = 
        balance_threads(num_root_nodes, length(example_ids), config; mine, maxe)

    args = (flows, root_ids, head_mask, Int32(num_ex_threads), example_ids, Int32(node_work))
    kernel(args...; threads, blocks)
    nothing
end

function multi_head_flows_circuit(flows, edge_aggr, mhbpc, head_mask, mars, example_ids, weights; mine, maxe, debug=false)
    bpc = mhbpc.bpc

    num_examples = length(example_ids)
    
    @assert length(mhbpc.root_ids) == size(head_mask, 2)

    multi_head_init_flows(flows, mhbpc.root_ids, head_mask, example_ids; mine, maxe)

    layer_start = 1
    for layer_end in bpc.edge_layers_down.ends
        multi_head_layer_down(flows, edge_aggr, mhbpc, head_mask, mars,
                              layer_start, layer_end, example_ids, weights; 
                              mine, maxe, debug)
        layer_start = layer_end + 1
    end
    nothing
end

function multi_head_probs_flows_circuit(flows, mars, edge_aggr, mhbpc, data, head_mask, example_ids; mine, maxe, soft_reg, soft_reg_width, 
                                        debug = false, weights = nothing)
    eval_multi_head_pc(mars, mhbpc, data, example_ids; mine, maxe, soft_reg, soft_reg_width, debug)
    multi_head_flows_circuit(flows, edge_aggr, mhbpc, head_mask, mars, example_ids, weights; mine, maxe, debug)
    input_flows_circuit_with_reg(flows, mhbpc.bpc, data, example_ids; mine, maxe, soft_reg, 
                                 soft_reg_width, debug, weights) # reuse from `em with reg`
    nothing
end