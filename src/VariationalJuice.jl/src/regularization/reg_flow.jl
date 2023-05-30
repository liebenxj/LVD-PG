

soft_flow(d, value, node_flow, heap, soft_reg, soft_reg_width) = begin
    if ismissing(value)
        CUDA.@atomic heap[d.heap_start+UInt32(2)*d.num_cats] += node_flow
    else
        c_start = max(0, value - soft_reg_width ÷ 2)
        c_end = min(c_start + soft_reg_width - 1, d.num_cats - 1)
        sp = zero(Float32)
        for cat_idx = c_start : c_end
            sp += exp(heap[d.heap_start + cat_idx])
        end
        sp /= (c_end - c_start + 1)
        base = (one(Float32) - soft_reg) * heap[d.heap_start+UInt32(value)] + soft_reg * sp

        CUDA.@atomic heap[d.heap_start+d.num_cats+UInt32(value)] += (one(Float32) - soft_reg) * heap[d.heap_start+UInt32(value)] * node_flow / base
        for cat_idx = c_start : c_end
            CUDA.@atomic heap[d.heap_start+d.num_cats+UInt32(cat_idx)] += soft_reg / (c_end - c_start + 1) * heap[d.heap_start+UInt32(cat_idx)] * node_flow / base
        end
    end
    nothing
end

function input_flows_circuit_with_reg_kernel(flows, nodes, input_node_ids, heap, data, example_ids, num_ex_threads::Int32, node_work::Int32, 
                                             soft_reg::Float32, soft_reg_width::Int32, weights)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = one(Int32) + (node_batch - one(Int32)) * node_work
    node_end = min(node_start + node_work - one(Int32), length(input_node_ids))

    @inbounds if ex_id <= length(example_ids)
        for node_id = node_start : node_end
            orig_ex_id::Int32 = example_ids[ex_id]
            orig_node_id::UInt32 = input_node_ids[node_id]
            node_flow::Float32 = flows[ex_id, orig_node_id]
            if !isnothing(weights)
                node_flow *= weights[orig_ex_id]
            end
            inputnode = nodes[orig_node_id]::BitsInput
            variable = inputnode.variable
            value = data[orig_ex_id, variable]
            soft_flow(dist(inputnode), value, node_flow, heap, soft_reg, soft_reg_width)
        end
    end
    nothing
end

function input_flows_circuit_with_reg_cat_kernel(flows, nodes, input_node_ids, heap, data, example_ids, num_ex_threads::Int32, node_work::Int32, 
                                                 soft_reg::Float32, soft_reg_width::Int32, weights)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = one(Int32) + (node_batch - one(Int32)) * node_work
    node_end = min(node_start + node_work - one(Int32), length(input_node_ids))

    @inbounds if ex_id <= length(example_ids)
        for node_id = node_start : node_end
            orig_ex_id::Int32 = example_ids[ex_id]
            orig_node_id::UInt32 = input_node_ids[node_id]
            node_flow::Float32 = flows[ex_id, orig_node_id]
            if !isnothing(weights)
                node_flow *= weights[orig_ex_id]
            end
            inputnode = nodes[orig_node_id]::BitsInput
            variable = inputnode.variable
            d = dist(inputnode)
            if d isa BitsCategorical
                logp = typemin(Float32)
                for i = 0 : d.num_cats - 1
                    logp = logsumexp(logp, data[orig_ex_id, variable, i+1] + heap[d.heap_start + i])
                end
                for i = 0 : d.num_cats - 1
                    CUDA.@atomic heap[d.heap_start+d.num_cats+i] += exp(data[orig_ex_id, variable, i+1] + heap[d.heap_start + i] - logp) * node_flow
                end
            end
        end
    end
    nothing
end

function input_flows_circuit_with_reg(flows, bpc, data, example_ids; mine, maxe, soft_reg, soft_reg_width, debug = false, weights = nothing)
    num_examples = length(example_ids)
    num_input_nodes = length(bpc.input_node_ids)

    dummy_args = (flows, bpc.nodes, bpc.input_node_ids,
                  bpc.heap, data, example_ids, Int32(1), Int32(1), Float32(soft_reg), Int32(soft_reg_width), weights)
    if data isa CuMatrix
        kernel = @cuda name="input_flows_circuit_with_reg" launch=false input_flows_circuit_with_reg_kernel(dummy_args...)
    elseif data isa CuArray{Float32, 3}
        kernel = @cuda name="input_flows_circuit_with_reg" launch=false input_flows_circuit_with_reg_cat_kernel(dummy_args...)
    else
        error("Unknown data type") 
    end
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, node_work = 
        balance_threads(num_input_nodes, num_examples, config; mine, maxe)

    args = (flows, bpc.nodes, bpc.input_node_ids,
            bpc.heap, data, example_ids, Int32(num_example_threads), Int32(node_work), Float32(soft_reg), 
            Int32(soft_reg_width), weights)
    if debug
        println("Flows of input nodes")
        @show threads blocks num_example_threads node_work num_nodes num_examples
        CUDA.@time kernel(args...; threads, blocks)
    else
        kernel(args...; threads, blocks)
    end

    nothing
end

##################################################################################
# Downward pass
##################################################################################

function reg_layer_down_kernel(flows, edge_aggr, edges, _mars, example_ids,
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
        active = (ex_id <= num_examples)
        
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

function reg_layer_down(flows, edge_aggr, bpc, mars, 
                    layer_start, layer_end, example_ids, weights; 
                    mine, maxe, debug=false)
    edges = bpc.edge_layers_down.vectors
    num_edges = layer_end-layer_start+1
    num_examples = length(example_ids)
    dummy_args = (flows, edge_aggr, edges, mars, example_ids,
                  Int32(32), Int32(num_examples), 
                  Int32(1), Int32(1), Int32(2), weights)
    kernel = @cuda name="reg_layer_down" launch=false reg_layer_down_kernel(dummy_args...) 
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

function reg_flows_circuit(flows, edge_aggr, bpc, mars, example_ids, weights; mine, maxe, debug=false)
    num_examples = length(example_ids)
    init_flows() = begin 
        flows .= zero(Float32)
        flows[:,end] .= one(Float32)
    end
    if debug
        println("Initializing flows")
        CUDA.@time CUDA.@sync init_flows()
    else
        init_flows()
    end

    layer_start = 1
    for layer_end in bpc.edge_layers_down.ends
        reg_layer_down(flows, edge_aggr, bpc, mars,
                   layer_start, layer_end, example_ids, weights; 
                   mine, maxe, debug)
        layer_start = layer_end + 1
    end
    nothing
end

function probs_flows_circuit_with_reg(flows, mars, edge_aggr, bpc, data, example_ids; mine, maxe, soft_reg, soft_reg_width, 
                                      debug = false, weights = nothing)
    eval_circuit_with_reg(mars, bpc, data, example_ids; mine, maxe, soft_reg, soft_reg_width, debug)
    reg_flows_circuit(flows, edge_aggr, bpc, mars, example_ids, weights; mine, maxe, debug)
    input_flows_circuit_with_reg(flows, bpc, data, example_ids; mine, maxe, soft_reg, soft_reg_width, debug, weights)
    nothing
end