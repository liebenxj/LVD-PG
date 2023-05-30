using PyCall
np = pyimport("numpy")

###########################################################
## Compute per-sample flows using a matrix of parameters
###########################################################

function per_sample_flows(cbpc::CuCondBitsProbCircuit, data::CuMatrix, params::CuMatrix, param_flows::CuMatrix; 
                          batch_size, mine = 2, maxe = 32, soft_reg = 0.0, soft_reg_width = 1, 
                          mars_mem = nothing, edge_mars_mem = nothing, flows_mem = nothing,
                          edge_flows_mem = nothing, input_edge_flows_mem = nothing, edge_groups_mem = nothing,
                          normalize_flows = false)
    n_examples = size(data, 1)
    n_nodes = num_nodes(cbpc)
    n_edges = num_edges(cbpc)
    n_params = num_parameters(cbpc)
    n_pargroups = cbpc.num_param_groups
    param2group = cbpc.param2group
    num_groups = maximum(param2group)
    input_node_ids = cbpc.bpc.input_node_ids

    normalized_flows = CUDA.zeros(Float32, n_examples, n_params)

    mars = prep_memory(mars_mem, (batch_size, n_nodes), (false, true))
    edge_mars = prep_memory(edge_mars_mem, (batch_size, n_edges), (false, true))
    flows = prep_memory(flows_mem, (batch_size, n_nodes), (false, true))
    edge_flows = prep_memory(edge_flows_mem, (batch_size, n_edges), (false, true))
    input_edge_flows = prep_memory(input_edge_flows_mem, (batch_size, cbpc.num_input_params), (false, true))
    edge_aggr = nothing

    edge_groups = prep_memory(edge_groups_mem, (batch_size, num_groups), (false, true))

    for batch_start = 1 : batch_size : n_examples
        batch_end = min(batch_start + batch_size - 1, n_examples)
        batch = batch_start : batch_end
        num_batch_examples = batch_end - batch_start + 1
        
        conditional_circuit_flows(flows, edge_flows, input_edge_flows, mars, edge_mars, edge_aggr, cbpc, data, params, 
                                  batch; mine, maxe, soft_reg, soft_reg_width)

        map_raw_flows(edge_flows, cbpc, param_flows, edge_groups, batch; mine, maxe, normalize_flows)
        map_input_raw_flows(input_edge_flows, cbpc, param_flows, data, batch; mine, maxe, normalize_flows)
    end

    nothing
end

##################################################################################
## Compute inner node flows using a matrix of parameters
##################################################################################

function cpc_layer_down_kernel(flows, edge_flows, edge_aggr, edges, params, edge2param, example_ids,
                               _mars, _down2upedge, num_ex_threads::Int32, num_examples::Int32, 
                               layer_start::Int32, edge_work::Int32, layer_end::Int32)

    mars = Base.Experimental.Const(_mars)
    down2upedge = Base.Experimental.Const(_down2upedge)
        
    threadid_block = threadIdx().x
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadid_block 

    edge_batch, ex_id = fldmod1(threadid, num_ex_threads)

    edge_start = layer_start + (edge_batch-one(Int32))*edge_work 
    edge_end = min(edge_start + edge_work - one(Int32), layer_end) 
   
    warp_lane = mod1(threadid_block, warpsize())

    local acc::Float32    
    local prime_mar::Float32

    owned_node::Bool = false
    
    @inbounds for edge_id = edge_start:edge_end

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

            up_edge_id = down2upedge[edge_id]

            if issum
                param_id = edge2param[up_edge_id]
                orig_ex_id = example_ids[ex_id]
                logp = params[orig_ex_id, param_id]
                parent_mar = mars[ex_id, parent_id]
                child_prob = mars[ex_id, prime_id] + logp
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

            edge_flows[ex_id, up_edge_id] = edge_flow
        end
        
        # make sure this is run on all warp threads, regardless of `active`
        if !isnothing(edge_aggr)
            !active && (edge_flow = zero(Float32))
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

function cpc_layer_down(flows, edge_flows, edge_aggr, bpc, mars, params, edge2param, 
                        layer_start, layer_end, example_ids; 
                        mine, maxe, debug=false)
    edges = bpc.edge_layers_down.vectors
    num_edges = layer_end-layer_start+1
    down2upedge = bpc.down2upedge
    num_examples = length(example_ids)

    dummy_args = (flows, edge_flows, edge_aggr, edges, params, edge2param, example_ids, mars, down2upedge, 
                  Int32(32), Int32(num_examples), 
                  Int32(1), Int32(1), Int32(2))
    kernel = @cuda name="cpc_layer_down" launch=false cpc_layer_down_kernel(dummy_args...) 
    config = launch_configuration(kernel.fun)

    # configure thread/block balancing
    threads, blocks, num_example_threads, edge_work = 
        balance_threads(num_edges, num_examples, config; mine, maxe, contiguous_warps=true)
    
    args = (flows, edge_flows, edge_aggr, edges, params, edge2param, example_ids, mars, down2upedge, 
            Int32(num_example_threads), Int32(num_examples), 
            Int32(layer_start), Int32(edge_work), Int32(layer_end))
    if debug
        println("Layer $layer_start:$layer_end")
        @show threads blocks num_example_threads edge_work, num_edges num_examples
        CUDA.@time kernel(args...; threads, blocks)
    else
        kernel(args...; threads, blocks)
    end
    nothing
end

function flows_conditional_circuit(flows, edge_flows, edge_aggr, cbpc::CuCondBitsProbCircuit, params, mars, example_ids; mine, maxe, 
                                   debug = false)
    bpc = cbpc.bpc
    edge2param = cbpc.edge2param

    init_flows() = begin 
        flows .= zero(Float32)
        edge_flows .= zero(Float32)
        flows[:,end] .= one(Float32)
        if edge_aggr !== nothing
            edge_aggr .= zero(Float32)
        end
    end
    if debug
        println("Initializing flows")
        CUDA.@time CUDA.@sync init_flows()
    else
        init_flows()
    end

    layer_start = 1
    for layer_end in bpc.edge_layers_down.ends
        cpc_layer_down(flows, edge_flows, edge_aggr, bpc, mars, 
                       params, edge2param, 
                       layer_start, layer_end, example_ids; 
                       mine, maxe, debug)
        layer_start = layer_end + 1
    end
    nothing
end

function input_flows_conditional_circuit_kernel(flows, input_edge_flows, nodes, params, input_node_ids, heap, data, 
                                                example_ids, innode2ncumparam, num_ex_threads::Int32, 
                                                node_work::Int32, soft_reg::Float32, soft_reg_width::Int32, num_inner_params)

    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = one(Int32) + (node_batch - one(Int32)) * node_work
    node_end = min(node_start + node_work - one(Int32), length(input_node_ids))

    @inbounds if ex_id <= length(example_ids)
        for node_id = node_start : node_end
            orig_ex_id::Int32 = example_ids[ex_id]
            
            orig_node_id::UInt32 = input_node_ids[node_id]
            node_flow::Float32 = flows[ex_id, orig_node_id]
            inputnode = nodes[orig_node_id]::BitsInput
            variable = inputnode.variable
            value = data[orig_ex_id, variable]
            d = dist(inputnode)

            nparams = num_parameters(d, false)
            if nparams > 0
                if soft_reg < Float32(1e-6)
                    for param_id = 1 : nparams
                        input_edge_flows[ex_id, innode2ncumparam[node_id] - nparams + param_id] = get_edge_flow(d, value, node_flow, param_id-1, heap)
                    end
                else
                    param_start = num_inner_params + innode2ncumparam[node_id] - nparams + 1
                    flow_start = innode2ncumparam[node_id] - nparams + 1
                    soft_flow_params(d, value, node_flow, params, input_edge_flows, ex_id, param_start, flow_start, soft_reg, soft_reg_width)
                end
            end
        end
    end
    nothing
end

function input_flows_conditional_circuit(flows, params, input_edge_flows, cbpc::CuCondBitsProbCircuit, data, example_ids; mine, maxe, 
                                         debug = false, soft_reg = 0.0, soft_reg_width = 1)
    bpc = cbpc.bpc

    num_examples = length(example_ids)
    num_input_nodes = length(bpc.input_node_ids)
    num_inner_params = length(cbpc.param2edge)

    @inbounds @views input_edge_flows .= zero(Float32)

    dummy_args = (flows, input_edge_flows, bpc.nodes, params, bpc.input_node_ids,
                  bpc.heap, data, example_ids, cbpc.innode2ncumparam, Int32(1), Int32(1), 
                  Float32(soft_reg), Int32(soft_reg_width), num_inner_params)
    kernel = @cuda name="input_flows_conditional_circuit" launch=false input_flows_conditional_circuit_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, node_work = 
        balance_threads(num_input_nodes, num_examples, config; mine, maxe)

    args = (flows, input_edge_flows, bpc.nodes, params, bpc.input_node_ids,
            bpc.heap, data, example_ids, cbpc.innode2ncumparam, Int32(num_example_threads), 
            Int32(node_work), Float32(soft_reg), Int32(soft_reg_width), num_inner_params)
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
# Full downward pass using a matrix of parameters
##################################################################################

function conditional_circuit_flows(flows, edge_flows, input_edge_flows, mars, edge_mars, edge_aggr, 
                                   cbpc::CuCondBitsProbCircuit, data, params, example_ids; mine, maxe, 
                                   soft_reg = 0.0, soft_reg_width = 1, debug=false)
    eval_conditional_circuit(mars, edge_mars, cbpc, params, data, example_ids; mine, maxe, debug, soft_reg, soft_reg_width)
    flows_conditional_circuit(flows, edge_flows, edge_aggr, cbpc, params, mars, example_ids; mine, maxe, debug)
    input_flows_conditional_circuit(flows, params, input_edge_flows, cbpc, data, example_ids; mine, maxe, debug, soft_reg, soft_reg_width)
    nothing
end

###################################################################################
# Map flows
###################################################################################

function map_raw_flows_kernel(edge_flows, param2edge, param2group, param_flows, example_ids, 
                              num_ex_threads::Int32, param_work::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    param_batch, ex_id = fldmod1(threadid, num_ex_threads)

    param_start = one(Int32) + (param_batch - one(Int32)) * param_work
    param_end = min(param_start + param_work - one(Int32), length(param2edge))

    @inbounds if ex_id <= length(example_ids)
        for param_id = param_start : param_end
            orig_ex_id = example_ids[ex_id]
            edge_id = param2edge[param_id]
            param_group_id = param2group[param_id]
            
            param_flows[orig_ex_id, param_id] = edge_flows[ex_id, edge_id]

        end
    end

    nothing
end

function map_raw_flows_normalize_kernel1(edge_flows, param2edge, param2group, param_flows, edge_groups, example_ids, 
                                         num_ex_threads::Int32, param_work::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    param_batch, ex_id = fldmod1(threadid, num_ex_threads)

    param_start = one(Int32) + (param_batch - one(Int32)) * param_work
    param_end = min(param_start + param_work - one(Int32), length(param2edge))

    @inbounds if ex_id <= length(example_ids)
        for param_id = param_start : param_end
            edge_id = param2edge[param_id]
            param_group_id = param2group[param_id]
            
            CUDA.@atomic edge_groups[ex_id, param_group_id] += edge_flows[ex_id, edge_id] + Float32(1e-6)
        end
    end

    nothing
end

function map_raw_flows_normalize_kernel2(edge_flows, param2edge, param2group, param_flows, edge_groups, example_ids, 
                                         num_ex_threads::Int32, param_work::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    param_batch, ex_id = fldmod1(threadid, num_ex_threads)

    param_start = one(Int32) + (param_batch - one(Int32)) * param_work
    param_end = min(param_start + param_work - one(Int32), length(param2edge))

    @inbounds if ex_id <= length(example_ids)
        for param_id = param_start : param_end
            orig_ex_id = example_ids[ex_id]
            edge_id = param2edge[param_id]
            param_group_id = param2group[param_id]
            
            param_flows[orig_ex_id, param_id] = (edge_flows[ex_id, edge_id] + Float32(1e-6)) / edge_groups[ex_id, param_group_id]
        end
    end

    nothing
end

function map_raw_flows(edge_flows, cbpc, param_flows, edge_groups, example_ids; mine = 2, maxe = 32, normalize_flows = false)
    bpc = cbpc.bpc
    param2edge = cbpc.param2edge
    param2group = cbpc.param2group

    if !normalize_flows
        dummy_args = (edge_flows, param2edge, param2group, param_flows, example_ids, Int32(1), Int32(1))
        kernel = @cuda name="map_raw_flows" launch=false map_raw_flows_kernel(dummy_args...)
        config = launch_configuration(kernel.fun)

        threads, blocks, num_example_threads, param_work = 
            balance_threads(length(param2edge), length(example_ids), config; mine, maxe)
        
        args = (edge_flows, param2edge, param2group, param_flows, example_ids, 
                Int32(num_example_threads), Int32(param_work))
        kernel(args...; threads, blocks)
    else
        @inbounds @views edge_groups[:,:] .= zero(Float32)

        dummy_args = (edge_flows, param2edge, param2group, param_flows, edge_groups, example_ids, Int32(1), Int32(1))
        kernel1 = @cuda name="map_raw_flows_normalize1" launch=false map_raw_flows_normalize_kernel1(dummy_args...)
        config = launch_configuration(kernel1.fun)

        threads, blocks, num_example_threads, param_work = 
            balance_threads(length(param2edge), length(example_ids), config; mine, maxe)
        
        args = (edge_flows, param2edge, param2group, param_flows, edge_groups, example_ids, 
                Int32(num_example_threads), Int32(param_work))
        kernel1(args...; threads, blocks)

        dummy_args = (edge_flows, param2edge, param2group, param_flows, edge_groups, example_ids, Int32(1), Int32(1))
        kernel2 = @cuda name="map_raw_flows_normalize2" launch=false map_raw_flows_normalize_kernel2(dummy_args...)
        config = launch_configuration(kernel2.fun)

        threads, blocks, num_example_threads, param_work = 
            balance_threads(length(param2edge), length(example_ids), config; mine, maxe)
        
        args = (edge_flows, param2edge, param2group, param_flows, edge_groups, example_ids, 
                Int32(num_example_threads), Int32(param_work))
        kernel2(args...; threads, blocks)
    end
    nothing
end

function map_input_raw_flows_kernel(input_edge_flows, nodes, param_flows, input_node_ids, innode2ncumparam, num_inner_params, 
                                    data, example_ids, num_ex_threads::Int32, node_work::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = one(Int32) + (node_batch - one(Int32)) * node_work
    node_end = min(node_start + node_work - one(Int32), length(input_node_ids))

    @inbounds if ex_id <= length(example_ids)
        for idx = node_start : node_end
            orig_ex_id = example_ids[ex_id]
            node_id = input_node_ids[idx]
            node = nodes[node_id]::BitsInput
            d = dist(node)

            if d isa BitsFixableCategorical
                if !d.fixed
                    nparams = num_parameters(d, false)
                    if nparams > 0
                        param_id_base = num_inner_params + innode2ncumparam[idx] - nparams
                        for param_id = 1 : nparams
                            param_flows[orig_ex_id, param_id_base + param_id] = input_edge_flows[ex_id, innode2ncumparam[idx] - nparams + param_id]
                        end
                    end
                end
            elseif d isa BitsCategorical
                nparams = d.num_cats
                param_id_base = num_inner_params + innode2ncumparam[idx] - nparams
                for param_id = 1 : nparams
                    param_flows[orig_ex_id, param_id_base + param_id] = input_edge_flows[ex_id, innode2ncumparam[idx] - nparams + param_id]
                end
            end
        end
    end
    nothing
end

function map_input_raw_flows_normalize_kernel(input_edge_flows, nodes, param_flows, input_node_ids, innode2ncumparam, num_inner_params, 
                                              data, example_ids, num_ex_threads::Int32, node_work::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = one(Int32) + (node_batch - one(Int32)) * node_work
    node_end = min(node_start + node_work - one(Int32), length(input_node_ids))
    cnt = 0
    @inbounds if ex_id <= length(example_ids)

        for idx = node_start : node_end
            orig_ex_id = example_ids[ex_id]
            node_id = input_node_ids[idx]
            node = nodes[node_id]::BitsInput
            d = dist(node)

            if d isa BitsFixableCategorical
                if !d.fixed
                    nparams = num_parameters(d, false)
                    if nparams > 0
                        param_id_base = num_inner_params + innode2ncumparam[idx] - nparams
                        sum_flow = zero(Float32)
                        for param_id = 1 : nparams
                            flow = input_edge_flows[ex_id, innode2ncumparam[idx] - nparams + param_id]
                            sum_flow += flow + Float32(1e-6)
                        end
                        for param_id = 1 : nparams
                            flow = input_edge_flows[ex_id, innode2ncumparam[idx] - nparams + param_id]
                            param_flows[orig_ex_id, param_id_base + param_id] = (flow + Float32(1e-6)) / sum_flow
                        end
                    end
                end
            elseif d isa BitsCategorical
                nparams = d.num_cats
                param_id_base = num_inner_params + innode2ncumparam[idx] - nparams
                sum_flow = zero(Float32)
                for param_id = 1 : nparams
                    flow = input_edge_flows[ex_id, innode2ncumparam[idx] - nparams + param_id]
                    sum_flow += flow + Float32(1e-6)
                end
                for param_id = 1 : nparams
                    flow = input_edge_flows[ex_id, innode2ncumparam[idx] - nparams + param_id]
                    param_flows[orig_ex_id, param_id_base + param_id] = (flow + Float32(1e-6)) / sum_flow

                end
            end
        end
    end
    nothing
end

function map_input_raw_flows(input_edge_flows, cbpc, param_flows, data, example_ids; mine = 2, maxe = 32, normalize_flows = false)
    bpc = cbpc.bpc
    num_inner_params = length(cbpc.param2edge)
    innode2ncumparam = cbpc.innode2ncumparam
    input_node_ids = bpc.input_node_ids

    if !normalize_flows
        dummy_args = (input_edge_flows, bpc.nodes, param_flows, input_node_ids, innode2ncumparam, 
                    num_inner_params, data, example_ids, Int32(1), Int32(1))
        kernel = @cuda name="map_input_raw_flows" launch=false map_input_raw_flows_kernel(dummy_args...)
        config = launch_configuration(kernel.fun)

        threads, blocks, num_example_threads, node_work = 
            balance_threads(length(input_node_ids), length(example_ids), config; mine, maxe)

        args = (input_edge_flows, bpc.nodes, param_flows, input_node_ids, innode2ncumparam, 
                num_inner_params, data, example_ids, Int32(num_example_threads), Int32(node_work))
        kernel(args...; threads, blocks)
    else
        dummy_args = (input_edge_flows, bpc.nodes, param_flows, input_node_ids, innode2ncumparam, 
                      num_inner_params, data, example_ids, Int32(1), Int32(1))
        kernel = @cuda name="map_input_raw_flows_normalize" launch=false map_input_raw_flows_normalize_kernel(dummy_args...)
        config = launch_configuration(kernel.fun)

        threads, blocks, num_example_threads, node_work = 
            balance_threads(length(input_node_ids), length(example_ids), config; mine, maxe)

        args = (input_edge_flows, bpc.nodes, param_flows, input_node_ids, innode2ncumparam, 
                num_inner_params, data, example_ids, Int32(num_example_threads), Int32(node_work))
        kernel(args...; threads, blocks)
    end
end