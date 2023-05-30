using ProbabilisticCircuits: ispartial, isonlysubedge


##################################################################################
# Downward pass
##################################################################################

function layer_down_kernel(flows, edge_flows, edge_aggr, edges, _mars, _down2upedge, 
                           num_ex_threads::Int32, num_examples::Int32, 
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

            up_edge_id = down2upedge[edge_id]
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

function layer_down(flows, edge_flows, edge_aggr, bpc, mars, 
                    layer_start, layer_end, num_examples; 
                    mine, maxe, debug=false)
    edges = bpc.edge_layers_down.vectors
    num_edges = layer_end-layer_start+1
    down2upedge = bpc.down2upedge
    dummy_args = (flows, edge_flows, edge_aggr, edges, mars, down2upedge, 
                  Int32(32), Int32(num_examples), 
                  Int32(1), Int32(1), Int32(2))
    kernel = @cuda name="layer_down" launch=false layer_down_kernel(dummy_args...) 
    config = launch_configuration(kernel.fun)

    # configure thread/block balancing
    threads, blocks, num_example_threads, edge_work = 
        balance_threads(num_edges, num_examples, config; mine, maxe, contiguous_warps=true)
    
    args = (flows, edge_flows, edge_aggr, edges, mars, down2upedge, 
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

function flows_circuit(flows, edge_flows, edge_aggr, mbpc::CuMetaBitsProbCircuit, mars, num_examples; mine, maxe, debug=false)
    bpc = mbpc.bpc

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
        layer_down(flows, edge_flows, edge_aggr, bpc, mars, 
                   layer_start, layer_end, num_examples; 
                   mine, maxe, debug)
        layer_start = layer_end + 1
    end
    nothing
end

##################################################################################
# Downward pass for input nodes
##################################################################################

function input_flows_circuit_kernel(flows, input_edge_flows, nodes, input_node_ids, heap, data, 
                                    example_ids, innode2ncumparam, num_ex_threads::Int32, node_work::Int32)

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
            flow(d, value, node_flow, heap)

            nparams = num_parameters(d, false)
            if nparams > 0
                for param_id = 1 : nparams
                    input_edge_flows[ex_id, innode2ncumparam[node_id] - nparams + param_id] = get_edge_flow(d, value, node_flow, param_id, heap)
                end
            end
        end
    end
    nothing
end

function input_flows_circuit(flows, input_edge_flows, mbpc::CuMetaBitsProbCircuit, data, example_ids; mine, maxe, debug=false)
    bpc = mbpc.bpc

    num_examples = length(example_ids)
    num_input_nodes = length(bpc.input_node_ids)

    dummy_args = (flows, input_edge_flows, bpc.nodes, bpc.input_node_ids,
                  bpc.heap, data, example_ids, mbpc.innode2ncumparam, Int32(1), Int32(1))
    kernel = @cuda name="input_flows_circuit" launch=false input_flows_circuit_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, node_work = 
        balance_threads(num_input_nodes, num_examples, config; mine, maxe)

    args = (flows, input_edge_flows, bpc.nodes, bpc.input_node_ids,
            bpc.heap, data, example_ids, mbpc.innode2ncumparam, Int32(num_example_threads), Int32(node_work))
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
# Full downward pass
##################################################################################

function probs_flows_circuit(flows, edge_flows, input_edge_flows, mars, edge_mars, edge_aggr, 
                             mbpc::CuMetaBitsProbCircuit, data, example_ids; mine, maxe, debug=false)
    eval_circuit(mars, edge_mars, mbpc, data, example_ids; mine, maxe, debug)
    flows_circuit(flows, edge_flows, edge_aggr, mbpc, mars, length(example_ids); mine, maxe, debug)
    input_flows_circuit(flows, input_edge_flows, mbpc, data, example_ids; mine, maxe, debug)
    nothing
end

##################################################################################
## Compute inner node flows using a matrix of parameters
##################################################################################

function layer_down_kernel(flows, edge_flows, edge_aggr, edges, params, edge2param, example_ids,
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

function layer_down(flows, edge_flows, edge_aggr, bpc, mars, params, edge2param, 
                    layer_start, layer_end, example_ids; 
                    mine, maxe, debug=false)
    edges = bpc.edge_layers_down.vectors
    num_edges = layer_end-layer_start+1
    down2upedge = bpc.down2upedge
    num_examples = length(example_ids)

    dummy_args = (flows, edge_flows, edge_aggr, edges, params, edge2param, example_ids, mars, down2upedge, 
                  Int32(32), Int32(num_examples), 
                  Int32(1), Int32(1), Int32(2))
    kernel = @cuda name="layer_down" launch=false layer_down_kernel(dummy_args...) 
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

function flows_circuit(flows, edge_flows, edge_aggr, mbpc::CuMetaBitsProbCircuit, params, mars, example_ids; mine, maxe, debug=false)
    bpc = mbpc.bpc
    edge2param = mbpc.edge2param

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
        layer_down(flows, edge_flows, edge_aggr, bpc, mars, 
                   params, edge2param, 
                   layer_start, layer_end, example_ids; 
                   mine, maxe, debug)
        layer_start = layer_end + 1
    end
    nothing
end

##################################################################################
# Full downward pass using a matrix of parameters
##################################################################################

function probs_flows_circuit(flows, edge_flows, input_edge_flows, mars, edge_mars, edge_aggr, 
                             mbpc::CuMetaBitsProbCircuit, data, params, example_ids; mine, maxe, debug=false)
    eval_circuit(mars, edge_mars, mbpc, params, data, example_ids; mine, maxe, debug)
    flows_circuit(flows, edge_flows, edge_aggr, mbpc, params, mars, example_ids; mine, maxe, debug)
    input_flows_circuit(flows, input_edge_flows, mbpc, data, example_ids; mine, maxe, debug)
    nothing
end

##################################################################################
# Get per-sample normalized flows
##################################################################################

function per_sample_normalized_flows(mbpc::CuMetaBitsProbCircuit, data::Matrix; batch_size, mine = 2, maxe = 32,
                                     mars_mem = nothing, edge_mars_mem = nothing, flows_mem = nothing, edge_flows_mem = nothing,
                                     input_edge_flows_mem = nothing, params_mem = nothing, par_groups_mem = nothing)
    data = cu(data)
    per_sample_normalized_flows(mbpc, data; batch_size, mine, maxe, mars_mem, edge_mars_mem, flows_mem, edge_flows_mem,
                                input_edge_flows_mem, params_mem, par_groups_mem)
end
function per_sample_normalized_flows(mbpc::CuMetaBitsProbCircuit, data::CuMatrix; batch_size, mine = 2, maxe = 32,
                                     mars_mem = nothing, edge_mars_mem = nothing, flows_mem = nothing, edge_flows_mem = nothing,
                                     input_edge_flows_mem = nothing, params_mem = nothing, par_groups_mem = nothing, 
                                     normalized_flows_mem = nothing)
    n_examples = size(data, 1)
    n_nodes = num_nodes(mbpc)
    n_edges = num_edges(mbpc)
    n_params = num_parameters(mbpc)
    n_pargroups = mbpc.num_param_groups
    # @assert size(normalized_flows, 2) == n_params

    # normalized_flows = CUDA.zeros(Float32, n_examples, n_params)
    normalized_flows = prep_memory(normalized_flows_mem, (n_examples, n_params))
    normalized_flows .= zero(Float32)

    mars = prep_memory(mars_mem, (batch_size, n_nodes), (false, true))
    edge_mars = prep_memory(edge_mars_mem, (batch_size, n_edges), (false, true))
    flows = prep_memory(flows_mem, (batch_size, n_nodes), (false, true))
    edge_flows = prep_memory(edge_flows_mem, (batch_size, n_edges), (false, true))
    input_edge_flows = prep_memory(input_edge_flows_mem, (batch_size, mbpc.num_input_params), (false, true))
    par_groups = prep_memory(par_groups_mem, (batch_size, n_pargroups), (false, true))
    edge_aggr = nothing

    for batch_start = 1 : batch_size : n_examples
        batch_end = min(batch_start + batch_size - 1, n_examples)
        batch = batch_start : batch_end
        num_batch_examples = batch_end - batch_start + 1
        
        probs_flows_circuit(flows, edge_flows, input_edge_flows, mars, edge_mars, edge_aggr, mbpc, data, batch; mine, maxe)
        map_flows(edge_flows, mbpc, normalized_flows, par_groups, batch; mine, maxe)
        map_input_flows(input_edge_flows, mbpc, normalized_flows, data, batch; mine, maxe)
    end
    
    if normalized_flows_mem === nothing
        Array(normalized_flows)
    else
        nothing
    end
end
function per_sample_normalized_flows(mbpc::CuMetaBitsProbCircuit, data::CuMatrix, params::CuMatrix; 
                                     batch_size, mine = 2, maxe = 32, mars_mem = nothing, edge_mars_mem = nothing, flows_mem = nothing,
                                     edge_flows_mem = nothing, input_edge_flows_mem = nothing, par_groups_mem = nothing)
    n_examples = size(data, 1)
    n_nodes = num_nodes(mbpc)
    n_edges = num_edges(mbpc)
    n_params = num_parameters(mbpc)
    n_pargroups = mbpc.num_param_groups
    # @assert size(normalized_flows, 2) == n_params

    normalized_flows = CUDA.zeros(Float32, n_examples, n_params)

    mars = prep_memory(mars_mem, (batch_size, n_nodes), (false, true))
    edge_mars = prep_memory(edge_mars_mem, (batch_size, n_edges), (false, true))
    flows = prep_memory(flows_mem, (batch_size, n_nodes), (false, true))
    edge_flows = prep_memory(edge_flows_mem, (batch_size, n_edges), (false, true))
    input_edge_flows = prep_memory(input_edge_flows_mem, (batch_size, mbpc.num_input_params), (false, true))
    par_groups = prep_memory(par_groups_mem, (batch_size, n_pargroups), (false, true))
    edge_aggr = nothing

    for batch_start = 1 : batch_size : n_examples
        batch_end = min(batch_start + batch_size - 1, n_examples)
        batch = batch_start : batch_end
        num_batch_examples = batch_end - batch_start + 1
        
        probs_flows_circuit(flows, edge_flows, input_edge_flows, mars, edge_mars, edge_aggr, mbpc, data, params, batch; mine, maxe)
        map_flows(edge_flows, mbpc, normalized_flows, par_groups, batch; mine, maxe)
        map_input_flows(input_edge_flows, mbpc, normalized_flows, data, batch; mine, maxe)
    end
    
    Array(normalized_flows)
end

function map_flows_kernel(edge_flows, param2edge, param2group, normalized_flows, par_groups, example_ids, 
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

            eflow = edge_flows[ex_id, edge_id]
            CUDA.@atomic par_groups[ex_id, param_group_id] += eflow
        end

        CUDA.sync_threads()

        for param_id = param_start : param_end
            orig_ex_id = example_ids[ex_id]
            edge_id = param2edge[param_id]
            param_group_id = param2group[param_id]
            
            eflow = edge_flows[ex_id, edge_id]
            normalized_flows[orig_ex_id, param_id] = log(eflow / par_groups[ex_id, param_group_id] + Float32(1e-8))
        end
    end

    nothing
end

function map_flows(edge_flows, mbpc, normalized_flows, par_groups, example_ids; mine = 2, maxe = 32)
    bpc = mbpc.bpc
    param2edge = mbpc.param2edge
    param2group = mbpc.param2group

    @inbounds @views par_groups .= Float32(1e-8)

    dummy_args = (edge_flows, param2edge, param2group, normalized_flows, par_groups, example_ids, Int32(1), Int32(1))
    kernel = @cuda name="map_flows" launch=false map_flows_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, param_work = 
        balance_threads(length(param2edge), length(example_ids), config; mine, maxe)
    
    args = (edge_flows, param2edge, param2group, normalized_flows, par_groups, example_ids, 
            Int32(num_example_threads), Int32(param_work))
    kernel(args...; threads, blocks)
end

function map_input_flows_kernel(input_edge_flows, nodes, normalized_flows, input_node_ids, innode2ncumparam, num_inner_params, 
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
                        sum_pars = zero(Float32)
                        for param_id = 1 : nparams
                            sum_pars += input_edge_flows[ex_id, innode2ncumparam[idx] - nparams + param_id]
                        end
                        for param_id = 1 : nparams
                            normalized_flows[orig_ex_id, param_id_base + param_id] = log(input_edge_flows[ex_id, innode2ncumparam[idx] - nparams + param_id] / sum_pars)
                        end
                    end
                end
            elseif d isa BitsCategorical
                nparams = d.num_cats
                param_id_base = num_inner_params + innode2ncumparam[idx] - nparams
                sum_pars = zero(Float32)
                for param_id = 1 : nparams
                    sum_pars += input_edge_flows[ex_id, innode2ncumparam[idx] - nparams + param_id]
                end
                for param_id = 1 : nparams
                    normalized_flows[orig_ex_id, param_id_base + param_id] = log(input_edge_flows[ex_id, innode2ncumparam[idx] - nparams + param_id] / sum_pars)
                end
            end
        end
    end
    nothing
end

function map_input_flows(input_edge_flows, mbpc, normalized_flows, data, example_ids; mine = 2, maxe = 32)
    bpc = mbpc.bpc
    num_inner_params = length(mbpc.param2edge)
    innode2ncumparam = mbpc.innode2ncumparam
    input_node_ids = bpc.input_node_ids

    dummy_args = (input_edge_flows, bpc.nodes, normalized_flows, input_node_ids, innode2ncumparam, 
                  num_inner_params, data, example_ids, Int32(1), Int32(1))
    kernel = @cuda name="map_input_flows" launch=false map_input_flows_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, node_work = 
        balance_threads(length(input_node_ids), length(example_ids), config; mine, maxe)

    args = (input_edge_flows, bpc.nodes, normalized_flows, input_node_ids, innode2ncumparam, 
            num_inner_params, data, example_ids, Int32(num_example_threads), Int32(node_work))
    kernel(args...; threads, blocks)
end