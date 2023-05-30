using ProbabilisticCircuits: prep_memory


######################################################
## Soft sampling using the Gumbel-Softmax trick
######################################################

function gumbel_params_down_kernel(edge_probs, cum_probs, edges, params, example_ids,
                                   _down2upedge, _edge2param, _param2group, temperature::Float32, no_gumbel::Bool, num_ex_threads::Int32,
                                   layer_start::Int32, edge_work::Int32, layer_end::Int32)

    down2upedge = Base.Experimental.Const(_down2upedge)
    edge2param = Base.Experimental.Const(_edge2param)
    param2group = Base.Experimental.Const(_param2group)
        
    threadid_block = threadIdx().x
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadid_block 

    edge_batch, ex_id = fldmod1(threadid, num_ex_threads)

    edge_start = layer_start + (edge_batch-one(Int32))*edge_work 
    edge_end = min(edge_start + edge_work - one(Int32), layer_end) 
    
    @inbounds if ex_id <= length(example_ids) 
        for edge_id = edge_start:edge_end

            edge = edges[edge_id]

            parent_id = edge.parent_id

            issum = edge isa SumEdge
            active = (ex_id <= length(example_ids))

            if active && issum

                up_edge_id = down2upedge[edge_id]
                param_id = edge2param[up_edge_id]
                orig_ex_id = example_ids[ex_id]
                param_group_id = param2group[param_id]

                logp = params[orig_ex_id, param_id]
                gumbel_val = if no_gumbel
                        zero(Float32)
                    else
                        -log(-log(rand(Float32) * (1.0 - 1e-6) + 1e-6))
                    end
                unnorm_prob = exp((logp + gumbel_val) / temperature)

                edge_probs[ex_id, param_id] = unnorm_prob

                CUDA.@atomic cum_probs[ex_id, param_group_id] += unnorm_prob

            end
        end
    end

    nothing
end

function gumbel_sample_down_kernel(td_probs, edge_td_probs, edge_probs, cum_probs, edges, example_ids,
                                   _down2upedge, _edge2param, _param2group, num_ex_threads::Int32,
                                   layer_start::Int32, edge_work::Int32, layer_end::Int32)

    down2upedge = Base.Experimental.Const(_down2upedge)
    edge2param = Base.Experimental.Const(_edge2param)
    param2group = Base.Experimental.Const(_param2group)
        
    threadid_block = threadIdx().x
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadid_block 

    edge_batch, ex_id = fldmod1(threadid, num_ex_threads)

    edge_start = layer_start + (edge_batch-one(Int32)) * edge_work 
    edge_end = min(edge_start + edge_work - one(Int32), layer_end) 
    
    warp_lane = mod1(threadid_block, warpsize())

    local acc::Float32

    owned_node::Bool = false
    
    @inbounds if ex_id <= length(example_ids) 
        for edge_id = edge_start:edge_end

            edge = edges[edge_id]

            parent_id = edge.parent_id
            prime_id = edge.prime_id
            sub_id = edge.sub_id

            tag = edge.tag
            firstedge = isfirst(tag)
            lastedge = islast(tag)
            issum = edge isa SumEdge
            active = (ex_id <= length(example_ids))
            up_edge_id = down2upedge[edge_id]
            
            if firstedge
                partial = ispartial(tag)
                owned_node = !partial
            end

            if active
                
                edge_td_prob = td_probs[ex_id, parent_id]

                if issum
                    param_id = edge2param[up_edge_id]
                    orig_ex_id = example_ids[ex_id]
                    param_group_id = param2group[param_id]
                    
                    prob = edge_probs[ex_id, param_id] / cum_probs[ex_id, param_group_id]
                    
                    edge_td_prob *= prob
                end

                if sub_id != 0 
                    if isonlysubedge(tag)
                        td_probs[ex_id, sub_id] = edge_td_prob
                    else
                        CUDA.@atomic td_probs[ex_id, sub_id] += edge_td_prob
                    end            
                end

                edge_td_probs[ex_id, up_edge_id] = edge_td_prob

                # accumulate td_probs from parents
                if firstedge || (edge_id == edge_start)  
                    acc = edge_td_prob
                else
                    acc += edge_td_prob
                end

                # write to global memory
                if lastedge || (edge_id == edge_end)   
                    if lastedge && owned_node
                        # no one else is writing to this global memory
                        td_probs[ex_id, prime_id] = acc
                    else
                        CUDA.@atomic td_probs[ex_id, prime_id] += acc
                    end
                end
            end
        end
    end

    nothing
end

function gumbel_sample_layer(td_probs, edge_td_probs, edge_probs, cum_probs, params, mbpc, temperature,
                             layer_start, layer_end, example_ids; mine, maxe, debug=false, no_gumbel=false)
    bpc = mbpc.bpc
    edges = bpc.edge_layers_down.vectors
    num_edges = layer_end-layer_start+1
    down2upedge = bpc.down2upedge
    edge2param = mbpc.edge2param
    param2group = mbpc.param2group
    num_examples = length(example_ids)

    ## Compute the Gumbel-Softmax parameters ##

    dummy_args = (edge_probs, cum_probs, edges, params, example_ids, down2upedge, edge2param,
                  param2group, Float32(temperature), no_gumbel, Int32(32), Int32(1), Int32(1), Int32(2))
    kernel1 = @cuda name="gumbel_params_down" launch=false gumbel_params_down_kernel(dummy_args...) 
    config = launch_configuration(kernel1.fun)

    # configure thread/block balancing
    threads, blocks, num_example_threads, edge_work = 
        balance_threads(num_edges, num_examples, config; mine, maxe, contiguous_warps=true)
    
    args = (edge_probs, cum_probs, edges, params, example_ids, down2upedge, edge2param,
            param2group, Float32(temperature), no_gumbel, Int32(num_example_threads), 
            Int32(layer_start), Int32(edge_work), Int32(layer_end))
    if debug
        println("Layer $layer_start:$layer_end")
        @show threads blocks num_example_threads edge_work, num_edges num_examples
        CUDA.@time kernel1(args...; threads, blocks)
    else
        kernel1(args...; threads, blocks)
    end

    ## Compute top-down probabilities using the new parameters ##

    dummy_args = (td_probs, edge_td_probs, edge_probs, cum_probs, edges, example_ids,
                  down2upedge, edge2param, param2group, Int32(32), Int32(1), Int32(1), Int32(2))
    kernel2 = @cuda name="gumbel_sample_down" launch=false gumbel_sample_down_kernel(dummy_args...)
    config = launch_configuration(kernel2.fun)

    # configure thread/block balancing
    threads, blocks, num_example_threads, edge_work = 
        balance_threads(num_edges, num_examples, config; mine, maxe, contiguous_warps=true)
    
    args = (td_probs, edge_td_probs, edge_probs, cum_probs, edges, example_ids,
            down2upedge, edge2param, param2group, Int32(num_example_threads), 
            Int32(layer_start), Int32(edge_work), Int32(layer_end))
    if debug
        println("Layer $layer_start:$layer_end")
        @show threads blocks num_example_threads edge_work, num_edges num_examples
        CUDA.@time kernel2(args...; threads, blocks)
    else
        kernel2(args...; threads, blocks)
    end

    nothing
end

function gumbel_sample_inner(td_probs, edge_td_probs, edge_probs, cum_probs, params, mbpc::CuMetaBitsProbCircuit, 
                             temperature, example_ids; mine, maxe, debug=false, no_gumbel=false)
    bpc = mbpc.bpc

    init_samples() = begin 
        td_probs .= zero(Float32)
        td_probs[:,end] .= one(Float32)
        cum_probs .= zero(Float32)
    end
    if debug
        println("Initializing flows")
        CUDA.@time CUDA.@sync init_samples()
    else
        init_samples()
    end

    layer_start = 1
    for layer_end in bpc.edge_layers_down.ends
        gumbel_sample_layer(td_probs, edge_td_probs, edge_probs, cum_probs, params, mbpc, temperature,
                            layer_start, layer_end, example_ids; mine, maxe, debug, no_gumbel)
        layer_start = layer_end + 1
    end

    nothing
end

function gumbel_input_sample_down_kernel(td_probs, params, input_aggr_params, nodes, heap, num_inner_params, input_node_ids, 
                                         innode2ncumparam, example_ids, norm_params::Bool, num_ex_threads::Int32, node_work::Int32)

    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = one(Int32) + (node_batch - one(Int32)) * node_work
    node_end = min(node_start + node_work - one(Int32), length(input_node_ids))

    @inbounds if ex_id <= length(example_ids)
        for node_id = node_start : node_end
            orig_ex_id::Int32 = example_ids[ex_id]
            orig_node_id::UInt32 = input_node_ids[node_id]
            td_prob::Float32 = td_probs[ex_id, orig_node_id]
            inputnode = nodes[orig_node_id]::BitsInput
            variable = inputnode.variable
            d = dist(inputnode)

            nparams = num_parameters(d, false)
            param_start_id = num_inner_params + innode2ncumparam[node_id] - nparams

            gumbel_input_step(d, param_start_id, input_aggr_params, heap,
                              ex_id, orig_ex_id, variable, td_prob, params, nparams, norm_params)
        end
    end
    nothing
end

function gumbel_input_step(d::BitsFixableCategorical, param_start_id, input_aggr_params, heap,
                           ex_id, orig_ex_id, variable, td_prob, params, nparams, norm_params::Bool)
    if nparams > 0 # fixed == false
        if norm_params
            for param_id = 1 : nparams
                CUDA.@atomic input_aggr_params[ex_id, variable, param_id] += td_prob * 
                    params[orig_ex_id, param_start_id + param_id]
            end
        else
            for param_id = 1 : nparams
                CUDA.@atomic input_aggr_params[ex_id, variable, param_id] += td_prob * exp(
                    params[orig_ex_id, param_start_id + param_id])
            end
        end
    else
        # fixed == true
        start_idx = d.heap_start - 1
        for param_id = 1 : d.num_cats
            CUDA.@atomic input_aggr_params[ex_id, variable, param_id] += td_prob * exp(heap[start_idx + param_id])
        end
    end
    nothing
end

function gumbel_input_step(d::BitsCategorical, param_start_id, input_aggr_params, heap,
                           ex_id, orig_ex_id, variable, td_prob, params, nparams, norm_params::Bool)
    if params === nothing
        start_idx = d.heap_start - 1
        for param_id = 1 : d.num_cats
            CUDA.@atomic input_aggr_params[ex_id, variable, param_id] += td_prob * exp(heap[start_idx + param_id])
        end
    else
        @assert false 
    end
end

function gumbel_input_step(d::BitsGaussian, param_start_id, input_aggr_params, heap,
                           ex_id, orig_ex_id, variable, td_prob, params, nparams)
    mu = params[orig_ex_id, param_start_id + 1]
    sigma = params[orig_ex_id, param_start_id + 2]
    CUDA.@atomic input_aggr_params[ex_id, variable, 1] += td_prob * mu
    CUDA.@atomic input_aggr_params[ex_id, variable, 2] += td_prob * sigma
end

function gumbel_input_sample_down(td_probs, params, input_aggr_params, mbpc::CuMetaBitsProbCircuit, 
                                  example_ids; mine, maxe, norm_params=false, debug=false)

    input_aggr_params .= zero(Float32)

    bpc = mbpc.bpc

    num_examples = length(example_ids)
    num_input_nodes = length(bpc.input_node_ids)
    num_inner_params = length(mbpc.param2edge)

    dummy_args = (td_probs, params, input_aggr_params, bpc.nodes, bpc.heap, num_inner_params, bpc.input_node_ids, 
                  mbpc.innode2ncumparam, example_ids, norm_params, Int32(1), Int32(1))
    kernel = @cuda name="gumbel_input_sample_down" launch=false gumbel_input_sample_down_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, node_work = 
        balance_threads(num_input_nodes, num_examples, config; mine, maxe)

    args = (td_probs, params, input_aggr_params, bpc.nodes, bpc.heap, num_inner_params, bpc.input_node_ids, 
            mbpc.innode2ncumparam, example_ids, norm_params, Int32(num_example_threads), Int32(node_work))
    if debug
        println("Flows of input nodes")
        @show threads blocks num_example_threads node_work num_nodes num_examples
        CUDA.@time kernel(args...; threads, blocks)
    else
        kernel(args...; threads, blocks)
    end

    nothing
end

function gumbel_sample(td_probs, edge_td_probs, edge_probs, cum_probs, params, input_aggr_params,
                       mbpc::CuMetaBitsProbCircuit, temperature, example_ids; mine, maxe, norm_params=false, 
                       debug=false, no_gumbel=false)
    gumbel_sample_inner(td_probs, edge_td_probs, edge_probs, cum_probs, params, mbpc, temperature, example_ids; 
                        mine, maxe, debug, no_gumbel)
    gumbel_input_sample_down(td_probs, params, input_aggr_params, mbpc, example_ids; mine, maxe, norm_params, debug)

    nothing
end

"High-level API for Gumbel Sampling"
function gumbel_sample(mbpc::CuMetaBitsProbCircuit, params, num_vars, par_buffer_size; temperature, no_gumbel = false,
                       reuse = nothing, example_ids = nothing, mine = 2, maxe = 32, norm_params = false, debug = false)
    if example_ids === nothing
        example_ids = 1 : size(params, 1)
    end
    num_examples = length(example_ids)

    n_nodes = num_nodes(mbpc)
    n_edges = num_edges(mbpc)

    if reuse !== nothing
        td_probs = cu(reuse[1])
        edge_td_probs = cu(reuse[2])
        edge_probs = cu(reuse[3])
        cum_probs = cu(reuse[4])
    else
        td_probs = prep_memory(nothing, (num_examples, n_nodes), (false, true))
        edge_td_probs = prep_memory(nothing, (num_examples, n_edges), (false, true))
        edge_probs = prep_memory(nothing, (num_examples, length(mbpc.param2edge)), (false, true))
        cum_probs = prep_memory(nothing, (num_examples, mbpc.num_param_groups), (false, true))
    end

    if params isa Matrix{Float32}
        params = cu(params)
    end

    # Generate aggregated parameters
    input_aggr_params = CUDA.zeros(Float32, num_examples, num_vars, par_buffer_size)
    
    # Perform sampling
    gumbel_sample(td_probs, edge_td_probs, edge_probs, cum_probs, params, input_aggr_params,
                  mbpc, temperature, example_ids; mine, maxe, norm_params, no_gumbel)

    Array(input_aggr_params), (Array(td_probs), Array(edge_td_probs), Array(edge_probs), Array(cum_probs))
end

######################################################
## Backpropagate through the soft sampling process
######################################################

function gumbel_params_up_layer_kernel(grads, td_probs, param_grads, edge_probs, cum_probs, cum_grads, edges, temperature, 
                                       params, example_ids, _edge2param, _param2group, num_ex_threads::Int32,
                                       layer_start::Int32, edge_work::Int32, layer_end::Int32, grad_wrt_logparams::Bool)

    edge2param = Base.Experimental.Const(_edge2param)
    param2group = Base.Experimental.Const(_param2group)
        
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    edge_batch, ex_id = fldmod1(threadid, num_ex_threads)

    edge_start = layer_start + (edge_batch - one(Int32)) * edge_work 
    edge_end = min(edge_start + edge_work - one(Int32), layer_end)

    if ex_id <= length(example_ids)

        local acc::Float32    
        owned_node::Bool = false
        
        for edge_id = edge_start:edge_end

            edge = edges[edge_id]

            tag = edge.tag
            isfirstedge = isfirst(tag)
            islastedge = islast(tag)
            issum = edge isa SumEdge
            owned_node |= isfirstedge

            parent_id = edge.parent_id

            # compute probability coming from child
            child_grad = grads[ex_id, edge.prime_id]
            if edge.sub_id != 0
                child_grad += grads[ex_id, edge.sub_id]
            end
            if issum
                orig_ex_id = example_ids[ex_id]
                param_id = edge2param[edge_id]
                param_group_id = param2group[param_id]
                edge_prob = edge_probs[ex_id, param_id]
                cum_prob = cum_probs[ex_id, param_group_id]
                norm_edge_prob = edge_prob / cum_prob

                param_grad = child_grad * td_probs[ex_id, parent_id]
                param_grads[orig_ex_id, param_id] = param_grad
                CUDA.@atomic cum_grads[ex_id, param_group_id] += param_grad * norm_edge_prob

                child_grad *= norm_edge_prob
            end

            # accumulate probability from child
            if isfirstedge || (edge_id == edge_start)  
                acc = child_grad
            else
                acc += child_grad
            end

            # write to global memory
            if islastedge || (edge_id == edge_end)   
                pid = edge.parent_id

                if islastedge && owned_node
                    # no one else is writing to this global memory
                    grads[ex_id, pid] = acc
                else
                    CUDA.@atomic grads[ex_id, pid] += acc
                end    
            end
        end
    end

    CUDA.sync_threads()

    if ex_id <= length(example_ids)
        for edge_id = edge_start:edge_end

            edge = edges[edge_id]
            parent_id = edge.parent_id

            if edge isa SumEdge
                orig_ex_id = example_ids[ex_id]
                param_id = edge2param[edge_id]
                param_group_id = param2group[param_id]
                edge_prob = edge_probs[ex_id, param_id]
                cum_prob = cum_probs[ex_id, param_group_id]
                norm_edge_prob = edge_prob / cum_prob

                param_grad = param_grads[orig_ex_id, param_id]
                if grad_wrt_logparams
                    param_grads[orig_ex_id, param_id] = (param_grad - cum_grads[ex_id, param_group_id]) * norm_edge_prob / temperature
                else
                    # we do not multiply by `p` because we want the grad be w.r.t. the probabilities, not the log-probs
                    param_grads[orig_ex_id, param_id] = (param_grad - cum_grads[ex_id, param_group_id]) / temperature 
                end
            end
        end
    end

    nothing
end

function gumbel_params_up_layer(grads, td_probs, param_grads, edge_probs, cum_probs, cum_grads, temperature, params, mbpc,
                                layer_start, layer_end, example_ids; mine, maxe, debug=false, grad_wrt_logparams=true)
    bpc = mbpc.bpc
    edges = bpc.edge_layers_up.vectors
    num_edges = layer_end-layer_start+1
    down2upedge = bpc.down2upedge
    edge2param = mbpc.edge2param
    param2group = mbpc.param2group
    num_examples = length(example_ids)

    dummy_args = (grads, td_probs, param_grads, edge_probs, cum_probs, cum_grads, 
                  edges, temperature, params, example_ids, edge2param,
                  param2group, Int32(32), Int32(1), Int32(1), Int32(2), grad_wrt_logparams)
    kernel = @cuda name="gumbel_params_up_layer" launch=false gumbel_params_up_layer_kernel(dummy_args...) 
    config = launch_configuration(kernel.fun)

    # configure thread/block balancing
    threads, blocks, num_example_threads, edge_work = 
        balance_threads(num_edges, num_examples, config; mine, maxe, contiguous_warps=true)
    
    args = (grads, td_probs, param_grads, edge_probs, cum_probs, cum_grads, 
            edges, temperature, params, example_ids, edge2param,
            param2group, Int32(num_example_threads), 
            Int32(layer_start), Int32(edge_work), Int32(layer_end), grad_wrt_logparams)
    if debug
        println("Layer $layer_start:$layer_end")
        @show threads blocks num_example_threads edge_work, num_edges num_examples
        CUDA.@time kernel(args...; threads, blocks)
    else
        kernel(args...; threads, blocks)
    end

    nothing
end

function gumbel_input_sample_up_kernel(td_probs, grads, param_grads, params, input_aggr_params, input_aggr_grads, nodes, num_inner_params, heap,
                                       input_node_ids, innode2ncumparam, example_ids, norm_params::Bool, num_ex_threads::Int32, node_work::Int32)
    
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = one(Int32) + (node_batch - one(Int32)) * node_work
    node_end = min(node_start + node_work - one(Int32), length(input_node_ids))

    @inbounds if ex_id <= length(example_ids)
        for node_id = node_start : node_end
            orig_ex_id::Int32 = example_ids[ex_id]
            orig_node_id::UInt32 = input_node_ids[node_id]
            td_prob::Float32 = td_probs[ex_id, orig_node_id]
            inputnode = nodes[orig_node_id]::BitsInput
            variable = inputnode.variable
            d = dist(inputnode)

            if d isa BitsFixableCategorical
                nparams = num_parameters(d, false)
                param_start_id = num_inner_params + innode2ncumparam[node_id] - nparams

                # gradient of input node
                innode_grad = zero(Float32)
                if nparams > 0
                    # d.fixed == false
                    if norm_params
                        for param_id = 1 : nparams
                            param_grads[ex_id, param_start_id + param_id] = input_aggr_grads[ex_id, variable, param_id] * td_prob
                            innode_grad += params[orig_ex_id, param_start_id + param_id] * input_aggr_grads[ex_id, variable, param_id]
                        end
                    else
                        for param_id = 1 : nparams
                            param_grads[ex_id, param_start_id + param_id] = input_aggr_grads[ex_id, variable, param_id] * td_prob
                            innode_grad += exp(params[orig_ex_id, param_start_id + param_id]) * input_aggr_grads[ex_id, variable, param_id]
                        end
                    end
                else
                    # d.fixed == true
                    heap_start = d.heap_start - 1
                    for param_id = 1 : d.num_cats
                        innode_grad += exp(heap[heap_start + param_id]) * input_aggr_grads[ex_id, variable, param_id]
                    end
                end
                grads[ex_id, orig_node_id] = innode_grad
            elseif d isa BitsGaussian
                param_start_id = num_inner_params + innode2ncumparam[node_id] - 2

                mu = params[orig_ex_id, param_start_id + 1]
                sigma = params[orig_ex_id, param_start_id + 2]

                # grad of mu
                param_grads[ex_id, param_start_id + 1] = input_aggr_grads[ex_id, variable, 1] * td_prob

                # grad of sigma
                param_grads[ex_id, param_start_id + 2] = input_aggr_grads[ex_id, variable, 2] * td_prob

                # grad of td_prob
                grads[ex_id, orig_node_id] = input_aggr_grads[ex_id, variable, 1] * mu + input_aggr_grads[ex_id, variable, 2] * sigma
            end
        end
    end

    nothing
end

function gumbel_input_sample_up(td_probs, grads, param_grads, params, input_aggr_params, input_aggr_grads,
                                mbpc::CuMetaBitsProbCircuit, example_ids; mine, maxe, norm_params=false, debug=false)

    bpc = mbpc.bpc

    num_examples = length(example_ids)
    num_input_nodes = length(bpc.input_node_ids)
    num_inner_params = length(mbpc.param2edge)

    dummy_args = (td_probs, grads, param_grads, params, input_aggr_params, input_aggr_grads, 
                  bpc.nodes, num_inner_params, bpc.heap, bpc.input_node_ids, 
                  mbpc.innode2ncumparam, example_ids, norm_params, Int32(1), Int32(1))
    kernel = @cuda name="gumbel_input_sample_up" launch=false gumbel_input_sample_up_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, node_work = 
        balance_threads(num_input_nodes, num_examples, config; mine, maxe)

    args = (td_probs, grads, param_grads, params, input_aggr_params, input_aggr_grads, 
            bpc.nodes, num_inner_params, bpc.heap, bpc.input_node_ids, 
            mbpc.innode2ncumparam, example_ids, norm_params, Int32(num_example_threads), Int32(node_work))
    if debug
        println("Gumbel backward of input nodes")
        @show threads blocks num_example_threads node_work num_nodes num_examples
        CUDA.@time kernel(args...; threads, blocks)
    else
        kernel(args...; threads, blocks)
    end

    nothing
end

function gumbel_sample_backward(td_probs, grads, param_grads, edge_td_probs, edge_probs, cum_probs, cum_grads, params, input_aggr_params,
                                input_aggr_grads, mbpc::CuMetaBitsProbCircuit, temperature, example_ids; mine, maxe, norm_params=false, 
                                debug=false, grad_wrt_logparams=true)
    
    grads .= zero(Float32)
    cum_grads .= zero(Float32)

    bpc = mbpc.bpc
    
    gumbel_input_sample_up(td_probs, grads, param_grads, params, input_aggr_params, input_aggr_grads,
                           mbpc, example_ids; mine, maxe, norm_params, debug)
    
    layer_start = 1
    for layer_end in bpc.edge_layers_up.ends
        gumbel_params_up_layer(grads, td_probs, param_grads, edge_probs, cum_probs, cum_grads, temperature, params, mbpc,
                               layer_start, layer_end, example_ids; mine, maxe, debug, grad_wrt_logparams)
        layer_start = layer_end + 1
    end
    
    nothing
end

"High-level API for backward Gumbel Sampling"
function gumbel_sample_backward(mbpc::CuMetaBitsProbCircuit, params, input_aggr_params, input_aggr_grads, temperature; reuse, 
                                reuse_grads = nothing, example_ids = nothing, mine = 2, maxe = 32, norm_params = false, grad_wrt_logparams = true)
    if example_ids === nothing
        example_ids = 1 : size(input_aggr_params, 1)
    end
    num_examples = length(example_ids)

    td_probs = cu(reuse[1])
    edge_td_probs = cu(reuse[2])
    edge_probs = cu(reuse[3])
    cum_probs = cu(reuse[4])

    n_nodes = num_nodes(mbpc)
    n_edges = num_edges(mbpc)
    n_params = num_parameters(mbpc)

    if reuse_grads !== nothing
        grads = cu(reuse_grads[1])
        param_grads = cu(reuse_grads[2])
        cum_grads = cu(reuse_grads[3])
    else
        grads = prep_memory(nothing, (num_examples, n_nodes), (false, true))
        param_grads = prep_memory(nothing, (num_examples, n_params), (false, true))
        cum_grads = prep_memory(nothing, (num_examples, mbpc.num_param_groups), (false, true))
    end

    if params isa Matrix
        params = cu(params)
    end
    if input_aggr_params isa Array
        input_aggr_params = cu(input_aggr_params)
    end
    if input_aggr_grads isa Array
        input_aggr_grads = cu(input_aggr_grads)
    end

    gumbel_sample_backward(td_probs, grads, param_grads, edge_td_probs, edge_probs, cum_probs, cum_grads, params, 
                           input_aggr_params, input_aggr_grads, mbpc, temperature, example_ids; mine, maxe, norm_params, grad_wrt_logparams)

    Array(param_grads)
end

############################################################
## Backward target computation for Gumbel softmax sampling 
############################################################

function gumbel_params_target_up_layer_kernel(td_probs, edge_td_probs, edge_probs, cum_probs, edges, params_target, target_td_probs,
                                              cum_par_buffer, example_ids, _edge2param, _param2group, step_size::Float32,
                                              num_ex_threads::Int32, layer_start::Int32, edge_work::Int32, layer_end::Int32)
    edge2param = Base.Experimental.Const(_edge2param)
    param2group = Base.Experimental.Const(_param2group)
        
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    edge_batch, ex_id = fldmod1(threadid, num_ex_threads)

    edge_start = layer_start + (edge_batch - one(Int32)) * edge_work 
    edge_end = min(edge_start + edge_work - one(Int32), layer_end)

    @inbounds if ex_id <= length(example_ids)

        for edge_id = edge_start:edge_end
            edge = edges[edge_id]
            if edge isa SumEdge
                param_id = edge2param[edge_id]
                param_group_id = param2group[param_id]
                cum_par_buffer[ex_id, param_group_id] = zero(Float32)
            end
        end

        CUDA.sync_threads()
        
        for edge_id = edge_start:edge_end
            edge = edges[edge_id]

            # target of the edge td_prob
            prime_weight = edge_td_probs[ex_id, edge_id] / td_probs[ex_id, edge.prime_id]
            target_edge_td_prob = target_td_probs[ex_id, edge.prime_id] * prime_weight
            if edge.sub_id != 0
                sub_weight = edge_td_probs[ex_id, edge_id] / td_probs[ex_id, edge.prime_id]
                target_edge_td_prob += target_td_probs[ex_id, edge.sub_id] * sub_weight
                target_edge_td_prob *= Float32(0.5)
            end

            if edge isa SumEdge
                orig_ex_id = example_ids[ex_id]
                param_id = edge2param[edge_id]
                param_group_id = param2group[param_id]

                # target edge parameter (unnormalized)
                param = max(target_edge_td_prob / td_probs[ex_id, edge.parent_id], 1e-8)
                params_target[orig_ex_id, param_id] = param
                CUDA.@atomic cum_par_buffer[ex_id, param_group_id] += param
            else # edge isa MulEdge
                target_td_probs[ex_id, edge.parent_id] = target_edge_td_prob
            end
        end

        CUDA.sync_threads()

        for edge_id = edge_start:edge_end
            edge = edges[edge_id]
            if edge isa SumEdge
                orig_ex_id = example_ids[ex_id]
                param_id = edge2param[edge_id]
                param_group_id = param2group[param_id]

                # logp
                edge_prob = edge_probs[ex_id, param_id]
                cum_prob = cum_probs[ex_id, param_group_id]
                norm_edge_prob = edge_prob / cum_prob

                param = params_target[orig_ex_id, param_id]
                cum_param = cum_par_buffer[ex_id, param_group_id]
                params_target[orig_ex_id, param_id] = log((one(Float32) - step_size) * norm_edge_prob + step_size * param / cum_param)

                target_td_probs[ex_id, edge.parent_id] = td_probs[ex_id, edge.parent_id] * cum_param
            end
        end
    end

    nothing
end

function gumbel_target_up_layer(td_probs, edge_td_probs, edge_probs, cum_probs, params_target, target_td_probs,
                                cum_par_buffer, mbpc, step_size, layer_start, layer_end, example_ids; mine, maxe, debug)
    bpc = mbpc.bpc
    edges = bpc.edge_layers_up.vectors
    num_edges = layer_end-layer_start+1
    down2upedge = bpc.down2upedge
    edge2param = mbpc.edge2param
    param2group = mbpc.param2group
    num_examples = length(example_ids)

    dummy_args = (td_probs, edge_td_probs, edge_probs, cum_probs, edges, params_target, target_td_probs,
                  cum_par_buffer, example_ids, edge2param, param2group, Float32(step_size), Int32(32), Int32(1), Int32(1), Int32(2))
    kernel = @cuda name="gumbel_params_target_up_layer" launch=false gumbel_params_target_up_layer_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    # configure thread/block balancing
    threads, blocks, num_example_threads, edge_work = 
        balance_threads(num_edges, num_examples, config; mine, maxe, contiguous_warps=true)
    
    args = (td_probs, edge_td_probs, edge_probs, cum_probs, edges, params_target, target_td_probs,
            cum_par_buffer, example_ids, edge2param, param2group, Float32(step_size), Int32(num_example_threads), 
            Int32(layer_start), Int32(edge_work), Int32(layer_end))
    
    kernel(args...; threads, blocks)

    nothing
end

function gumbel_input_target_sample_up_kernel(target_td_probs, input_aggr_targets, nodes, num_inner_params, heap,
                                              input_node_ids, innode2ncumparam, example_ids, norm_params::Bool, 
                                              num_ex_threads::Int32, node_work::Int32)
    
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = one(Int32) + (node_batch - one(Int32)) * node_work
    node_end = min(node_start + node_work - one(Int32), length(input_node_ids))

    @inbounds if ex_id <= length(example_ids)
        for node_id = node_start : node_end
            orig_ex_id::Int32 = example_ids[ex_id]
            orig_node_id::UInt32 = input_node_ids[node_id]
            inputnode = nodes[orig_node_id]::BitsInput
            variable = inputnode.variable
            d = dist(inputnode)

            if d isa BitsFixableCategorical
                nparams = num_parameters(d, false)
                param_start_id = num_inner_params + innode2ncumparam[node_id] - nparams

                # target td_prob
                target_td_prob = zero(Float32)
                if nparams > 0
                    # d.fixed == false
                    @assert false # not implemented
                else
                    # d.fixed == true
                    heap_start = d.heap_start - 1
                    for param_id = 1 : d.num_cats
                        target_td_prob += exp(heap[heap_start + param_id]) * input_aggr_targets[ex_id, variable, param_id]
                    end
                end
                target_td_probs[ex_id, orig_node_id] = target_td_prob
            elseif d isa BitsGaussian
                @assert false # not implemented
            end
        end
    end

    nothing
end

function gumbel_input_target_sample_up(target_td_probs, input_aggr_targets,
                                       mbpc, example_ids; mine, maxe, norm_params, debug)
    bpc = mbpc.bpc

    num_examples = length(example_ids)
    num_input_nodes = length(bpc.input_node_ids)
    num_inner_params = length(mbpc.param2edge)

    dummy_args = (target_td_probs, input_aggr_targets, bpc.nodes, num_inner_params, bpc.heap, bpc.input_node_ids, 
                  mbpc.innode2ncumparam, example_ids, norm_params, Int32(1), Int32(1))
    kernel = @cuda name="gumbel_input_target_sample_up" launch=false gumbel_input_target_sample_up_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, node_work = 
        balance_threads(num_input_nodes, num_examples, config; mine, maxe)

    args = (target_td_probs, input_aggr_targets, bpc.nodes, num_inner_params, bpc.heap, bpc.input_node_ids, 
            mbpc.innode2ncumparam, example_ids, norm_params, Int32(num_example_threads), Int32(node_work))
    if debug
        println("Flows of input nodes")
        @show threads blocks num_example_threads node_work num_nodes num_examples
        CUDA.@time kernel(args...; threads, blocks)
    else
        kernel(args...; threads, blocks)
    end

    nothing
end

function gumbel_sample_param_target(td_probs, edge_td_probs, edge_probs, cum_probs, params_target, 
                                    target_td_probs, cum_par_buffer, input_aggr_targets, 
                                    mbpc::CuMetaBitsProbCircuit, step_size, example_ids; mine, 
                                    maxe, norm_params, debug = false)
    bpc = mbpc.bpc
    target_td_probs .= zero(Float32)

    gumbel_input_target_sample_up(target_td_probs, input_aggr_targets,
                                  mbpc, example_ids; mine, maxe, norm_params, debug)

    layer_start = 1
    for layer_end in bpc.edge_layers_up.ends
        gumbel_target_up_layer(td_probs, edge_td_probs, edge_probs, cum_probs, params_target, target_td_probs, 
                               cum_par_buffer, mbpc, step_size, layer_start, layer_end, example_ids; mine, maxe, debug)
        layer_start = layer_end + 1
    end
    
    nothing
end

function gumbel_sample_param_target(mbpc::CuMetaBitsProbCircuit, input_aggr_targets; step_size, reuse,
                                    reuse_grads = nothing, example_ids = nothing, mine = 2, maxe = 32, norm_params = false)
    if example_ids === nothing
        example_ids = 1 : size(input_aggr_targets, 1)
    end
    num_examples = length(example_ids)

    td_probs = cu(reuse[1])
    edge_td_probs = cu(reuse[2])
    edge_probs = cu(reuse[3])
    cum_probs = cu(reuse[4])

    n_nodes = num_nodes(mbpc)
    n_edges = num_edges(mbpc)
    n_params = num_parameters(mbpc)
    n_pargroups = mbpc.num_param_groups

    if reuse_grads !== nothing
        params_target = cu(reuse_grads[1])
        target_td_probs = cu(reuse_grads[2])
        cum_par_buffer = cu(reuse_grads[3])
    else
        params_target = prep_memory(nothing, (num_examples, n_params), (false, true))
        target_td_probs = prep_memory(nothing, (num_examples, n_nodes), (false, true))
        cum_par_buffer = prep_memory(nothing, (num_examples, n_pargroups), (false, true))
    end

    if input_aggr_targets isa Array
        input_aggr_targets = cu(input_aggr_targets)
    end

    gumbel_sample_param_target(td_probs, edge_td_probs, edge_probs, cum_probs, params_target, target_td_probs,
                               cum_par_buffer, input_aggr_targets, mbpc, step_size, example_ids; mine, maxe, norm_params)

    Array(params_target)
end
