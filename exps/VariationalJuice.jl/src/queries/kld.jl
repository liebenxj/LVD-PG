
###############################################################
## Input node forward pass for KLD (with matrix parameters)
###############################################################

function init_kld!_kernel(klds, nodes, params1, params2, innode2ncumparam, node2inputid, num_inner_params, example_ids, heap, 
                          norm_params::Bool, num_ex_threads::Int32, node_work::Int32)

    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = one(Int32) + (node_batch - one(Int32)) * node_work 
    node_end = min(node_start + node_work - one(Int32), length(nodes))

    if ex_id <= length(example_ids)
        for node_id = node_start:node_end

            node = nodes[node_id]
            
            klds[ex_id, node_id] = 
                if (node isa BitsSum) || (node isa BitsMul)
                    zero(Float32)
                else
                    orig_ex_id::Int32 = example_ids[ex_id]
                    input_node_id = node2inputid[node_id]

                    inputnode = node::BitsInput

                    param_start_id = num_inner_params + innode2ncumparam[input_node_id] - num_parameters(dist(inputnode), false) + 1

                    kl_div(dist(inputnode), heap, params1, params2, orig_ex_id, param_start_id, norm_params)
                end
        end
    end
    nothing
end

function init_kld!(klds, mbpc, params1, params2, example_ids; mine, maxe, norm_params, debug=false)
    bpc = mbpc.bpc
    num_examples = length(example_ids)
    num_nodes = length(bpc.nodes)
    innode2ncumparam = mbpc.innode2ncumparam
    node2inputid = mbpc.node2inputid
    num_inner_params = length(mbpc.param2edge)
    
    dummy_args = (klds, bpc.nodes, params1, params2, innode2ncumparam, node2inputid, num_inner_params,
                  example_ids, bpc.heap, norm_params, Int32(1), Int32(1))
    kernel = @cuda name="init_kld!" launch=false init_kld!_kernel(dummy_args...) 
    config = launch_configuration(kernel.fun)
    
    threads, blocks, num_example_threads, node_work = 
        balance_threads(num_nodes, num_examples, config; mine, maxe)
    
    args = (klds, bpc.nodes, params1, params2, innode2ncumparam, node2inputid, num_inner_params, 
            example_ids, bpc.heap, norm_params, Int32(num_example_threads), Int32(node_work))
    if debug
        println("Node initialization")
        @show threads blocks num_example_threads node_work num_nodes num_examples
        CUDA.@time kernel(args...; threads, blocks)
    else
        kernel(args...; threads, blocks)
    end
    nothing
end

###############################################################
## Inner node forward pass for KLD (with matrix parameters)
###############################################################

function kld_layer_up_kernel(klds, edge_klds, edges, params1, params2, edge2param, example_ids, num_ex_threads::Int32, 
                             layer_start::Int32, edge_work::Int32, layer_end::Int32)

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

            # compute KLDs coming from child
            kld = klds[ex_id, edge.prime_id]
            if edge.sub_id != 0
                kld += klds[ex_id, edge.sub_id]
            end
            if issum
                orig_ex_id = example_ids[ex_id]
                param_id = edge2param[edge_id] # use the specific parameter
                param1 = params1[orig_ex_id, param_id]
                param2 = params2[orig_ex_id, param_id]
                kld = exp(param1) * (kld + param1 - param2)
            end

            # record edge kld
            edge_klds[ex_id, edge_id] = kld

            # accumulate probability from child
            if isfirstedge || (edge_id == edge_start)  
                acc = kld
            else
                acc += kld
            end

            # write to global memory
            if islastedge || (edge_id == edge_end)   
                pid = edge.parent_id
                
                if islastedge && owned_node
                    # no one else is writing to this global memory
                    klds[ex_id, pid] = acc
                else
                    CUDA.@atomic klds[ex_id, pid] += acc
                end
            end
        end
    end
    nothing
end

function kld_layer_up(klds, edge_klds, bpc, params1, params2, edge2param, example_ids, layer_start, layer_end; mine, maxe, debug=false)
    edges = bpc.edge_layers_up.vectors
    num_edges = layer_end - layer_start + 1
    dummy_args = (klds, edge_klds, edges, params1, params2, edge2param, example_ids,
                  Int32(32), Int32(1), Int32(1), Int32(2))
    kernel = @cuda name="kld_layer_up" launch=false kld_layer_up_kernel(dummy_args...) 
    config = launch_configuration(kernel.fun)
    num_examples = length(example_ids)
    
    # configure thread/block balancing
    threads, blocks, num_example_threads, edge_work = 
        balance_threads(num_edges, num_examples, config; mine, maxe)
    
    args = (klds, edge_klds, edges, params1, params2, edge2param, example_ids,
            Int32(num_example_threads), Int32(layer_start), Int32(edge_work), Int32(layer_end))
    if debug
        println("Layer $layer_start:$layer_end")
        @show num_edges num_examples threads blocks num_example_threads edge_work
        CUDA.@time kernel(args...; threads, blocks)
    else
        kernel(args...; threads, blocks)
    end
    nothing
end

###############################################################
## Run entire forward pass
###############################################################

function kld(klds, edge_klds, mbpc::CuMetaBitsProbCircuit, params1, params2, example_ids; mine, maxe, norm_params=false, debug=false)
    bpc = mbpc.bpc

    init_kld!(klds, mbpc, params1, params2, example_ids; mine, maxe, norm_params, debug)
    layer_start = 1
    for layer_end in bpc.edge_layers_up.ends
        kld_layer_up(klds, edge_klds, bpc, params1, params2, mbpc.edge2param, example_ids, layer_start, layer_end; mine, maxe, debug)
        layer_start = layer_end + 1
    end
    nothing
end

"High-level API for KLD"
function kld(mbpc::CuMetaBitsProbCircuit, params1, params2; reuse = nothing, mine = 2, maxe = 32, norm_params = false, example_ids = nothing)
    if example_ids === nothing
        example_ids = 1 : size(params1, 1)
    end
    num_examples = length(example_ids)

    n_nodes = num_nodes(mbpc)
    n_edges = num_edges(mbpc)

    if reuse !== nothing
        klds = cu(reuse[1])
        edge_klds = cu(reuse[2])
    else
        klds = prep_memory(nothing, (num_examples, n_nodes), (false, true))
        edge_klds = prep_memory(nothing, (num_examples, n_edges), (false, true))
    end

    if params1 isa Matrix
        params1 = cu(params1)
    end
    if params2 isa Matrix
        params2 = cu(params2)
    end

    kld(klds, edge_klds, mbpc, params1, params2, example_ids; mine, maxe, norm_params)

    klds, edge_klds = Array(klds), Array(edge_klds)

    klds[:,end], (klds, edge_klds)
end

################################################################
## Inner node backward pass for KLD (with matrix parameters)
################################################################

function kld_layer_down_logparam_kernel(grads, edge_grads1, edge_grads2, edge_aggr, edges, params1, params2, edge2param, example_ids,
                                        _klds, _edge_klds, _down2upedge, num_ex_threads::Int32, num_examples::Int32, 
                                        layer_start::Int32, edge_work::Int32, layer_end::Int32)

    klds = Base.Experimental.Const(_klds)
    edge_klds = Base.Experimental.Const(_edge_klds)
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
            
            par_grad = grads[ex_id, parent_id]
            edge_grad = par_grad

            up_edge_id = down2upedge[edge_id]

            if issum
                param_id = edge2param[up_edge_id]
                orig_ex_id = example_ids[ex_id]
                logp1 = params1[orig_ex_id, param_id]
                p1 = exp(logp1)
                logp2 = params2[orig_ex_id, param_id]
                parent_kld = klds[ex_id, parent_id]
                edge_kld = edge_klds[ex_id, up_edge_id]
                edge_grad = edge_grad * p1

                edge_grads1[ex_id, up_edge_id] = par_grad * (edge_kld + p1)
                edge_grads2[ex_id, up_edge_id] = - par_grad * p1
            else
                # there is no edge parameters for MulEdge, so we simply set the gradients to zero
                edge_grads1[ex_id, up_edge_id] = zero(Float32)
                edge_grads2[ex_id, up_edge_id] = zero(Float32)
            end

            if sub_id != 0 
                if isonlysubedge(tag)
                    grads[ex_id, sub_id] = edge_grad
                else
                    CUDA.@atomic grads[ex_id, sub_id] += edge_grad
                end            
            end
        end
        
        # make sure this is run on all warp threads, regardless of `active`
        if !isnothing(edge_aggr)
            !active && (edge_grad = zero(Float32))
            edge_grad_warp = CUDA.reduce_warp(+, edge_grad)
            if warp_lane == 1
                CUDA.@atomic edge_aggr[edge_id] += edge_grad_warp
            end
        end

        if active

            # accumulate gradients from parents
            if firstedge || (edge_id == edge_start)  
                acc = edge_grad
            else
                acc += edge_grad
            end

            # write to global memory
            if lastedge || (edge_id == edge_end)   
                if lastedge && owned_node
                    # no one else is writing to this global memory
                    grads[ex_id, prime_id] = acc
                else
                    CUDA.@atomic grads[ex_id, prime_id] += acc
                end
            end
        end
        
    end

    nothing
end

function kld_layer_down_param_kernel(grads, edge_grads1, edge_grads2, edge_aggr, edges, params1, params2, edge2param, example_ids,
                                     _klds, _edge_klds, _down2upedge, num_ex_threads::Int32, num_examples::Int32, 
                                     layer_start::Int32, edge_work::Int32, layer_end::Int32)

    klds = Base.Experimental.Const(_klds)
    edge_klds = Base.Experimental.Const(_edge_klds)
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
            
            par_grad = grads[ex_id, parent_id]
            edge_grad = par_grad

            up_edge_id = down2upedge[edge_id]

            if issum
                param_id = edge2param[up_edge_id]
                orig_ex_id = example_ids[ex_id]
                logp1 = params1[orig_ex_id, param_id]
                p1 = exp(logp1)
                logp2 = params2[orig_ex_id, param_id]
                p2 = exp(logp2)
                parent_kld = klds[ex_id, parent_id]
                edge_kld = edge_klds[ex_id, up_edge_id]
                edge_grad = edge_grad * p1

                edge_grads1[ex_id, up_edge_id] = par_grad * (edge_kld + logp1 + 1)
                edge_grads2[ex_id, up_edge_id] = - par_grad * p1 / (p2 + 1e-6)
            else
                # there is no edge parameters for MulEdge, so we simply set the gradients to zero
                edge_grads1[ex_id, up_edge_id] = zero(Float32)
                edge_grads2[ex_id, up_edge_id] = zero(Float32)
            end

            if sub_id != 0 
                if isonlysubedge(tag)
                    grads[ex_id, sub_id] = edge_grad
                else
                    CUDA.@atomic grads[ex_id, sub_id] += edge_grad
                end            
            end
        end
        
        # make sure this is run on all warp threads, regardless of `active`
        if !isnothing(edge_aggr)
            !active && (edge_grad = zero(Float32))
            edge_grad_warp = CUDA.reduce_warp(+, edge_grad)
            if warp_lane == 1
                CUDA.@atomic edge_aggr[edge_id] += edge_grad_warp
            end
        end

        if active

            # accumulate gradients from parents
            if firstedge || (edge_id == edge_start)  
                acc = edge_grad
            else
                acc += edge_grad
            end

            # write to global memory
            if lastedge || (edge_id == edge_end)   
                if lastedge && owned_node
                    # no one else is writing to this global memory
                    grads[ex_id, prime_id] = acc
                else
                    CUDA.@atomic grads[ex_id, prime_id] += acc
                end
            end
        end
        
    end

    nothing
end

function kld_layer_down(grads, edge_grads1, edge_grads2, edge_aggr, bpc, klds, edge_klds, params1, params2, edge2param, 
                        layer_start, layer_end, example_ids; mine, maxe, debug=false, grad_wrt_logparams = true)
    edges = bpc.edge_layers_down.vectors
    num_edges = layer_end-layer_start+1
    down2upedge = bpc.down2upedge
    num_examples = length(example_ids)

    dummy_args = (grads, edge_grads1, edge_grads2, edge_aggr, edges, params1, params2, edge2param, example_ids, 
                  klds, edge_klds, down2upedge, 
                  Int32(32), Int32(num_examples), Int32(1), Int32(1), Int32(2))
    if grad_wrt_logparams
        kernel = @cuda name="kld_layer_down" launch=false kld_layer_down_logparam_kernel(dummy_args...)
    else
        kernel = @cuda name="kld_layer_down" launch=false kld_layer_down_param_kernel(dummy_args...)
    end
    config = launch_configuration(kernel.fun)

    # configure thread/block balancing
    threads, blocks, num_example_threads, edge_work = 
        balance_threads(num_edges, num_examples, config; mine, maxe, contiguous_warps=true)
    
    args = (grads, edge_grads1, edge_grads2, edge_aggr, edges, params1, params2, edge2param, example_ids, 
            klds, edge_klds, down2upedge, 
            Int32(num_example_threads), Int32(num_examples), Int32(layer_start), Int32(edge_work), Int32(layer_end))
    if debug
        println("Layer $layer_start:$layer_end")
        @show threads blocks num_example_threads edge_work, num_edges num_examples
        CUDA.@time kernel(args...; threads, blocks)
    else
        kernel(args...; threads, blocks)
    end
    nothing
end

###############################################################
## Run entire backward pass
###############################################################

function kld_backward_inner(grads, edge_grads1, edge_grads2, edge_aggr, mbpc::CuMetaBitsProbCircuit, 
                            params1, params2, klds, edge_klds, example_ids; mine, maxe, debug = false, grad_wrt_logparams = true)
    bpc = mbpc.bpc
    edge2param = mbpc.edge2param

    init_grads() = begin 
        grads .= zero(Float32)
        grads[:,end] .= one(Float32)
    end
    if debug
        println("Initializing gradients")
        CUDA.@time CUDA.@sync init_grads()
    else
        init_grads()
    end

    layer_start = 1
    for layer_end in bpc.edge_layers_down.ends
        kld_layer_down(grads, edge_grads1, edge_grads2, edge_aggr, bpc, klds, edge_klds,
                       params1, params2, edge2param, 
                       layer_start, layer_end, example_ids; 
                       mine, maxe, debug, grad_wrt_logparams)
        layer_start = layer_end + 1
    end
    nothing
end

################################################################
## Input node backward pass for KLD (with matrix parameters)
################################################################

function input_kld_down_kernel(grads, input_edge_grads1, input_edge_grads2, nodes, input_node_ids, params1, params2, num_inner_params,
                               example_ids, innode2ncumparam, norm_params::Bool, num_ex_threads::Int32, node_work::Int32)

    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = one(Int32) + (node_batch - one(Int32)) * node_work
    node_end = min(node_start + node_work - one(Int32), length(input_node_ids))

    if ex_id <= length(example_ids)
        for node_id = node_start : node_end
            orig_ex_id::Int32 = example_ids[ex_id]
            orig_node_id::UInt32 = input_node_ids[node_id]
            node_grad::Float32 = grads[ex_id, orig_node_id]
            inputnode = nodes[orig_node_id]::BitsInput
            variable = inputnode.variable
            d = dist(inputnode)

            nparams = num_parameters(d, false)
            if nparams > 0
                param_start_id = innode2ncumparam[node_id] - nparams
                if d isa BitsFixableCategorical
                    for param_id = 1 : nparams
                        param1 = params1[orig_ex_id, num_inner_params + param_start_id + param_id]
                        param2 = params2[orig_ex_id, num_inner_params + param_start_id + param_id]
                        grad1, grad2 = get_edge_kld_grad(d, node_grad, param1, param2, param_id, params1, params2, norm_params)
                        input_edge_grads1[ex_id, param_start_id + param_id] = grad1
                        input_edge_grads2[ex_id, param_start_id + param_id] = grad2
                    end
                elseif d isa BitsGaussian
                    grad_mu1, grad_mu2, grad_sigma1, grad_sigma2 = get_edge_kld_grad(
                        d, node_grad, params1, params2, ex_id, num_inner_params + param_start_id + 1, norm_params
                    )
                    input_edge_grads1[ex_id, param_start_id + 1] = grad_mu1
                    input_edge_grads1[ex_id, param_start_id + 2] = grad_sigma1
                    input_edge_grads2[ex_id, param_start_id + 1] = grad_mu2
                    input_edge_grads2[ex_id, param_start_id + 2] = grad_sigma2
                else
                    @assert false
                end
            end
        end
    end
    nothing
end

function input_kld_down(grads, input_edge_grads1, input_edge_grads2, mbpc::CuMetaBitsProbCircuit, 
                        params1, params2, example_ids; mine, maxe, norm_params=false, debug=false)
    bpc = mbpc.bpc

    num_examples = length(example_ids)
    num_input_nodes = length(bpc.input_node_ids)
    num_inner_params = length(mbpc.param2edge)

    dummy_args = (grads, input_edge_grads1, input_edge_grads2, bpc.nodes, bpc.input_node_ids,
                  params1, params2, num_inner_params, example_ids, mbpc.innode2ncumparam, norm_params, Int32(1), Int32(1))
    kernel = @cuda name="input_kld_down" launch=false input_kld_down_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, node_work = 
        balance_threads(num_input_nodes, num_examples, config; mine, maxe)

    args = (grads, input_edge_grads1, input_edge_grads2, bpc.nodes, bpc.input_node_ids,
            params1, params2, num_inner_params, example_ids, mbpc.innode2ncumparam, 
            norm_params, Int32(num_example_threads), Int32(node_work))
    if debug
        println("Flows of input nodes")
        @show threads blocks num_example_threads node_work num_nodes num_examples
        CUDA.@time kernel(args...; threads, blocks)
    else
        kernel(args...; threads, blocks)
    end

    nothing
end

###############################################################
## Run entire forward and backward pass
###############################################################

function kld_forward_backward(klds, edge_klds, grads, edge_grads1, edge_grads2, input_edge_grads1,
                              input_edge_grads2, mbpc::CuMetaBitsProbCircuit,
                              params1, params2, example_ids; mine, maxe, norm_params=false, debug=false)
    kld(klds, edge_klds, mbpc, params1, params2, example_ids; mine, maxe, debug)
    kld_backward_inner(grads, edge_grads1, edge_grads2, nothing, mbpc, 
                       params1, params2, klds, edge_klds, example_ids; mine, maxe, debug)
    input_kld_down(grads, input_edge_grads1, input_edge_grads2, mbpc, params1, params2, example_ids; mine, maxe, norm_params, debug)
end

"High-level API for KLD backward"
function kld_backward(mbpc::CuMetaBitsProbCircuit, params1, params2; reuse, reuse_grads = nothing,
                      example_ids = nothing, mine = 2, maxe = 32, norm_params = false, get_reuse_grad = false, grad_wrt_logparams = true)
    if example_ids === nothing
        example_ids = 1 : size(params1, 1)
    end
    num_examples = length(example_ids)

    klds = cu(reuse[1])
    edge_klds = cu(reuse[2])

    n_nodes = num_nodes(mbpc)
    n_edges = num_edges(mbpc)
    n_params = num_parameters(mbpc)
    num_input_params = mbpc.num_input_params

    if reuse_grads !== nothing
        grads = cu(reuse[1])
        edge_grads1 = cu(reuse[2])
        edge_grads2 = cu(reuse[3])
        input_edge_grads1 = cu(reuse[4])
        input_edge_grads2 = cu(reuse[5])
    else
        grads = prep_memory(nothing, (num_examples, n_nodes), (false, true))
        edge_grads1 = prep_memory(nothing, (num_examples, n_edges), (false, true))
        edge_grads2 = prep_memory(nothing, (num_examples, n_edges), (false, true))
        input_edge_grads1 = prep_memory(nothing, (num_examples, num_input_params), (false, true))
        input_edge_grads2 = prep_memory(nothing, (num_examples, num_input_params), (false, true))
    end

    if params1 isa Matrix
        params1 = cu(params1)
    end
    if params2 isa Matrix
        params2 = cu(params2)
    end

    # initialize buffer for parameter grads
    param_grads1 = prep_memory(nothing, (num_examples, n_params), (false, true))
    param_grads2 = prep_memory(nothing, (num_examples, n_params), (false, true))

    kld_backward_inner(grads, edge_grads1, edge_grads2, nothing, mbpc, params1, params2,
                       klds, edge_klds, example_ids; mine, maxe, debug = false, grad_wrt_logparams)
    input_kld_down(grads, input_edge_grads1, input_edge_grads2, mbpc, params1, params2, 
                   example_ids; mine, maxe, norm_params, debug = false)

    merge_params(mbpc, edge_grads1, input_edge_grads1, param_grads1)
    merge_params(mbpc, edge_grads2, input_edge_grads2, param_grads2)

    if get_reuse_grad
        reuse_grads = (Array(grads), Array(edge_grads1), Array(edge_grads2), 
                       Array(input_edge_grads1), Array(input_edge_grads2))

        Array(param_grads1), Array(param_grads2), reuse_grads
    else
        Array(param_grads1), Array(param_grads2)
    end
end

##############################################################
## New KLD update method - directly compute update target 
##############################################################

function init_kld_with_target!_kernel(klds, nodes, params1, params2, params1_target, newton_step_size, newton_nsteps, 
                                      innode2ncumparam, node2inputid, num_inner_params, example_ids, heap, 
                                      norm_params::Bool, num_ex_threads::Int32, node_work::Int32)

    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = one(Int32) + (node_batch - one(Int32)) * node_work 
    node_end = min(node_start + node_work - one(Int32), length(nodes))

    if ex_id <= length(example_ids)
        for node_id = node_start:node_end

            node = nodes[node_id]

            if (node isa BitsSum) || (node isa BitsMul)
                klds[ex_id, node_id] = zero(Float32)
            else
                orig_ex_id::Int32 = example_ids[ex_id]
                input_node_id = node2inputid[node_id]

                inputnode = node::BitsInput

                param_start_id = num_inner_params + innode2ncumparam[input_node_id] - num_parameters(dist(inputnode), false) + 1

                kld = kl_div(dist(inputnode), heap, params1, params2, orig_ex_id, param_start_id, norm_params)
                klds[ex_id, node_id] = kld

                # compute target parameters
                compute_target_params(dist(inputnode), heap, params1, params2, params1_target, newton_step_size, 
                                      newton_nsteps, orig_ex_id, param_start_id, norm_params)
            end
        end
    end
    nothing
end

function init_kld_with_target!(klds, mbpc, params1, params2, params1_target, newton_step_size, newton_nsteps, 
                               example_ids; mine, maxe, norm_params, debug=false)
    bpc = mbpc.bpc
    num_examples = length(example_ids)
    num_nodes = length(bpc.nodes)
    innode2ncumparam = mbpc.innode2ncumparam
    node2inputid = mbpc.node2inputid
    num_inner_params = length(mbpc.param2edge)
    
    dummy_args = (klds, bpc.nodes, params1, params2, params1_target, newton_step_size, newton_nsteps, 
                  innode2ncumparam, node2inputid, num_inner_params,
                  example_ids, bpc.heap, norm_params, Int32(1), Int32(1))
    kernel = @cuda name="init_kld_with_target!" launch=false init_kld_with_target!_kernel(dummy_args...) 
    config = launch_configuration(kernel.fun)
    
    threads, blocks, num_example_threads, node_work = 
        balance_threads(num_nodes, num_examples, config; mine, maxe)
    
    args = (klds, bpc.nodes, params1, params2, params1_target, newton_step_size, newton_nsteps, 
            innode2ncumparam, node2inputid, num_inner_params, 
            example_ids, bpc.heap, norm_params, Int32(num_example_threads), Int32(node_work))
    if debug
        println("Node initialization")
        @show threads blocks num_example_threads node_work num_nodes num_examples
        CUDA.@time kernel(args...; threads, blocks)
    else
        kernel(args...; threads, blocks)
    end
    nothing
end

function kld_copyparams_kernel(klds, edges, params1, params2, params1_target, par_buffer, cum_par_buffer,
                               cum_logp_buffer, edge2param, param2group, example_ids, newton_step_size::Float32, newton_nsteps, 
                               num_ex_threads::Int32, layer_start::Int32, edge_work::Int32, layer_end::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    edge_batch, ex_id = fldmod1(threadid, num_ex_threads)

    edge_start = layer_start + (edge_batch - one(Int32)) * edge_work 
    edge_end = min(edge_start + edge_work - one(Int32), layer_end)

    if ex_id <= length(example_ids)
        # copy parameters from `params1` to `params1_target`
        for edge_id = edge_start:edge_end
            edge = edges[edge_id]
            if edge isa SumEdge
                orig_ex_id = example_ids[ex_id]
                param_id = edge2param[edge_id]
                params1_target[orig_ex_id, param_id] = params1[orig_ex_id, param_id]
            end
        end
    end

    nothing
end

function kld_clear_buffer_kernel(klds, edges, params1, params2, params1_target, par_buffer, cum_par_buffer,
                                 cum_logp_buffer, edge2param, param2group, example_ids, newton_step_size::Float32, newton_nsteps, 
                                 num_ex_threads::Int32, layer_start::Int32, edge_work::Int32, layer_end::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    edge_batch, ex_id = fldmod1(threadid, num_ex_threads)

    edge_start = layer_start + (edge_batch - one(Int32)) * edge_work 
    edge_end = min(edge_start + edge_work - one(Int32), layer_end)

    @inbounds if ex_id <= length(example_ids)
        # clear `cum_par_buffer` and `cum_logp_buffer`
        for edge_id = edge_start:edge_end
            edge = edges[edge_id]
            if edge isa SumEdge
                param_id = edge2param[edge_id]
                param_group_id = param2group[param_id]
                cum_par_buffer[ex_id, param_group_id] = zero(Float32)
                cum_logp_buffer[ex_id, param_group_id] = zero(Float32)
            end
        end
    end

    nothing
end

function kld_update_params_kernel(klds, edges, params1, params2, params1_target, par_buffer, cum_par_buffer,
                                  cum_logp_buffer, edge2param, param2group, example_ids, newton_step_size::Float32, newton_nsteps, 
                                  num_ex_threads::Int32, layer_start::Int32, edge_work::Int32, layer_end::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    edge_batch, ex_id = fldmod1(threadid, num_ex_threads)

    edge_start = layer_start + (edge_batch - one(Int32)) * edge_work 
    edge_end = min(edge_start + edge_work - one(Int32), layer_end)

    @inbounds if ex_id <= length(example_ids)
        # compute updated parameters
        for edge_id = edge_start:edge_end
            edge = edges[edge_id]
            if edge isa SumEdge
                orig_ex_id = example_ids[ex_id]

                ch_kld = klds[ex_id, edge.prime_id]
                if edge.sub_id != 0
                    ch_kld += klds[ex_id, edge.sub_id]
                end

                param_id = edge2param[edge_id]
                param_group_id = param2group[param_id]
                param1 = params1_target[orig_ex_id, param_id]
                p1 = exp(param1)
                param2 = params2[orig_ex_id, param_id]

                newp = p1 - newton_step_size * p1 * (param1 + ch_kld - param2)
                newp = max(newp, 1e-8)
                par_buffer[ex_id, param_id] = newp
                CUDA.@atomic cum_par_buffer[ex_id, param_group_id] += newp
            end
        end
    end

    nothing
end

function kld_shift_params_kernel(klds, edges, params1, params2, params1_target, par_buffer, cum_par_buffer,
                                 cum_logp_buffer, edge2param, param2group, example_ids, newton_step_size::Float32, newton_nsteps, 
                                 num_ex_threads::Int32, layer_start::Int32, edge_work::Int32, layer_end::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    edge_batch, ex_id = fldmod1(threadid, num_ex_threads)

    edge_start = layer_start + (edge_batch - one(Int32)) * edge_work 
    edge_end = min(edge_start + edge_work - one(Int32), layer_end)

    @inbounds if ex_id <= length(example_ids)
        # shift parameters according to the lagrange factor
        for edge_id = edge_start:edge_end
            edge = edges[edge_id]
            if edge isa SumEdge
                orig_ex_id = example_ids[ex_id]

                param_id = edge2param[edge_id]
                param_group_id = param2group[param_id]
                param1 = params1_target[orig_ex_id, param_id]
                
                param = par_buffer[ex_id, param_id] + exp(param1) * (one(Float32) - cum_par_buffer[ex_id, param_group_id])
                param = max(param, 1e-8)
                
                params1_target[orig_ex_id, param_id] = param
                CUDA.@atomic cum_logp_buffer[orig_ex_id, param_group_id] += param
            end
        end
    end

    nothing
end

function kld_norm_params_kernel(klds, edges, params1, params2, params1_target, par_buffer, cum_par_buffer,
                                cum_logp_buffer, edge2param, param2group, example_ids, newton_step_size::Float32, newton_nsteps, 
                                num_ex_threads::Int32, layer_start::Int32, edge_work::Int32, layer_end::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    edge_batch, ex_id = fldmod1(threadid, num_ex_threads)

    edge_start = layer_start + (edge_batch - one(Int32)) * edge_work 
    edge_end = min(edge_start + edge_work - one(Int32), layer_end)

    @inbounds if ex_id <= length(example_ids)
        # normalize parameters
        for edge_id = edge_start:edge_end
            edge = edges[edge_id]
            if edge isa SumEdge
                orig_ex_id = example_ids[ex_id]

                param_id = edge2param[edge_id]
                param_group_id = param2group[param_id]

                param = params1_target[orig_ex_id, param_id]
                cum_param = cum_logp_buffer[orig_ex_id, param_group_id]
                params1_target[orig_ex_id, param_id] = log(param / cum_param)
            end
        end
    end

    nothing
end

function kld_with_target_layer_up(klds, bpc, params1, params2, params1_target, par_buffer, cum_par_buffer, cum_logp_buffer,
                                  edge2param, param2group, example_ids, newton_step_size, newton_nsteps, layer_start, 
                                  layer_end; mine, maxe, debug=false)
    edges = bpc.edge_layers_up.vectors
    num_edges = layer_end - layer_start + 1
    dummy_args = (klds, edges, params1, params2, params1_target, par_buffer, cum_par_buffer, cum_logp_buffer,
                  edge2param, param2group, example_ids, Float32(newton_step_size), newton_nsteps, Int32(32), Int32(1), Int32(1), Int32(2))
    # Copy parameters from `params1` to `params1_target`
    kernel1 = @cuda name="kld_copyparams" launch=false kld_copyparams_kernel(dummy_args...) 
    # Clear `cum_par_buffer` and `cum_logp_buffer`
    kernel2 = @cuda name="kld_clear_buffer" launch=false kld_clear_buffer_kernel(dummy_args...) 
    # 
    kernel3 = @cuda name="kld_update_params" launch=false kld_update_params_kernel(dummy_args...) 
    kernel4 = @cuda name="kld_shift_params" launch=false kld_shift_params_kernel(dummy_args...) 
    kernel5 = @cuda name="kld_norm_params" launch=false kld_norm_params_kernel(dummy_args...) 
    config = launch_configuration(kernel1.fun)
    num_examples = length(example_ids)
    
    # configure thread/block balancing
    threads, blocks, num_example_threads, edge_work = 
        balance_threads(num_edges, num_examples, config; mine, maxe)
    
    args = (klds, edges, params1, params2, params1_target, par_buffer, cum_par_buffer, cum_logp_buffer,
            edge2param, param2group, example_ids, Float32(newton_step_size), newton_nsteps,
            Int32(num_example_threads), Int32(layer_start), Int32(edge_work), Int32(layer_end))
    
    kernel1(args...; threads, blocks)
    for _ = 1 : newton_nsteps
        kernel2(args...; threads, blocks)
        kernel3(args...; threads, blocks)
        kernel4(args...; threads, blocks)
        kernel5(args...; threads, blocks)
    end

    nothing
end

function kld_with_target(klds, edge_klds, new_klds, new_edge_klds, mbpc::CuMetaBitsProbCircuit, params1, params2, params1_target, 
                         par_buffer, cum_par_buffer, cum_logp_buffer, example_ids, newton_step_size, newton_nsteps; 
                         mine, maxe, norm_params=false, debug=false)
    bpc = mbpc.bpc

    init_kld_with_target!(klds, mbpc, params1, params2, params1_target, newton_step_size, newton_nsteps, 
                          example_ids; mine, maxe, norm_params, debug)
    new_klds[:,:] .= klds[:,:]

    layer_start = 1
    # params1_target .= params1
    for layer_end in bpc.edge_layers_up.ends
        # compute KLD of the current layer with the original parameters
        kld_layer_up(klds, edge_klds, bpc, params1, params2, mbpc.edge2param, example_ids, layer_start, layer_end; mine, maxe, debug)
        
        # update the current layer's sum parameters
        kld_with_target_layer_up(new_klds, bpc, params1, params2, params1_target, 
                                 par_buffer, cum_par_buffer, cum_logp_buffer,
                                 mbpc.edge2param, mbpc.param2group, example_ids, newton_step_size, newton_nsteps, 
                                 layer_start, layer_end; mine, maxe, debug)

        kld_layer_up(new_klds, new_edge_klds, bpc, params1_target, params2, mbpc.edge2param, example_ids, layer_start, layer_end; mine, maxe, debug)
        
        layer_start = layer_end + 1
    end
    nothing
end

"High-level API for KLD"
function kld_with_update_target(mbpc::CuMetaBitsProbCircuit, params1, params2; newton_step_size, newton_nsteps, reuse = nothing, 
                                mine = 2, maxe = 32, norm_params = false, example_ids = nothing)
    if example_ids === nothing
        example_ids = 1 : size(params1, 1)
    end
    num_examples = length(example_ids)

    @assert 0.0 < newton_step_size <= 1.0
    @assert newton_nsteps >= 1

    n_nodes = num_nodes(mbpc)
    n_edges = num_edges(mbpc)
    n_pars = num_parameters(mbpc)
    n_pargroups = mbpc.num_param_groups

    if reuse !== nothing
        klds = cu(reuse[1])
        edge_klds = cu(reuse[2])
        params1_target = cu(reuse[3])
        par_buffer = cu(reuse[4])
        cum_par_buffer = cu(reuse[5])
        cum_logp_buffer = cu(reuse[6])
        new_klds = cu(reuse[7])
        new_edge_klds = cu(reuse[8])
    else
        klds = prep_memory(nothing, (num_examples, n_nodes), (false, true))
        edge_klds = prep_memory(nothing, (num_examples, n_edges), (false, true))
        params1_target = prep_memory(nothing, (num_examples, n_pars), (false, true))
        par_buffer = prep_memory(nothing, (num_examples, n_pars), (false, true))
        cum_par_buffer = prep_memory(nothing, (num_examples, n_pargroups), (false, true))
        cum_logp_buffer = prep_memory(nothing, (num_examples, n_pargroups), (false, true))
        new_klds = prep_memory(nothing, (num_examples, n_nodes), (false, true))
        new_edge_klds = prep_memory(nothing, (num_examples, n_edges), (false, true))
    end

    if params1 isa Matrix
        params1 = cu(params1)
    end
    if params2 isa Matrix
        params2 = cu(params2)
    end

    kld_with_target(klds, edge_klds, new_klds, new_edge_klds, mbpc, params1, params2, params1_target, par_buffer, cum_par_buffer,
                    cum_logp_buffer, example_ids, newton_step_size, newton_nsteps; mine, maxe, norm_params)

    klds, edge_klds, params1_target = Array(klds), Array(edge_klds), Array(params1_target)
    new_klds, new_edge_klds = Array(new_klds), Array(new_edge_klds)

    klds[:,end], params1_target, (klds, edge_klds, new_klds, new_edge_klds)
end