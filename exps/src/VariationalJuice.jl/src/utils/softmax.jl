
############################
# Log-softmax forward pass #
############################

function pc_softmax_kernel(params, cum_params, edges, edge2param, param2group, example_ids, layer_start::Int32, layer_end::Int32, 
                           num_ex_threads::Int32, edge_work::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    edge_batch, ex_id = fldmod1(threadid, num_ex_threads)

    edge_start = layer_start + (edge_batch - one(Int32)) * edge_work 
    edge_end = min(edge_start + edge_work - one(Int32), layer_end)

    @inbounds if ex_id <= length(example_ids)
        for edge_id = edge_start:edge_end
            edge = edges[edge_id]
            if edge isa SumEdge
                orig_ex_id = example_ids[ex_id]

                param_id = edge2param[edge_id]
                param_group_id = param2group[param_id]

                p = exp(params[orig_ex_id, param_id])
                CUDA.@atomic cum_params[orig_ex_id, param_group_id] += p
            end
        end

        CUDA.sync_threads()

        for edge_id = edge_start:edge_end
            edge = edges[edge_id]
            if edge isa SumEdge
                orig_ex_id = example_ids[ex_id]

                param_id = edge2param[edge_id]
                param_group_id = param2group[param_id]

                cum_logparam = log(cum_params[orig_ex_id, param_group_id])
                logp = params[orig_ex_id, param_id]
                params[orig_ex_id, param_id] = logp - cum_logparam
            end
        end
    end

    nothing
end

function pc_softmax(mbpc::CuMetaBitsProbCircuit, params, cum_params, example_ids; mine = 2, maxe = 32, debug = false)
    bpc = mbpc.bpc
    edges = bpc.edge_layers_up.vectors
    num_edges = length(edges)
    edge2param = mbpc.edge2param
    param2group = mbpc.param2group
    num_examples = length(example_ids)

    cum_params .= zero(Float32)

    dummy_args = (params, cum_params, edges, edge2param, param2group, example_ids, Int32(1), Int32(num_edges), Int32(1), Int32(1))
    kernel = @cuda name="pc_softmax" launch=false pc_softmax_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    # configure thread/block balancing
    threads, blocks, num_ex_threads, edge_work = 
        balance_threads(num_edges, num_examples, config; mine, maxe)
    
    args = (params, cum_params, edges, edge2param, param2group, example_ids, Int32(1), Int32(num_edges), Int32(num_ex_threads), Int32(edge_work))
    kernel(args...; threads, blocks)

    nothing
end

function pc_softmax(mbpc::CuMetaBitsProbCircuit, params; reuse = nothing, example_ids = nothing, mine = 2, maxe = 32, debug = false)
    if example_ids === nothing
        example_ids = 1 : size(params, 1)
    end
    num_examples = length(example_ids)

    n_pargroups = mbpc.num_param_groups

    if reuse !== nothing
        cum_params = reuse[1]
    else
        cum_params = prep_memory(nothing, (num_examples, n_pargroups), (false, true))
    end

    if params isa Matrix
        params = cu(params)
    end

    pc_softmax(mbpc, params, cum_params, example_ids; mine, maxe, debug)

    Array(params)
end

#############################
# Log-softmax backward pass #
#############################

function pc_softmax_backward_kernel(norm_params, grads, params_grad, cum_params_grad, edges, edge2param, param2group, example_ids, layer_start::Int32, layer_end::Int32, 
                                    num_ex_threads::Int32, edge_work::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    edge_batch, ex_id = fldmod1(threadid, num_ex_threads)

    edge_start = layer_start + (edge_batch - one(Int32)) * edge_work 
    edge_end = min(edge_start + edge_work - one(Int32), layer_end)

    @inbounds if ex_id <= length(example_ids)
        for edge_id = edge_start:edge_end
            edge = edges[edge_id]
            if edge isa SumEdge
                orig_ex_id = example_ids[ex_id]

                param_id = edge2param[edge_id]
                param_group_id = param2group[param_id]

                grad = grads[orig_ex_id, param_id]
                p = exp(norm_params[orig_ex_id, param_id])
                CUDA.@atomic cum_params_grad[orig_ex_id, param_group_id] += grad * p
            end
        end

        CUDA.sync_threads()

        for edge_id = edge_start:edge_end
            edge = edges[edge_id]
            if edge isa SumEdge
                orig_ex_id = example_ids[ex_id]

                param_id = edge2param[edge_id]
                param_group_id = param2group[param_id]

                grad = grads[orig_ex_id, param_id]
                p = exp(norm_params[orig_ex_id, param_id])
                params_grad[orig_ex_id, param_id] = (grad - cum_params_grad[orig_ex_id, param_group_id]) * p
            end
        end
    end

    nothing
end

function pc_softmax_backward(mbpc::CuMetaBitsProbCircuit, norm_params, grads, params_grad, cum_params_grad, example_ids; mine = 2, maxe = 32, debug = false)
    bpc = mbpc.bpc
    edges = bpc.edge_layers_up.vectors
    num_edges = length(edges)
    edge2param = mbpc.edge2param
    param2group = mbpc.param2group
    num_examples = length(example_ids)

    cum_params_grad .= zero(Float32)

    dummy_args = (norm_params, grads, params_grad, cum_params_grad, edges, edge2param, param2group, example_ids, Int32(1), Int32(num_edges), Int32(1), Int32(1))
    kernel = @cuda name="pc_softmax_backward" launch=false pc_softmax_backward_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    # configure thread/block balancing
    threads, blocks, num_ex_threads, edge_work = 
        balance_threads(num_edges, num_examples, config; mine, maxe)
    
    args = (norm_params, grads, params_grad, cum_params_grad, edges, edge2param, param2group, example_ids, Int32(1), Int32(num_edges), Int32(num_ex_threads), Int32(edge_work))
    kernel(args...; threads, blocks)

    nothing
end

function pc_softmax_backward(mbpc::CuMetaBitsProbCircuit, norm_params, grads; reuse_grad = nothing, example_ids = nothing, mine = 2, maxe = 32, debug = false)
    if example_ids === nothing
        example_ids = 1 : size(norm_params, 1)
    end
    num_examples = length(example_ids)

    n_params = num_parameters(mbpc)
    n_pargroups = mbpc.num_param_groups

    if reuse_grad !== nothing
        params_grad = reuse[1]
        cum_params_grad = reuse[2]
    else
        params_grad = prep_memory(nothing, (num_examples, n_params), (false, true))
        cum_params_grad = prep_memory(nothing, (num_examples, n_pargroups), (false, true))
    end

    if norm_params isa Matrix
        norm_params = cu(norm_params)
    end
    if grads isa Matrix
        grads = cu(grads)
    end

    pc_softmax_backward(mbpc, norm_params, grads, params_grad, cum_params_grad, example_ids; mine, maxe, debug)

    Array(params_grad)
end