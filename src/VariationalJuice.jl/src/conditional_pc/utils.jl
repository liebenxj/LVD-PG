

#####################################
## log_softmax
#####################################

function log_softmax(cbpc::CuCondBitsProbCircuit, params::CuMatrix{Float32}; mine = 2, maxe = 32,
                     edge_groups_mem = nothing, input_groups_mem = nothing)
    num_examples = size(params, 1)
    nparams = num_parameters(cbpc)
    @assert nparams == size(params, 2) "$(nparams) != $(size(params, 2))"

    param2group = cbpc.param2group
    num_groups = maximum(param2group)
    innode2ncumparam = cbpc.innode2ncumparam
    input_node_ids = cbpc.bpc.input_node_ids

    edge_groups = prep_memory(edge_groups_mem, (num_examples, num_groups), (false, true))
    @inbounds @views edge_groups[:,:] .= typemin(Float32)
    input_groups = prep_memory(input_groups_mem, (num_examples, length(input_node_ids)), (false, true))
    @inbounds @views input_groups[:,:] .= typemin(Float32)

    log_softmax_edge_params(params, edge_groups, param2group; mine, maxe)
    log_softmax_input_params(params, cbpc, input_groups; mine, maxe)

    nothing
end

function cum_edge_log_params_kernel(params, edge_groups, param2group, num_ex_threads::Int32, param_work::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    param_batch, ex_id = fldmod1(threadid, num_ex_threads)

    param_start = one(Int32) + (param_batch - one(Int32)) * param_work
    param_end = min(param_start + param_work - one(Int32), length(param2group))

    @inbounds if ex_id <= size(params, 1)
        for param_id = param_start : param_end
            group_id = param2group[param_id]
            CUDA.@atomic edge_groups[ex_id, group_id] = logsumexp(edge_groups[ex_id, group_id], params[ex_id, param_id])
        end
    end
    nothing
end

function log_softmax_edge_params_kernel(params, edge_groups, param2group, num_ex_threads::Int32, param_work::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    param_batch, ex_id = fldmod1(threadid, num_ex_threads)

    param_start = one(Int32) + (param_batch - one(Int32)) * param_work
    param_end = min(param_start + param_work - one(Int32), length(param2group))

    @inbounds if ex_id <= size(params, 1)
        for param_id = param_start : param_end
            group_id = param2group[param_id]
            params[ex_id, param_id] = params[ex_id, param_id] - edge_groups[ex_id, group_id]
        end
    end
    nothing
end

function log_softmax_edge_params(params, edge_groups, param2group; mine, maxe)
    num_examples = size(params, 1)

    # Initialize
    @inbounds @views edge_groups .= typemin(Float32)

    dummy_args = (params, edge_groups, param2group, Int32(1), Int32(1))
    kernel1 = @cuda name="cum_edge_log_params" launch=false cum_edge_log_params_kernel(dummy_args...)
    config = launch_configuration(kernel1.fun)

    threads, blocks, num_example_threads, param_work = 
        balance_threads(length(param2group), num_examples, config; mine, maxe)

    args = (params, edge_groups, param2group, Int32(num_example_threads), Int32(param_work))
    kernel1(args...; threads, blocks)

    dummy_args = (params, edge_groups, param2group, Int32(1), Int32(1))
    kernel2 = @cuda name="log_softmax_edge_params" launch=false log_softmax_edge_params_kernel(dummy_args...)
    config = launch_configuration(kernel2.fun)

    threads, blocks, num_example_threads, param_work = 
        balance_threads(length(param2group), num_examples, config; mine, maxe)

    args = (params, edge_groups, param2group, Int32(num_example_threads), Int32(param_work))
    kernel2(args...; threads, blocks)
    nothing
end

function cum_input_log_params_kernel(params, input_groups, nodes, input_node_ids, innode2ncumparam, num_inner_params,
                                     num_ex_threads::Int32, node_work::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = one(Int32) + (node_batch - one(Int32)) * node_work
    node_end = min(node_start + node_work - one(Int32), length(input_node_ids))

    @inbounds if ex_id <= size(params, 1)
        for idx = node_start : node_end
            node_id = input_node_ids[idx]
            node = nodes[node_id]::BitsInput
            d = dist(node)

            nparams = num_parameters(d, false)
            if nparams > 0
                param_id_base = num_inner_params + innode2ncumparam[idx] - nparams
                for param_id = 1 : nparams
                    CUDA.@atomic input_groups[ex_id, idx] = logsumexp(input_groups[ex_id, idx], params[ex_id, param_id_base + param_id])
                end
            end
        end
    end
    nothing
end

function log_softmax_input_params_kernel(params, input_groups, nodes, input_node_ids, innode2ncumparam, num_inner_params,
                                         num_ex_threads::Int32, node_work::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = one(Int32) + (node_batch - one(Int32)) * node_work
    node_end = min(node_start + node_work - one(Int32), length(input_node_ids))

    @inbounds if ex_id <= size(params, 1)
        for idx = node_start : node_end
            node_id = input_node_ids[idx]
            node = nodes[node_id]::BitsInput
            d = dist(node)

            nparams = num_parameters(d, false)
            if nparams > 0
                param_id_base = num_inner_params + innode2ncumparam[idx] - nparams
                for param_id = 1 : nparams
                    params[ex_id, param_id_base + param_id] = params[ex_id, param_id_base + param_id] - input_groups[ex_id, idx]
                end
            end
        end
    end
    nothing
end

function log_softmax_input_params(params, mbpc, input_groups; mine = 2, maxe = 32)
    bpc = mbpc.bpc
    num_inner_params = length(mbpc.param2edge)
    innode2ncumparam = mbpc.innode2ncumparam
    input_node_ids = bpc.input_node_ids
    nodes = bpc.nodes

    # Initialize
    @inbounds @views input_groups .= typemin(Float32)

    dummy_args = (params, input_groups, nodes, input_node_ids, innode2ncumparam, num_inner_params, Int32(1), Int32(1))
    kernel1 = @cuda name="cum_input_log_params" launch=false cum_input_log_params_kernel(dummy_args...)
    config = launch_configuration(kernel1.fun)

    threads, blocks, num_example_threads, node_work = 
        balance_threads(length(input_node_ids), size(params, 1), config; mine, maxe)

    args = (params, input_groups, nodes, input_node_ids, innode2ncumparam, num_inner_params, 
            Int32(num_example_threads), Int32(node_work))
    kernel1(args...; threads, blocks)

    dummy_args = (params, input_groups, nodes, input_node_ids, innode2ncumparam, num_inner_params, Int32(1), Int32(1))
    kernel2 = @cuda name="log_softmax_input_params" launch=false log_softmax_input_params_kernel(dummy_args...)
    config = launch_configuration(kernel2.fun)

    threads, blocks, num_example_threads, node_work = 
        balance_threads(length(input_node_ids), size(params, 1), config; mine, maxe)

    args = (params, input_groups, nodes, input_node_ids, innode2ncumparam, num_inner_params, 
            Int32(num_example_threads), Int32(node_work))
    kernel2(args...; threads, blocks)
    nothing
end

#####################################
## gradients of log_softmax
#####################################

"We assume the input `params` is the output of `log_softmax` in the forward pass"
function log_softmax_grad(cbpc::CuCondBitsProbCircuit, params::CuMatrix{Float32}, grads::CuMatrix{Float32}; mine = 2, maxe = 32,
                          edge_groups_mem = nothing, input_groups_mem = nothing)
    num_examples = size(params, 1)
    nparams = num_parameters(cbpc)
    @assert nparams == size(params, 2) "$(nparams) != $(size(params, 2))"
    @assert size(params, 1) == size(grads, 1)
    @assert size(params, 2) == size(grads, 2)

    param2group = cbpc.param2group
    num_groups = maximum(param2group)
    innode2ncumparam = cbpc.innode2ncumparam
    input_node_ids = cbpc.bpc.input_node_ids

    edge_groups = prep_memory(edge_groups_mem, (num_examples, num_groups), (false, true))
    @inbounds @views edge_groups[:,:] .= zero(Float32)
    input_groups = prep_memory(input_groups_mem, (num_examples, length(input_node_ids)), (false, true))
    @inbounds @views input_groups[:,:] .= zero(Float32)

    log_softmax_edge_params_grad(params, grads, edge_groups, param2group; mine, maxe)
    log_softmax_input_params_grad(params, grads, cbpc, input_groups; mine, maxe)

    nothing
end

function cum_edge_log_params_grad_kernel(params, grads, edge_groups, param2group, num_ex_threads::Int32, param_work::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    param_batch, ex_id = fldmod1(threadid, num_ex_threads)

    param_start = one(Int32) + (param_batch - one(Int32)) * param_work
    param_end = min(param_start + param_work - one(Int32), length(param2group))

    @inbounds if ex_id <= size(params, 1)
        for param_id = param_start : param_end
            group_id = param2group[param_id]
            CUDA.@atomic edge_groups[ex_id, group_id] += grads[ex_id, param_id]
        end
    end
    nothing
end

function log_softmax_edge_params_grad_kernel(params, grads, edge_groups, param2group, num_ex_threads::Int32, param_work::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    param_batch, ex_id = fldmod1(threadid, num_ex_threads)

    param_start = one(Int32) + (param_batch - one(Int32)) * param_work
    param_end = min(param_start + param_work - one(Int32), length(param2group))

    @inbounds if ex_id <= size(params, 1)
        for param_id = param_start : param_end
            group_id = param2group[param_id]
            param = exp(params[ex_id, param_id])
            grads[ex_id, param_id] = grads[ex_id, param_id] - edge_groups[ex_id, group_id] * param
        end
    end
    nothing
end

function log_softmax_edge_params_grad(params, grads, edge_groups, param2group; mine, maxe)
    num_examples = size(params, 1)

    # Initialize
    @inbounds @views edge_groups .= zero(Float32)

    dummy_args = (params, grads, edge_groups, param2group, Int32(1), Int32(1))
    kernel1 = @cuda name="cum_edge_log_params_grad" launch=false cum_edge_log_params_grad_kernel(dummy_args...)
    config = launch_configuration(kernel1.fun)

    threads, blocks, num_example_threads, param_work = 
        balance_threads(length(param2group), num_examples, config; mine, maxe)

    args = (params, grads, edge_groups, param2group, Int32(num_example_threads), Int32(param_work))
    kernel1(args...; threads, blocks)

    dummy_args = (params, grads, edge_groups, param2group, Int32(1), Int32(1))
    kernel2 = @cuda name="log_softmax_edge_params_grad" launch=false log_softmax_edge_params_grad_kernel(dummy_args...)
    config = launch_configuration(kernel2.fun)

    threads, blocks, num_example_threads, param_work = 
        balance_threads(length(param2group), num_examples, config; mine, maxe)

    args = (params, grads, edge_groups, param2group, Int32(num_example_threads), Int32(param_work))
    kernel2(args...; threads, blocks)
    nothing
end

function cum_input_log_params_grad_kernel(params, grads, input_groups, nodes, input_node_ids, innode2ncumparam, num_inner_params,
                                          num_ex_threads::Int32, node_work::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = one(Int32) + (node_batch - one(Int32)) * node_work
    node_end = min(node_start + node_work - one(Int32), length(input_node_ids))

    @inbounds if ex_id <= size(params, 1)
        for idx = node_start : node_end
            node_id = input_node_ids[idx]
            node = nodes[node_id]::BitsInput
            d = dist(node)

            nparams = num_parameters(d, false)
            if nparams > 0
                param_id_base = num_inner_params + innode2ncumparam[idx] - nparams
                for param_id = 1 : nparams
                    CUDA.@atomic input_groups[ex_id, idx] += grads[ex_id, param_id_base + param_id]
                end
            end
        end
    end
    nothing
end

function log_softmax_input_params_grad_kernel(params, grads, input_groups, nodes, input_node_ids, innode2ncumparam, num_inner_params,
                                              num_ex_threads::Int32, node_work::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = one(Int32) + (node_batch - one(Int32)) * node_work
    node_end = min(node_start + node_work - one(Int32), length(input_node_ids))

    @inbounds if ex_id <= size(params, 1)
        for idx = node_start : node_end
            node_id = input_node_ids[idx]
            node = nodes[node_id]::BitsInput
            d = dist(node)

            nparams = num_parameters(d, false)
            if nparams > 0
                param_id_base = num_inner_params + innode2ncumparam[idx] - nparams
                for param_id = 1 : nparams
                    param = exp(params[ex_id, param_id_base + param_id])
                    grads[ex_id, param_id_base + param_id] = grads[ex_id, param_id_base + param_id] - input_groups[ex_id, idx] * param
                end
            end
        end
    end
    nothing
end

function log_softmax_input_params_grad(params, grads, mbpc, input_groups; mine = 2, maxe = 32)
    bpc = mbpc.bpc
    num_inner_params = length(mbpc.param2edge)
    innode2ncumparam = mbpc.innode2ncumparam
    input_node_ids = bpc.input_node_ids
    nodes = bpc.nodes

    # Initialize
    @inbounds @views input_groups .= zero(Float32)

    dummy_args = (params, grads, input_groups, nodes, input_node_ids, innode2ncumparam, num_inner_params, Int32(1), Int32(1))
    kernel1 = @cuda name="cum_input_log_params_grad" launch=false cum_input_log_params_grad_kernel(dummy_args...)
    config = launch_configuration(kernel1.fun)

    threads, blocks, num_example_threads, node_work = 
        balance_threads(length(input_node_ids), size(params, 1), config; mine, maxe)

    args = (params, grads, input_groups, nodes, input_node_ids, innode2ncumparam, num_inner_params, 
            Int32(num_example_threads), Int32(node_work))
    kernel1(args...; threads, blocks)

    dummy_args = (params, grads, input_groups, nodes, input_node_ids, innode2ncumparam, num_inner_params, Int32(1), Int32(1))
    kernel2 = @cuda name="log_softmax_input_params_grad" launch=false log_softmax_input_params_grad_kernel(dummy_args...)
    config = launch_configuration(kernel2.fun)

    threads, blocks, num_example_threads, node_work = 
        balance_threads(length(input_node_ids), size(params, 1), config; mine, maxe)

    args = (params, grads, input_groups, nodes, input_node_ids, innode2ncumparam, num_inner_params, 
            Int32(num_example_threads), Int32(node_work))
    kernel2(args...; threads, blocks)
    nothing
end

##########################################
## Check if parameters are normalized
##########################################

function check_params_normalized(cbpc::CuCondBitsProbCircuit, params::CuMatrix{Float32}; mine = 2, maxe = 32,
                                 edge_groups_mem = nothing, input_groups_mem = nothing, logspace = false,
                                 edge_only = false)
    num_examples = size(params, 1)
    nparams = num_parameters(cbpc)
    @assert nparams == size(params, 2) "$(nparams) != $(size(params, 2))"

    param2group = cbpc.param2group
    num_groups = maximum(param2group)
    innode2ncumparam = cbpc.innode2ncumparam
    input_node_ids = cbpc.bpc.input_node_ids

    edge_groups = prep_memory(edge_groups_mem, (num_examples, num_groups), (false, true))
    input_groups = prep_memory(input_groups_mem, (num_examples, length(input_node_ids)), (false, true))

    flag1 = check_edge_params_normalized(params, edge_groups, param2group; mine, maxe, logspace)
    flag2 = check_input_params_normalized(params, cbpc, input_groups; mine, maxe, logspace)

    flag1 && (flag2 || edge_only)
end

function check_edge_params_normalized_kernel(params, edge_groups, param2group, 
                                             num_ex_threads::Int32, param_work::Int32, logspace::Bool)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    param_batch, ex_id = fldmod1(threadid, num_ex_threads)

    param_start = one(Int32) + (param_batch - one(Int32)) * param_work
    param_end = min(param_start + param_work - one(Int32), length(param2group))

    @inbounds if ex_id <= size(params, 1)
        for param_id = param_start : param_end
            group_id = param2group[param_id]
            if logspace
                CUDA.@atomic edge_groups[ex_id, group_id] = logsumexp(edge_groups[ex_id, group_id], params[ex_id, param_id])
            else
                CUDA.@atomic edge_groups[ex_id, group_id] += params[ex_id, param_id]
            end
        end
    end
    nothing
end

function check_edge_params_normalized(params, edge_groups, param2group; mine, maxe, logspace)
    num_examples = size(params, 1)

    # Initialize
    if logspace
        @inbounds @views edge_groups .= typemin(Float32)
    else
        @inbounds @views edge_groups .= zero(Float32)
    end

    dummy_args = (params, edge_groups, param2group, Int32(1), Int32(1), logspace)
    kernel = @cuda name="check_edge_params_normalized" launch=false check_edge_params_normalized_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, param_work = 
        balance_threads(length(param2group), num_examples, config; mine, maxe)

    args = (params, edge_groups, param2group, Int32(num_example_threads), Int32(param_work), logspace)
    kernel(args...; threads, blocks)
    
    if logspace
        all(isapprox.(Array(edge_groups), zeros(Float32, size(edge_groups)); atol = 1e-5))
    else
        all(isapprox.(Array(edge_groups), ones(Float32, size(edge_groups)); atol = 1e-5))
    end
end

function check_input_params_normalized_kernel(params, input_groups, nodes, input_node_ids, innode2ncumparam, num_inner_params,
                                              num_ex_threads::Int32, node_work::Int32, logspace::Bool)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = one(Int32) + (node_batch - one(Int32)) * node_work
    node_end = min(node_start + node_work - one(Int32), length(input_node_ids))

    @inbounds if ex_id <= size(params, 1)
        for idx = node_start : node_end
            node_id = input_node_ids[idx]
            node = nodes[node_id]::BitsInput
            d = dist(node)

            nparams = num_parameters(d, false)
            if nparams > 0
                param_id_base = num_inner_params + innode2ncumparam[idx] - nparams
                for param_id = 1 : nparams
                    if logspace
                        input_groups[ex_id, idx] = logsumexp(input_groups[ex_id, idx], params[ex_id, param_id_base + param_id])
                    else
                        input_groups[ex_id, idx] += params[ex_id, param_id_base + param_id]
                    end
                end
            end
        end
    end
    nothing
end

function check_input_params_normalized(params, mbpc, input_groups; mine = 2, maxe = 32, logspace)
    bpc = mbpc.bpc
    num_inner_params = length(mbpc.param2edge)
    innode2ncumparam = mbpc.innode2ncumparam
    input_node_ids = bpc.input_node_ids
    nodes = bpc.nodes

    # Initialize
    if logspace
        @inbounds @views input_groups .= typemin(Float32)
    else
        @inbounds @views input_groups .= zero(Float32)
    end

    dummy_args = (params, input_groups, nodes, input_node_ids, innode2ncumparam, num_inner_params, 
                  Int32(1), Int32(1), logspace)
    kernel = @cuda name="check_input_params_normalized" launch=false check_input_params_normalized_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, node_work = 
        balance_threads(length(input_node_ids), size(params, 1), config; mine, maxe)

    args = (params, input_groups, nodes, input_node_ids, innode2ncumparam, num_inner_params, 
            Int32(num_example_threads), Int32(node_work), logspace)
    kernel(args...; threads, blocks)

    if logspace
        all(isapprox.(Array(input_groups), zeros(Float32, size(input_groups)); atol = 1e-5))
    else
        all(isapprox.(Array(input_groups), ones(Float32, size(input_groups)); atol = 1e-5))
    end
end

##########################################
## Normalize parameters
##########################################

function normalize_parameters(cbpc::CuCondBitsProbCircuit, params::CuMatrix{Float32}; mine = 2, maxe = 32,
                              edge_groups_mem = nothing, input_groups_mem = nothing, logspace = false, pseudocount = 0.0)
    num_examples = size(params, 1)
    nparams = num_parameters(cbpc)

    @assert nparams == size(params, 2) "$(nparams) != $(size(params, 2))"
    

    param2group = cbpc.param2group
    num_groups = maximum(param2group)
    innode2ncumparam = cbpc.innode2ncumparam
    input_node_ids = cbpc.bpc.input_node_ids

    edge_groups = prep_memory(edge_groups_mem, (num_examples, num_groups), (false, true))
    input_groups = prep_memory(input_groups_mem, (num_examples, length(input_node_ids)), (false, true))

    normalize_edge_params(params, edge_groups, param2group; mine, maxe, logspace, pseudocount)
    normalize_input_params(params, cbpc, input_groups; mine, maxe, logspace, pseudocount)
    
    nothing
end

function normalize_edge_params_kernel1(params, edge_groups, param2group, num_ex_threads::Int32, param_work::Int32, logspace::Bool, pseudocount::Float32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    param_batch, ex_id = fldmod1(threadid, num_ex_threads)

    param_start = one(Int32) + (param_batch - one(Int32)) * param_work
    param_end = min(param_start + param_work - one(Int32), length(param2group))

    @inbounds if ex_id <= size(params, 1)
        for param_id = param_start : param_end
            group_id = param2group[param_id]
            CUDA.@atomic edge_groups[ex_id, group_id] += one(Float32)
        end
    end
    nothing
end

function normalize_edge_params_kernel2(params, edge_groups, param2group, num_ex_threads::Int32, param_work::Int32, logspace::Bool, pseudocount::Float32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    param_batch, ex_id = fldmod1(threadid, num_ex_threads)

    param_start = one(Int32) + (param_batch - one(Int32)) * param_work
    param_end = min(param_start + param_work - one(Int32), length(param2group))

    @inbounds if ex_id <= size(params, 1)
        for param_id = param_start : param_end
            group_id = param2group[param_id]
            if logspace
                params[ex_id, param_id] = logsumexp(params[ex_id, param_id], log(pseudocount / edge_groups[ex_id, group_id]))
            else
                params[ex_id, param_id] += pseudocount / edge_groups[ex_id, group_id]
            end
        end
    end
    nothing
end

function normalize_edge_params_kernel3(params, edge_groups, param2group, num_ex_threads::Int32, param_work::Int32, logspace::Bool, pseudocount::Float32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    param_batch, ex_id = fldmod1(threadid, num_ex_threads)

    param_start = one(Int32) + (param_batch - one(Int32)) * param_work
    param_end = min(param_start + param_work - one(Int32), length(param2group))

    @inbounds if ex_id <= size(params, 1)
        for param_id = param_start : param_end
            group_id = param2group[param_id]
            if logspace
                CUDA.@atomic edge_groups[ex_id, group_id] = logsumexp(edge_groups[ex_id, group_id], params[ex_id, param_id])
            else
                CUDA.@atomic edge_groups[ex_id, group_id] += params[ex_id, param_id]
            end
        end
    end
    nothing
end

function normalize_edge_params_kernel4(params, edge_groups, param2group, num_ex_threads::Int32, param_work::Int32, logspace::Bool, pseudocount::Float32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    param_batch, ex_id = fldmod1(threadid, num_ex_threads)

    param_start = one(Int32) + (param_batch - one(Int32)) * param_work
    param_end = min(param_start + param_work - one(Int32), length(param2group))

    @inbounds if ex_id <= size(params, 1)
        for param_id = param_start : param_end
            group_id = param2group[param_id]
            if logspace
                params[ex_id, param_id] -= edge_groups[ex_id, group_id]
            else
                params[ex_id, param_id] /= edge_groups[ex_id, group_id] + Float32(1e-8)
            end
        end
    end
    nothing
end

function normalize_edge_params(params, edge_groups, param2group; mine, maxe, logspace, pseudocount)
    num_examples = size(params, 1)

    # Step #1
    @inbounds @views edge_groups .= zero(Float32)

    # Step #2
    dummy_args = (params, edge_groups, param2group, Int32(1), Int32(1), logspace, Float32(pseudocount))
    kernel1 = @cuda name="normalize_edge_params1" launch=false normalize_edge_params_kernel1(dummy_args...)
    config = launch_configuration(kernel1.fun)

    threads, blocks, num_example_threads, param_work = 
        balance_threads(length(param2group), num_examples, config; mine, maxe)

    args = (params, edge_groups, param2group, Int32(num_example_threads), Int32(param_work), logspace, Float32(pseudocount))
    CUDA.@sync kernel1(args...; threads, blocks)

    # Step #3
    dummy_args = (params, edge_groups, param2group, Int32(1), Int32(1), logspace, Float32(pseudocount))
    kernel2 = @cuda name="normalize_edge_params2" launch=false normalize_edge_params_kernel2(dummy_args...)
    config = launch_configuration(kernel2.fun)

    threads, blocks, num_example_threads, param_work = 
        balance_threads(length(param2group), num_examples, config; mine, maxe)

    args = (params, edge_groups, param2group, Int32(num_example_threads), Int32(param_work), logspace, Float32(pseudocount))
    CUDA.@sync kernel2(args...; threads, blocks)

    # Step #4
    if logspace
        @inbounds @views edge_groups .= typemin(Float32)
    else
        @inbounds @views edge_groups .= zero(Float32)
    end

    # Step #5
    dummy_args = (params, edge_groups, param2group, Int32(1), Int32(1), logspace, Float32(pseudocount))
    kernel3 = @cuda name="normalize_edge_params3" launch=false normalize_edge_params_kernel3(dummy_args...)
    config = launch_configuration(kernel3.fun)

    threads, blocks, num_example_threads, param_work = 
        balance_threads(length(param2group), num_examples, config; mine, maxe)

    args = (params, edge_groups, param2group, Int32(num_example_threads), Int32(param_work), logspace, Float32(pseudocount))
    CUDA.@sync kernel3(args...; threads, blocks)

    # Step #6
    dummy_args = (params, edge_groups, param2group, Int32(1), Int32(1), logspace, Float32(pseudocount))
    kernel4 = @cuda name="normalize_edge_params4" launch=false normalize_edge_params_kernel4(dummy_args...)
    config = launch_configuration(kernel4.fun)

    threads, blocks, num_example_threads, param_work = 
        balance_threads(length(param2group), num_examples, config; mine, maxe)

    args = (params, edge_groups, param2group, Int32(num_example_threads), Int32(param_work), logspace, Float32(pseudocount))
    CUDA.@sync kernel4(args...; threads, blocks)

    nothing
end

function normalize_input_params_kernel(params, input_groups, nodes, input_node_ids, innode2ncumparam, num_inner_params,
                                       num_ex_threads::Int32, node_work::Int32, logspace::Bool, pseudocount::Float32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = one(Int32) + (node_batch - one(Int32)) * node_work
    node_end = min(node_start + node_work - one(Int32), length(input_node_ids))

    @inbounds if ex_id <= size(params, 1)
        for idx = node_start : node_end
            node_id = input_node_ids[idx]
            node = nodes[node_id]::BitsInput
            d = dist(node)

            nparams = num_parameters(d, false)
            if nparams > 0
                param_id_base = num_inner_params + innode2ncumparam[idx] - nparams

                for param_id = 1 : nparams
                    if logspace
                        params[ex_id, param_id_base + param_id] = logsumexp(params[ex_id, param_id_base + param_id], log(pseudocount / nparams))
                    else
                        params[ex_id, param_id_base + param_id] += pseudocount / nparams
                    end
                end

                if logspace
                    cum = typemin(Float32)
                else
                    cum = zero(Float32)
                end

                for param_id = 1 : nparams
                    if logspace
                        cum = logsumexp(cum, params[ex_id, param_id_base + param_id])
                    else
                        cum += params[ex_id, param_id_base + param_id]
                    end
                end

                for param_id = 1 : nparams
                    if logspace
                        params[ex_id, param_id_base + param_id] -= cum
                    else
                        params[ex_id, param_id_base + param_id] /= cum
                    end
                end
            end
        end
    end
    nothing
end

function normalize_input_params(params, mbpc, input_groups; mine = 2, maxe = 32, logspace, pseudocount)
    bpc = mbpc.bpc
    num_inner_params = length(mbpc.param2edge)
    innode2ncumparam = mbpc.innode2ncumparam
    input_node_ids = bpc.input_node_ids
    nodes = bpc.nodes

    # Initialize
    if logspace
        @inbounds @views input_groups .= typemin(Float32)
    else
        @inbounds @views input_groups .= zero(Float32)
    end

    dummy_args = (params, input_groups, nodes, input_node_ids, innode2ncumparam, num_inner_params, 
                  Int32(1), Int32(1), logspace, Float32(pseudocount))
    kernel = @cuda name="normalize_input_params" launch=false normalize_input_params_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, node_work = 
        balance_threads(length(input_node_ids), size(params, 1), config; mine, maxe)

    args = (params, input_groups, nodes, input_node_ids, innode2ncumparam, num_inner_params, 
            Int32(num_example_threads), Int32(node_work), logspace, Float32(pseudocount))
    kernel(args...; threads, blocks)
    
    nothing
end

##########################################
## Update parameters to bpc
##########################################

function update_parameters_to_bpc(cbpc::CuCondBitsProbCircuit, params::CuMatrix{Float32}; reverse = false)
    @assert size(params, 1) == 1
    nparams = num_parameters(cbpc)
    @assert nparams == size(params, 2) "$(nparams) != $(size(params, 2))"

    edge2param = cbpc.edge2param

    update_edge_parameters_to_bpc(cbpc.bpc, params, edge2param, reverse)
    update_input_parameters_to_bpc(cbpc.bpc, params, cbpc, reverse)

    nothing
end

function update_edge_parameters_to_bpc_kernel(edges_down, edges_up, _down2upedge, params, _edge2param, reverse)
    down2upedge = Base.Experimental.Const(_down2upedge)
    edge2param = Base.Experimental.Const(_edge2param)
    
    edge_id_down = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    @inbounds if edge_id_down <= length(edges_down)
        edge_down = edges_down[edge_id_down]
        if edge_down isa SumEdge 
            
            edge_id_up = down2upedge[edge_id_down]
            # only difference is the tag
            edge_up_tag = edges_up[edge_id_up].tag

            if !(isfirst(edge_up_tag) && islast(edge_up_tag))
                parent_id = edge_down.parent_id

                if !reverse
                    logp = params[1, edge2param[edge_id_up]]

                    edges_down[edge_id_down] = 
                        SumEdge(parent_id, edge_down.prime_id, edge_down.sub_id, 
                                logp, edge_down.tag)

                    edges_up[edge_id_up] = 
                        SumEdge(parent_id, edge_down.prime_id, edge_down.sub_id, 
                                logp, edge_up_tag)
                else
                    logp = edge_down.logp
                    idx = edge2param[edge_id_up]
                    params[1,idx] = logp
                end
            end
        end
    end      
    nothing
end

function update_edge_parameters_to_bpc(bpc::CuBitsProbCircuit, params::CuMatrix{Float32}, edge2param, reverse)
    edges_down = bpc.edge_layers_down.vectors
    edges_up = bpc.edge_layers_up.vectors
    down2upedge = bpc.down2upedge
    @assert length(edges_down) == length(down2upedge) == length(edges_up)

    args = (edges_down, edges_up, down2upedge, params, edge2param, reverse)
    kernel = @cuda name="update_edge_parameters_to_bpc" launch=false update_edge_parameters_to_bpc_kernel(args...) 
    threads = launch_configuration(kernel.fun).threads
    blocks = cld(length(edges_down), threads)
    
    kernel(args...; threads, blocks)
    
    nothing
end

function update_input_parameters_to_bpc_kernel(nodes, input_node_ids, heap, params, num_inner_params, innode2ncumparam, reverse)

    node_id = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    @inbounds if node_id <= length(input_node_ids)
        orig_node_id::UInt32 = input_node_ids[node_id]
        inputnode = nodes[orig_node_id]::BitsInput
        direct_update_params(dist(inputnode), heap, params, num_inner_params, innode2ncumparam, node_id, reverse)
    end
    nothing
end

function update_input_parameters_to_bpc(bpc, params, cbpc, reverse)
    num_input_nodes = length(bpc.input_node_ids)

    num_inner_params = length(cbpc.param2edge)
    innode2ncumparam = cbpc.innode2ncumparam

    args = (bpc.nodes, bpc.input_node_ids, bpc.heap, params, num_inner_params, innode2ncumparam, reverse)
    kernel = @cuda name="update_input_parameters_to_bpc" launch=false update_input_parameters_to_bpc_kernel(args...)
    threads = launch_configuration(kernel.fun).threads
    blocks = cld(num_input_nodes, threads)

    kernel(args...; threads, blocks)
    
    nothing
end

function direct_update_params(d::BitsCategorical, heap, params, num_inner_params, innode2ncumparam, input_node_id, reverse)
    heap_start = d.heap_start
    num_cats = d.num_cats

    @inbounds begin
        param_start_id = num_inner_params + innode2ncumparam[input_node_id] - num_cats + 1
        for i = 0 : num_cats-1
            if !reverse
                heap[heap_start+i] = params[1, param_start_id+i]
            else
                params[1,param_start_id+i] = heap[heap_start+i]
            end
        end
    end
    nothing
end

#################################
## Check parameters are equal
#################################

function check_params_equal(cbpc::CuCondBitsProbCircuit, params::CuMatrix{Float32}; atol = 1e-6)
    @assert size(params, 1) == 1
    nparams = num_parameters(cbpc)
    @assert nparams == size(params, 2) "$(nparams) != $(size(params, 2))"

    equal_flag = CUDA.zeros(Bool, nparams)

    edge2param = cbpc.edge2param

    check_edge_params_equal(cbpc.bpc, params, edge2param, equal_flag; atol)
    check_input_params_equal(cbpc.bpc, params, cbpc, equal_flag; atol)

    # all(Array(equal_flag))
    Array(equal_flag)
end

function check_edge_params_equal_kernel(edges_down, edges_up, _down2upedge, _params, _edge2param, equal_flag, atol)
    down2upedge = Base.Experimental.Const(_down2upedge)
    params = Base.Experimental.Const(_params)
    edge2param = Base.Experimental.Const(_edge2param)
    
    edge_id_down = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    @inbounds if edge_id_down <= length(edges_down)
        edge_down = edges_down[edge_id_down]
        if edge_down isa SumEdge 
            
            edge_id_up = down2upedge[edge_id_down]
            edge_up_tag = edges_up[edge_id_up].tag

            if !(isfirst(edge_up_tag) && islast(edge_up_tag))
                parent_id = edge_down.parent_id

                logp = params[1, edge2param[edge_id_up]]

                edge_logp = edge_down.logp
                
                if abs(logp - edge_logp) < atol
                    equal_flag[edge2param[edge_id_up]] = true
                else
                    equal_flag[edge2param[edge_id_up]] = false
                end
            else
                if abs(params[1, edge2param[edge_id_up]]) < atol
                    equal_flag[edge2param[edge_id_up]] = true
                else
                    equal_flag[edge2param[edge_id_up]] = false
                end
            end
        end
    end      
    nothing
end

function check_edge_params_equal(bpc::CuBitsProbCircuit, params::CuMatrix{Float32}, edge2param, equal_flag; atol)
    edges_down = bpc.edge_layers_down.vectors
    edges_up = bpc.edge_layers_up.vectors
    down2upedge = bpc.down2upedge
    @assert length(edges_down) == length(down2upedge) == length(edges_up)

    args = (edges_down, edges_up, down2upedge, params, edge2param, equal_flag, Float32(atol))
    kernel = @cuda name="check_edge_params_equal" launch=false check_edge_params_equal_kernel(args...) 
    threads = launch_configuration(kernel.fun).threads
    blocks = cld(length(edges_down), threads)
    
    kernel(args...; threads, blocks)
    
    nothing
end

function check_input_params_equal_kernel(nodes, input_node_ids, heap, params, num_inner_params, innode2ncumparam, equal_flag, atol)

    node_id = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    @inbounds if node_id <= length(input_node_ids)
        orig_node_id::UInt32 = input_node_ids[node_id]
        inputnode = nodes[orig_node_id]::BitsInput
        check_input_params_equal(dist(inputnode), heap, params, num_inner_params, innode2ncumparam, node_id, equal_flag, atol)
    end
    nothing
end

function check_input_params_equal(bpc, params, cbpc, equal_flag; atol)
    num_input_nodes = length(bpc.input_node_ids)

    num_inner_params = length(cbpc.param2edge)
    innode2ncumparam = cbpc.innode2ncumparam

    args = (bpc.nodes, bpc.input_node_ids, bpc.heap, params, num_inner_params, innode2ncumparam, equal_flag, Float32(atol))
    kernel = @cuda name="check_input_params_equal" launch=false check_input_params_equal_kernel(args...)
    threads = launch_configuration(kernel.fun).threads
    blocks = cld(num_input_nodes, threads)

    kernel(args...; threads, blocks)
    
    nothing
end

function check_input_params_equal(d::BitsCategorical, heap, params, num_inner_params, innode2ncumparam, input_node_id, equal_flag, atol)
    heap_start = d.heap_start
    num_cats = d.num_cats

    @inbounds begin
        param_start_id = num_inner_params + innode2ncumparam[input_node_id] - num_cats + 1
        for i = 0 : num_cats-1
            if abs(heap[heap_start+i] - params[1, param_start_id+i]) < atol
                equal_flag[param_start_id+i] = true
            else
                equal_flag[param_start_id+i] = false
            end
        end
    end
    nothing
end
