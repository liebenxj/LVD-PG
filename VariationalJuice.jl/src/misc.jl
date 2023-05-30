using ProbabilisticCircuits: BitsInput, prep_memory


"Copy the parameters from `mbpc` to `params`"
function vectorize_parameters(mbpc::CuMetaBitsProbCircuit, params::CuVector{Float32})
    nparams = num_parameters(mbpc)
    @assert nparams == length(params) "$(nparams) != $(length(params))"

    bpc = mbpc.bpc
    nodes = bpc.nodes
    edges = bpc.edge_layers_up.vectors
    input_node_ids = bpc.input_node_ids
    heap = bpc.heap
    edge2param = mbpc.edge2param
    innode2ncumparam = mbpc.innode2ncumparam
    num_inner_params = length(mbpc.param2edge)

    args = (edges, edge2param, params)
    kernel1 = @cuda name="vectorize_parameters_inner" launch=false vectorize_parameters_inner_kernel(args...)
    threads = launch_configuration(kernel1.fun).threads
    blocks = cld(length(edges), threads)

    kernel1(args...; threads, blocks)

    args = (nodes, input_node_ids, heap, params, innode2ncumparam, num_inner_params)
    kernel2 = @cuda name="vectorize_parameters_input" launch=false vectorize_parameters_input_kernel(args...)
    threads = launch_configuration(kernel2.fun).threads
    blocks = cld(length(input_node_ids), threads)

    kernel2(args...; threads, blocks)
    nothing
end
function vectorize_parameters(mbpc::CuMetaBitsProbCircuit)
    nparams = num_parameters(mbpc)
    params = cu(zeros(Float32, nparams))

    vectorize_parameters(mbpc, params)

    Array(params)
end

function vectorize_parameters_inner_kernel(edges, edge2param, params)
    edge_id = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x
    @inbounds if edge_id <= length(edges)
        edge = edges[edge_id]
        if edge isa SumEdge
            param_id = edge2param[edge_id]
            params[param_id] = edge.logp
        end
    end
    nothing
end

function vectorize_parameters_input_kernel(nodes, input_node_ids, heap, params, innode2ncumparam, num_inner_params)
    idx = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x
    @inbounds if idx <= length(input_node_ids)
        node_id = input_node_ids[idx]
        node = nodes[node_id]::BitsInput
        d = node.dist
        nparams = num_parameters(d, false)
        if nparams > 0
            for param_id = 1 : nparams
                params[num_inner_params + innode2ncumparam[idx] - nparams + param_id] = get_param(d, param_id, heap)
            end
        end
    end
    nothing
end

import ProbabilisticCircuits: update_parameters # extend

"Copy the parameters from `params` to `mbpc`"
function update_parameters(mbpc::CuMetaBitsProbCircuit, params::CuVector{Float32})
    nparams = num_parameters(mbpc)
    @assert nparams == length(params) "$(nparams) != $(length(params))"

    bpc = mbpc.bpc
    nodes = bpc.nodes
    edges_up = bpc.edge_layers_up.vectors
    edges_down = bpc.edge_layers_down.vectors
    up2downedge = mbpc.up2downedge
    input_node_ids = bpc.input_node_ids
    heap = bpc.heap
    param2edge = mbpc.param2edge
    innode2ncumparam = mbpc.innode2ncumparam
    num_inner_params = length(mbpc.param2edge)

    args = (edges_up, edges_down, up2downedge, param2edge, params)
    kernel1 = @cuda name="update_params_inner" launch=false update_params_inner_kernel(args...)
    threads = launch_configuration(kernel1.fun).threads
    blocks = cld(length(edges_up), threads)

    kernel1(args...; threads, blocks)

    args = (nodes, input_node_ids, heap, params, innode2ncumparam, num_inner_params)
    kernel2 = @cuda name="update_params_input" launch=false update_params_input_kernel(args...)
    threads = launch_configuration(kernel2.fun).threads
    blocks = cld(length(input_node_ids), threads)

    kernel2(args...; threads, blocks)
    nothing
end

function update_params_inner_kernel(edges_up, edges_down, up2downedge, param2edge, params)
    param_id = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x
    if param_id <= length(param2edge)
        param = params[param_id]
        edge_up_id = param2edge[param_id]
        edge_down_id = up2downedge[edge_up_id]

        edge_up = edges_up[edge_up_id]
        edges_up[edge_up_id] = SumEdge(edge_up.parent_id, edge_up.prime_id, edge_up.sub_id, param, edge_up.tag)

        edge_down = edges_down[edge_down_id]
        edges_down[edge_down_id] = SumEdge(edge_down.parent_id, edge_down.prime_id, edge_down.sub_id, param, edge_down.tag)
    end
    nothing
end

function update_params_input_kernel(nodes, input_node_ids, heap, params, innode2ncumparam, num_inner_params)
    idx = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x
    if idx <= length(input_node_ids)
        node_id = input_node_ids[idx]
        node = nodes[node_id]::BitsInput
        d = node.dist
        nparams = num_parameters(d)
        if nparams > 0
            for param_id = 1 : nparams
                param = params[num_inner_params + innode2ncumparam[idx] - nparams + param_id]
                set_param(d, param_id, param, heap)
            end
        end
    end
    nothing
end

function normalize_parameters(mbpc::CuMetaBitsProbCircuit, params::CuMatrix{Float32}; is_log_params::Bool, mine = 2, maxe = 32,
                              edge_groups_mem = nothing, input_groups_mem = nothing)
    num_examples = size(params, 1)
    nparams = num_parameters(mbpc)
    @assert nparams == size(params, 2) "$(nparams) != $(size(params, 2))"

    param2group = mbpc.param2group
    num_groups = maximum(param2group)
    innode2ncumparam = mbpc.innode2ncumparam
    input_node_ids = mbpc.bpc.input_node_ids

    edge_groups = prep_memory(edge_groups_mem, (num_examples, num_groups), (false, true))
    @inbounds @views edge_groups[:,:] .= zero(Float32)
    input_groups = prep_memory(input_groups_mem, (num_examples, length(input_node_ids)), (false, true))
    @inbounds @views input_groups[:,:] .= zero(Float32)

    normalize_edge_params(params, edge_groups, param2group, is_log_params; mine, maxe)
    normalize_input_params(params, mbpc, input_groups, is_log_params; mine, maxe)
    nothing
end

function cum_edge_params_kernel(params, edge_groups, param2group, is_log_params, num_ex_threads::Int32, param_work::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    param_batch, ex_id = fldmod1(threadid, num_ex_threads)

    param_start = one(Int32) + (param_batch - one(Int32)) * param_work
    param_end = min(param_start + param_work - one(Int32), length(param2group))

    @inbounds if ex_id <= size(params, 1)
        for param_id = param_start : param_end
            group_id = param2group[param_id]
            param = if is_log_params
                    exp(params[ex_id, param_id])
                else
                    params[ex_id, param_id]
                end
            CUDA.@atomic edge_groups[ex_id, group_id] += param
        end
    end
    nothing
end

function normalize_edge_params_kernel(params, edge_groups, param2group, is_log_params, num_ex_threads::Int32, param_work::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    param_batch, ex_id = fldmod1(threadid, num_ex_threads)

    param_start = one(Int32) + (param_batch - one(Int32)) * param_work
    param_end = min(param_start + param_work - one(Int32), length(param2group))

    @inbounds if ex_id <= size(params, 1)
        for param_id = param_start : param_end
            group_id = param2group[param_id]
            param = if is_log_params
                    exp(params[ex_id, param_id])
                else
                    params[ex_id, param_id]
                end
            params[ex_id, param_id] = log(param / edge_groups[ex_id, group_id])
        end
    end
    nothing
end

function normalize_edge_params(params, edge_groups, param2group, is_log_params; mine, maxe)
    num_examples = size(params, 1)

    dummy_args = (params, edge_groups, param2group, is_log_params, Int32(1), Int32(1))
    kernel1 = @cuda name="cum_edge_params" launch=false cum_edge_params_kernel(dummy_args...)
    config = launch_configuration(kernel1.fun)

    threads, blocks, num_example_threads, param_work = 
        balance_threads(length(param2group), num_examples, config; mine, maxe)

    args = (params, edge_groups, param2group, is_log_params, Int32(num_example_threads), Int32(param_work))
    kernel1(args...; threads, blocks)

    dummy_args = (params, edge_groups, param2group, is_log_params, Int32(1), Int32(1))
    kernel2 = @cuda name="normalize_edge_params" launch=false normalize_edge_params_kernel(dummy_args...)
    config = launch_configuration(kernel2.fun)

    threads, blocks, num_example_threads, param_work = 
        balance_threads(length(param2group), num_examples, config; mine, maxe)

    args = (params, edge_groups, param2group, is_log_params, Int32(num_example_threads), Int32(param_work))
    kernel2(args...; threads, blocks)
    nothing
end

function cum_input_params_kernel(params, input_groups, nodes, input_node_ids, innode2ncumparam, num_inner_params,
                                 is_log_params, num_ex_threads::Int32, node_work::Int32)
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
                    param = if is_log_params
                            exp(params[ex_id, param_id_base + param_id])
                        else
                            params[ex_id, param_id_base + param_id]
                        end
                    CUDA.@atomic input_groups[ex_id, idx] += param
                end
            end
        end
    end
    nothing
end

function normalize_input_params_kernel(params, input_groups, nodes, input_node_ids, innode2ncumparam, num_inner_params,
                                 is_log_params, num_ex_threads::Int32, node_work::Int32)
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
                    param = if is_log_params
                            exp(params[ex_id, param_id_base + param_id])
                        else
                            params[ex_id, param_id_base + param_id]
                        end
                    params[ex_id, param_id_base + param_id] = log(param / input_groups[ex_id, idx])
                end
            end
        end
    end
    nothing
end

function normalize_input_params(params, mbpc, input_groups, is_log_params; mine = 2, maxe = 32)
    bpc = mbpc.bpc
    num_inner_params = length(mbpc.param2edge)
    innode2ncumparam = mbpc.innode2ncumparam
    input_node_ids = bpc.input_node_ids
    nodes = bpc.nodes

    dummy_args = (params, input_groups, nodes, input_node_ids, innode2ncumparam, num_inner_params, is_log_params, Int32(1), Int32(1))
    kernel1 = @cuda name="cum_input_params" launch=false cum_input_params_kernel(dummy_args...)
    config = launch_configuration(kernel1.fun)

    threads, blocks, num_example_threads, node_work = 
        balance_threads(length(input_node_ids), size(params, 1), config; mine, maxe)

    args = (params, input_groups, nodes, input_node_ids, innode2ncumparam, num_inner_params, 
            is_log_params, Int32(num_example_threads), Int32(node_work))
    kernel1(args...; threads, blocks)

    dummy_args = (params, input_groups, nodes, input_node_ids, innode2ncumparam, num_inner_params, is_log_params, Int32(1), Int32(1))
    kernel2 = @cuda name="normalize_input_params" launch=false normalize_input_params_kernel(dummy_args...)
    config = launch_configuration(kernel2.fun)

    threads, blocks, num_example_threads, node_work = 
        balance_threads(length(input_node_ids), size(params, 1), config; mine, maxe)

    args = (params, input_groups, nodes, input_node_ids, innode2ncumparam, num_inner_params, 
            is_log_params, Int32(num_example_threads), Int32(node_work))
    kernel2(args...; threads, blocks)
    nothing
end

param_buffer_size(pc::ProbCircuit) = maximum(map(n->param_buffer_size(dist(n)), inputnodes(pc)))
param_buffer_size(mbpc::CuMetaBitsProbCircuit) = begin
    pbuf_size = 0
    nodes = Array(mbpc.bpc.nodes)
    for node in nodes
        if node isa BitsInput
            ncat = dist(node).num_cats
            if ncat > pbuf_size
                pbuf_size = ncat
            end
        end
    end
    pbuf_size
end

function preprocess_input_params_kernel(nodes, input_node_ids, innode2ncumparam, num_inner_params, params,
                                        process_func, num_examples::Int32, num_ex_threads::Int32, node_work::Int32)

    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = one(Int32) + (node_batch - one(Int32)) * node_work
    node_end = min(node_start + node_work - one(Int32), length(input_node_ids))

    @inbounds if ex_id <= num_examples
        for node_id = node_start : node_end
            orig_node_id::UInt32 = input_node_ids[node_id]
            inputnode = nodes[orig_node_id]::BitsInput
            d = dist(inputnode)::InputDist

            nparams = num_parameters(d, false)
            param_start_id = num_inner_params + innode2ncumparam[node_id] - nparams

            process_func(d, params, ex_id, param_start_id)
        end
    end

    nothing
end

function normalize_params_step(d::BitsFixableCategorical, params, ex_id, param_start_id)
    if !d.fixed
        num_cats = d.num_cats
        cum_val = zero(Float32)
        for i = 1 : num_cats
            param = exp(params[ex_id, param_start_id + i])
            params[ex_id, param_start_id + i] = param
            cum_val += param
        end
        for i = 1 : num_cats
            params[ex_id, param_start_id + i] /= cum_val
        end
    end
    nothing
end

function undo_normalize_params_step(d::BitsFixableCategorical, params, ex_id, param_start_id)
    if !d.fixed
        num_cats = d.num_cats
        for i = 1 : num_cats
            param = params[ex_id, param_start_id + i]
            logp = if param > 0
                    log(param + Float32(1e-6))
                else
                    param
                end
            params[ex_id, param_start_id + i] = logp
        end
    end
    nothing
end

function normalize_params_step(d::BitsGaussian, params, ex_id, param_start_id)
    params[ex_id, param_start_id + 2] = exp(Float32(0.5) * params[ex_id, param_start_id + 2])
end

function undo_normalize_params_step(d::BitsGaussian, params, ex_id, param_start_id)
    params[ex_id, param_start_id + 2] = Float32(2.0) * log(params[ex_id, param_start_id + 2])
end

function preprocess_inner_params_kernel(edges, edge2param, param2group, params, cum_params, num_examples::Int32, 
                                        num_ex_threads::Int32, edge_work::Int32, layer_start::Int32, layer_end::Int32)
    threadid_block = threadIdx().x
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadid_block 

    edge_batch, ex_id = fldmod1(threadid, num_ex_threads)

    edge_start = layer_start + (edge_batch-one(Int32))*edge_work 
    edge_end = min(edge_start + edge_work - one(Int32), layer_end) 
    
    @inbounds if ex_id <= num_examples
        for edge_id = edge_start:edge_end

            edge = edges[edge_id]

            if edge isa SumEdge
                param_id = edge2param[edge_id]
                param_group_id = param2group[param_id]

                p = exp(params[ex_id, param_id])
                CUDA.@atomic cum_params[ex_id, param_group_id] += p
            end
        end
    end

    CUDA.sync_threads()

    @inbounds if ex_id <= num_examples
        for edge_id = edge_start:edge_end

            edge = edges[edge_id]

            if edge isa SumEdge
                param_id = edge2param[edge_id]
                param_group_id = param2group[param_id]

                logp = params[ex_id, param_id] - log(cum_params[ex_id, param_group_id])
                params[ex_id, param_id] = logp
            end
        end
    end

    nothing
end

function normalize_params(mbpc::CuMetaBitsProbCircuit, params; mine = 2, maxe = 32, undo = false, cum_params_mem = nothing)
    if params isa Matrix{Float32}
        params = cu(params)
    end

    bpc = mbpc.bpc
    num_inner_params = length(mbpc.param2edge)
    num_examples = size(params, 1)
    num_input_nodes = length(bpc.input_node_ids)

    if !undo
        process_func = normalize_params_step
    else
        process_func = undo_normalize_params_step
    end

    # input nodes
    dummy_args = (bpc.nodes, bpc.input_node_ids, mbpc.innode2ncumparam, num_inner_params, params, 
                  process_func, Int32(1), Int32(1), Int32(1))
    kernel = @cuda name="preprocess_input_params" launch=false preprocess_input_params_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, node_work = 
        balance_threads(num_input_nodes, num_examples, config; mine, maxe)

    args = (bpc.nodes, bpc.input_node_ids, mbpc.innode2ncumparam, num_inner_params, params,
            process_func, Int32(num_examples), Int32(num_example_threads), Int32(node_work))
    kernel(args...; threads, blocks)

    # inner nodes
    cum_params = prep_memory(cum_params_mem, (num_examples, mbpc.num_param_groups), (false, true))
    cum_params .= zero(Float32)

    edges = bpc.edge_layers_up.vectors
    n_edges = length(edges)
    edge2param = mbpc.edge2param
    param2group = mbpc.param2group

    dummy_args = (edges, edge2param, param2group, params, cum_params, Int32(1), Int32(1), Int32(1), Int32(1), Int32(n_edges))
    kernel = @cuda name="preprocess_inner_params" launch=false preprocess_inner_params_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, edge_work = 
        balance_threads(length(edges), num_examples, config; mine, maxe)

    args = (edges, edge2param, param2group, params, cum_params, Int32(num_examples), 
            Int32(num_example_threads), Int32(edge_work), Int32(1), Int32(n_edges))
    kernel(args...; threads, blocks)

    Array(params)
end

function unnormalize_input_params_grad_kernel(nodes, input_node_ids, innode2ncumparam, num_inner_params, 
                                              params, params_grad, num_examples, num_ex_threads, node_work)

    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = one(Int32) + (node_batch - one(Int32)) * node_work
    node_end = min(node_start + node_work - one(Int32), length(input_node_ids))

    @inbounds if ex_id <= num_examples
        for node_id = node_start : node_end
            orig_node_id::UInt32 = input_node_ids[node_id]
            inputnode = nodes[orig_node_id]::BitsInput
            d = dist(inputnode)

            nparams = num_parameters(d, false)
            param_start_id = num_inner_params + innode2ncumparam[node_id] - nparams

            unnormalize_params_grad_step(d, params, params_grad, ex_id, param_start_id)
        end
    end

    nothing
end

function unnormalize_params_grad_step(d::BitsGaussian, params, params_grad, ex_id, param_start_id)
    params_grad[ex_id, param_start_id + 2] *= Float32(0.5) * params[ex_id, param_start_id + 2]
end

function unnormalize_params_grad_step(d::BitsFixableCategorical, params, params_grad, ex_id, param_start_id)
    if !d.fixed
        num_cats = d.num_cats
        res_grad = zero(Float32)
        for i = 1 : num_cats
            res_grad += params[ex_id, param_start_id + i] * params_grad[ex_id, param_start_id + i]
        end
        for i = 1 : num_cats
            grad = params[ex_id, param_start_id + i] * (params_grad[ex_id, param_start_id + i] - res_grad)
            params_grad[ex_id, param_start_id + i] = grad
        end
    end
    nothing
end

function unnormalize_inner_params_grad_kernel(edges, edge2param, param2group, params, params_grad, cum_grads, num_examples::Int32, 
                                              num_ex_threads::Int32, edge_work::Int32, layer_start::Int32, layer_end::Int32)
    threadid_block = threadIdx().x
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadid_block 

    edge_batch, ex_id = fldmod1(threadid, num_ex_threads)

    edge_start = layer_start + (edge_batch-one(Int32))*edge_work 
    edge_end = min(edge_start + edge_work - one(Int32), layer_end) 
    
    @inbounds if ex_id <= num_examples
        for edge_id = edge_start:edge_end

            edge = edges[edge_id]

            if edge isa SumEdge
                param_id = edge2param[edge_id]
                param_group_id = param2group[param_id]

                param = exp(params[ex_id, param_id])
                grad = params_grad[ex_id, param_id]

                CUDA.@atomic cum_grads[ex_id, param_group_id] += param * grad
            end
        end
    end

    CUDA.sync_threads()

    @inbounds if ex_id <= num_examples
        for edge_id = edge_start:edge_end

            edge = edges[edge_id]

            if edge isa SumEdge
                param_id = edge2param[edge_id]
                param_group_id = param2group[param_id]

                grad = params_grad[ex_id, param_id]
                cum_grad = cum_grads[ex_id, param_group_id]

                params_grad[ex_id, param_id] = grad - cum_grad
            end
        end
    end

    nothing
end

function shift_grads_kernel(edges, edge2param, param2group, pargroup_sizes, params_grad, cum_grads, num_examples::Int32, 
                            num_ex_threads::Int32, edge_work::Int32, layer_start::Int32, layer_end::Int32)
    threadid_block = threadIdx().x
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadid_block 

    edge_batch, ex_id = fldmod1(threadid, num_ex_threads)

    edge_start = layer_start + (edge_batch-one(Int32))*edge_work 
    edge_end = min(edge_start + edge_work - one(Int32), layer_end) 
    
    @inbounds if ex_id <= num_examples
        for edge_id = edge_start:edge_end

            edge = edges[edge_id]

            if edge isa SumEdge
                param_id = edge2param[edge_id]
                param_group_id = param2group[param_id]

                grad = params_grad[ex_id, param_id]

                CUDA.@atomic cum_grads[ex_id, param_group_id] += grad
            end
        end
    end

    CUDA.sync_threads()

    @inbounds if ex_id <= num_examples
        for edge_id = edge_start:edge_end

            edge = edges[edge_id]

            if edge isa SumEdge
                param_id = edge2param[edge_id]
                param_group_id = param2group[param_id]

                cum_grad = cum_grads[ex_id, param_group_id]
                nparams = pargroup_sizes[param_group_id]

                params_grad[ex_id, param_id] -= cum_grad / nparams
            end
        end
    end

    nothing
end

function unnormalize_params_grad(mbpc::CuMetaBitsProbCircuit, params, params_grad; mine = 2, maxe = 32, shift_inner_grads = false)
    if params isa Matrix
        params = cu(params)
    end
    if params_grad isa Matrix
        params_grad = cu(params_grad)
    end

    bpc = mbpc.bpc
    num_inner_params = length(mbpc.param2edge)
    num_examples = size(params, 1)
    num_input_nodes = length(bpc.input_node_ids)

    # input nodes
    dummy_args = (bpc.nodes, bpc.input_node_ids, mbpc.innode2ncumparam, num_inner_params, params, params_grad, 
                  Int32(1), Int32(1), Int32(1))
    kernel = @cuda name="unnormalize_input_params_grad" launch=false unnormalize_input_params_grad_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, node_work = 
        balance_threads(num_input_nodes, num_examples, config; mine, maxe)

    args = (bpc.nodes, bpc.input_node_ids, mbpc.innode2ncumparam, num_inner_params, params, params_grad, 
            Int32(num_examples), Int32(num_example_threads), Int32(node_work))
    kernel(args...; threads, blocks)

    # inner nodes
    cum_grads = prep_memory(nothing, (num_examples, mbpc.num_param_groups), (false, true))
    cum_grads .= zero(Float32)

    edges = bpc.edge_layers_up.vectors
    n_edges = length(edges)
    edge2param = mbpc.edge2param
    param2group = mbpc.param2group

    dummy_args = (edges, edge2param, param2group, params, params_grad, cum_grads, Int32(1), 
                  Int32(1), Int32(1), Int32(1), Int32(n_edges))
    kernel = @cuda name="unnormalize_inner_params_grad" launch=false unnormalize_inner_params_grad_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, edge_work = 
        balance_threads(length(edges), num_examples, config; mine, maxe)

    args = (edges, edge2param, param2group, params, params_grad, cum_grads, Int32(num_examples), 
            Int32(num_example_threads), Int32(edge_work), Int32(1), Int32(n_edges))
    kernel(args...; threads, blocks)

    if shift_inner_grads
        cum_grads .= zero(Float32)

        pargroup_sizes = mbpc.pargroup_sizes

        dummy_args = (edges, edge2param, param2group, pargroup_sizes, params_grad, cum_grads, Int32(1), 
                      Int32(1), Int32(1), Int32(1), Int32(n_edges))
        kernel = @cuda name="shift_grads" launch=false shift_grads_kernel(dummy_args...)
        config = launch_configuration(kernel.fun)

        threads, blocks, num_example_threads, edge_work = 
            balance_threads(length(edges), num_examples, config; mine, maxe)

        args = (edges, edge2param, param2group, pargroup_sizes, params_grad, cum_grads, Int32(num_examples), 
                Int32(num_example_threads), Int32(edge_work), Int32(1), Int32(n_edges))
        kernel(args...; threads, blocks)
    end

    Array(params_grad)
end

function merge_params_kernel(edge2param, edge_params, all_params, num_examples::Int32, 
                             num_ex_threads::Int32, edge_work::Int32, layer_start::Int32, layer_end::Int32)
    threadid_block = threadIdx().x
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadid_block 

    edge_batch, ex_id = fldmod1(threadid, num_ex_threads)

    edge_start = layer_start + (edge_batch - one(Int32)) * edge_work 
    edge_end = min(edge_start + edge_work - one(Int32), layer_end) 
    
    @inbounds if ex_id <= num_examples
        for edge_id = edge_start:edge_end
            param_id = edge2param[edge_id]
            if param_id >= 1
                all_params[ex_id, param_id] = edge_params[ex_id, edge_id]
            end
        end
    end

    nothing
end

function merge_params(mbpc::CuMetaBitsProbCircuit, edge_params, input_params, all_params; mine = 2, maxe = 32)
    edge2param = mbpc.edge2param

    bpc = mbpc.bpc
    edges = bpc.edge_layers_up.vectors
    num_inner_params = length(mbpc.param2edge)
    num_input_params = size(input_params, 2)
    num_examples = size(input_params, 1)

    dummy_args = (edge2param, edge_params, all_params, Int32(1), Int32(1), Int32(1), Int32(1), Int32(1))
    kernel = @cuda name="merge_params" launch=false merge_params_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, edge_work = 
        balance_threads(length(edges), num_examples, config; mine, maxe)

    args = (edge2param, edge_params, all_params, Int32(num_examples), Int32(num_example_threads), Int32(edge_work),
            Int32(1), Int32(length(edges)))
    kernel(args...; threads, blocks)

    all_params[:,num_inner_params+1:end] .= input_params

    nothing
end