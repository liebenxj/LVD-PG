

function copy_parameters_to_vec_kernel(edges, params)
    edge_id = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 
    @inbounds if edge_id <= length(edges)
        edge = edges[edge_id]
        if edge isa SumEdge
            params[edge_id] = edge.logp
        else
            params[edge_id] = zero(Float32)
        end
    end
    nothing
end

function copy_parameters_to_vec(bpc, params)
    edges = bpc.edge_layers_down.vectors
    args = (edges, params)
    kernel = @cuda name="copy_parameters_to_vec" launch=false copy_parameters_to_vec_kernel(args...)
    threads = launch_configuration(kernel.fun).threads
    blocks = cld(length(edges), threads)

    kernel(args...; threads, blocks)
    nothing
end

function compute_gradients_kernel(edges, edge_aggr, node_aggr)
    edge_id = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    @inbounds if edge_id <= length(edges)
        edge = edges[edge_id]
        if edge isa SumEdge
            parent_id = edge.parent_id
            edge_flow = edge_aggr[edge_id]
            CUDA.@atomic node_aggr[parent_id] += edge_flow
        end
    end

    CUDA.sync_threads()

    @inbounds if edge_id <= length(edges)
        edge = edges[edge_id]
        if edge isa SumEdge
            parent_id = edge.parent_id
            edge_aggr[edge_id] -= exp(edge.logp) * node_aggr[parent_id]
        end
    end
    nothing
end

function compute_gradients(bpc, edge_aggr, node_aggr)
    node_aggr .= zero(Float32)
    edges = bpc.edge_layers_down.vectors
    args = (edges, edge_aggr, node_aggr)
    kernel = @cuda name="compute_gradients" launch=false compute_gradients_kernel(args...)
    threads = launch_configuration(kernel.fun).threads
    blocks = cld(length(edges), threads)

    kernel(args...; threads, blocks)
    nothing
end

function apply_gradients_kernel(edges, params, adam_m, adam_v, lr::Float32, beta1::Float32, beta2::Float32, eps::Float32, opt_count::Int32)
    edge_id = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 
    @inbounds if edge_id <= length(params)
        edge = edges[edge_id]
        if edge isa SumEdge
            m_hat = adam_m[edge_id] / (one(Float32) - beta1^opt_count)
            v_hat = adam_v[edge_id] / (one(Float32) - beta2^opt_count)
            g = m_hat / (sqrt(v_hat) + eps)
            params[edge_id] += lr * g
        end
    end
    nothing
end

function apply_gradients(bpc, params, adam_m, adam_v, lr, beta1, beta2, eps, opt_count)
    edges = bpc.edge_layers_down.vectors
    args = (edges, params, adam_m, adam_v, Float32(lr), Float32(beta1), Float32(beta2), Float32(eps), Int32(opt_count))
    kernel = @cuda name="apply_gradients" launch=false apply_gradients_kernel(args...)
    threads = launch_configuration(kernel.fun).threads
    blocks = cld(length(edges), threads)

    kernel(args...; threads, blocks)
    nothing
end

function hard_update_params_kernel(edges_down, edges_up, _down2upedge, params, node_aggr)
    down2upedge = Base.Experimental.Const(_down2upedge)

    edge_id_down = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    @inbounds if edge_id_down <= length(edges_down)
        edge_down = edges_down[edge_id_down]
        if edge_down isa SumEdge
            parent_id = edge_down.parent_id
            p = exp(params[edge_id_down])
            CUDA.@atomic node_aggr[parent_id] += p
        end
    end

    CUDA.sync_threads()

    @inbounds if edge_id_down <= length(edges_down)
        edge_down = edges_down[edge_id_down]
        if edge_down isa SumEdge

            edge_id_up = down2upedge[edge_id_down]
            # only difference is the tag
            edge_up_tag = edges_up[edge_id_up].tag

            if !(isfirst(edge_up_tag) && islast(edge_up_tag))
                parent_id = edge_down.parent_id
                logp = params[edge_id_down] - log(node_aggr[parent_id])

                edges_down[edge_id_down] = 
                    SumEdge(parent_id, edge_down.prime_id, edge_down.sub_id, 
                            logp, edge_down.tag)

                edges_up[edge_id_up] = 
                    SumEdge(parent_id, edge_down.prime_id, edge_down.sub_id, 
                            logp, edge_up_tag)
            end
        end
    end
    nothing
end

function hard_update_params(bpc, params, node_aggr)
    node_aggr .= zero(Float32)
    edges_down = bpc.edge_layers_down.vectors
    edges_up = bpc.edge_layers_up.vectors
    down2upedge = bpc.down2upedge
    @assert length(edges_down) == length(down2upedge) == length(edges_up)

    args = (edges_down, edges_up, down2upedge, params, node_aggr)
    kernel = @cuda name="hard_update_params" launch=false hard_update_params_kernel(args...)
    threads = launch_configuration(kernel.fun).threads
    blocks = cld(length(edges_down), threads)

    kernel(args...; threads, blocks)
    nothing
end

function get_inputs_aux_heap(bpc)
    input_node_ids = Array(bpc.input_node_ids)
    input2heapid = zeros(Int32, length(input_node_ids))
    nodes = Array(bpc.nodes)
    heap_start = one(Int32)
    for idx = 1 : length(input_node_ids)
        node_id = input_node_ids[idx]
        inputnode = nodes[node_id]::BitsInput
        d = dist(inputnode)
        input2heapid[idx] = heap_start
        if d isa BitsCategorical
            heap_start += d.num_cats * 3 + 1
        else
            error("Not implemented")
        end
    end
    input2heapid, heap_start - 1
end

function copy_input_parameters_to_heap_kernel(nodes, input_node_ids, heap, aux_heap, input2heapid)
    node_id = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x
    @inbounds if node_id <= length(input_node_ids)
        orig_node_id::UInt32 = input_node_ids[node_id]
        heap_id = input2heapid[node_id]
        inputnode = nodes[orig_node_id]::BitsInput
        copy_params_to_heap(dist(inputnode), heap, aux_heap, heap_id)
    end
end

function copy_input_parameters_to_heap(bpc, aux_heap, input2heapid)
    num_input_nodes = length(bpc.input_node_ids)

    args = (bpc.nodes, bpc.input_node_ids, bpc.heap, aux_heap, input2heapid)
    kernel = @cuda name="copy_input_parameters_to_heap" launch=false copy_input_parameters_to_heap_kernel(args...)
    threads = launch_configuration(kernel.fun).threads
    blocks = cld(num_input_nodes, threads)

    kernel(args...; threads, blocks)
    nothing
end

copy_params_to_heap(d::BitsCategorical, heap, aux_heap, heap_id) = begin
    for i = 0 : d.num_cats-1
        aux_heap[heap_id+i] = heap[d.heap_start+i]
    end
    nothing
end

function update_input_node_params_adam_kernel(nodes, input_node_ids, heap, aux_heap, input2heapid, 
                                              lr::Float32, beta1::Float32, beta2::Float32, eps::Float32, opt_count::Int32)
    node_id = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x
    @inbounds if node_id <= length(input_node_ids)
        orig_node_id::UInt32 = input_node_ids[node_id]
        heap_id = input2heapid[node_id]
        inputnode = nodes[orig_node_id]::BitsInput
        update_params_adam(dist(inputnode), heap, aux_heap, heap_id, lr, beta1, beta2, eps, opt_count)
    end
    nothing
end

function update_input_node_params_adam(bpc, aux_heap, input2heapid, lr, beta1, beta2, eps, opt_count)
    num_input_nodes = length(bpc.input_node_ids)

    args = (bpc.nodes, bpc.input_node_ids, bpc.heap, aux_heap, input2heapid, Float32(lr), Float32(beta1), 
            Float32(beta2), Float32(eps), Int32(opt_count))
    kernel = @cuda name="update_input_node_params_adam" launch=false update_input_node_params_adam_kernel(args...)
    threads = launch_configuration(kernel.fun).threads
    blocks = cld(num_input_nodes, threads)

    kernel(args...; threads, blocks)
    nothing
end

update_params_adam(d::BitsCategorical, heap, aux_heap, heap_id, lr, beta1, beta2, eps, opt_count) = begin
    num_cats = d.num_cats

    @inbounds begin
        node_flow = zero(Float32)
        for i = 0 : num_cats-1
            node_flow += heap[d.heap_start+num_cats+i]
        end

        for i = 0 : num_cats-1
            p = exp(heap[d.heap_start+i])
            g = heap[d.heap_start+num_cats+i] + p * node_flow

            adam_m = beta1 * aux_heap[heap_id+num_cats+i] + (one(Float32) - beta1) * g
            adam_v = beta2 .* aux_heap[heap_id+UInt(2)*num_cats+i] + (one(Float32) - beta2) * g^2
            aux_heap[heap_id+num_cats+i] = adam_m
            aux_heap[heap_id+UInt(2)*num_cats+i] = adam_v

            m_hat = adam_m / (one(Float32) - beta1^opt_count)
            v_hat = adam_v / (one(Float32) - beta2^opt_count)
            g = m_hat / (sqrt(v_hat) + eps)
            aux_heap[heap_id+i] += lr * g
        end
    end
end

function adam(bpc::CuBitsProbCircuit, data::CuArray, num_epochs; batch_size, lr, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, val_data = nothing,
              mars_mem = nothing, flows_mem = nothing, edge_aggr_mem = nothing, params_mem = nothing, adam_m_mem = nothing,
              adam_v_mem = nothing, node_aggr_mem = nothing, aux_heap_mem = nothing, mine = 2, maxe = 32, verbose = true)
    num_examples = size(data, 1)
    num_nodes = length(bpc.nodes)
    num_edges = length(bpc.edge_layers_down.vectors)
    num_batches = num_examples ÷ batch_size # drop last incomplete batch

    lr = Float32(lr)
    beta1 = Float32(beta1)
    beta2 = Float32(beta2)
    eps = Float32(eps)

    @assert batch_size <= num_examples

    mars = prep_memory(mars_mem, (batch_size, num_nodes), (false, true))
    flows = prep_memory(flows_mem, (batch_size, num_nodes), (false, true))
    edge_aggr = prep_memory(edge_aggr_mem, (num_edges,))
    node_aggr = prep_memory(node_aggr_mem, (num_nodes,))
    params = prep_memory(params_mem, (num_edges,))
    adam_m = prep_memory(adam_m_mem, (num_edges,))
    adam_v = prep_memory(adam_v_mem, (num_edges,))

    @views edge_aggr .= zero(Float32)
    PCs.clear_input_node_mem(bpc; rate = 0)
    @views adam_m .= zero(Float32)
    @views adam_v .= zero(Float32)

    shuffled_indices_cpu = Vector{Int32}(undef, num_examples)
    shuffled_indices = CuVector{Int32}(undef, num_examples)
    batches = [@view shuffled_indices[1+(b-1)*batch_size : b*batch_size]
                for b in 1:num_batches]

    do_shuffle() = begin
        randperm!(shuffled_indices_cpu)
        copyto!(shuffled_indices, shuffled_indices_cpu)
    end

    log_likelihoods = Vector{Float32}()
    log_likelihoods_epoch = CUDA.zeros(Float32, num_batches, 1)

    input2heapid, n_heapparams = get_inputs_aux_heap(bpc) 
    input2heapid = cu(input2heapid)
    aux_heap = prep_memory(aux_heap_mem, (n_heapparams,)) # stores params, adam_m, adam_v, etc.
    aux_heap .= zero(Float32)

    copy_parameters_to_vec(bpc, params)
    copy_input_parameters_to_heap(bpc, aux_heap, input2heapid)
    
    opt_count = zero(Int32)
    for epoch = 1 : num_epochs

        log_likelihoods_epoch .= zero(Float32)

        do_shuffle()

        for (batch_id, batch) in enumerate(batches)

            opt_count += one(Int32)

            edge_aggr .= zero(Float32)
            PCs.clear_input_node_mem(bpc; rate = 0)

            PCs.probs_flows_circuit(flows, mars, edge_aggr, bpc, data, batch; mine, maxe)
            
            @views sum!(log_likelihoods_epoch[batch_id:batch_id, 1:1],
                        mars[1:batch_size,end:end])
            
            @views edge_aggr ./= batch_size
            compute_gradients(bpc, edge_aggr, node_aggr)
            @views adam_m .= beta1 .* adam_m .+ (one(Float32) - beta1) .* edge_aggr
            @views adam_v .= beta2 .* adam_v .+ (one(Float32) - beta2) .* edge_aggr.^2
            apply_gradients(bpc, params, adam_m, adam_v, lr, beta1, beta2, eps, opt_count)
            if rand() < 0.01
                println("params ", minimum(params), " ", maximum(params))
            end
            
            hard_update_params(bpc, params, node_aggr)

            update_input_node_params_adam(bpc, aux_heap, input2heapid, lr, beta1, beta2, eps, opt_count)

        end

        log_likelihood = sum(log_likelihoods_epoch) / batch_size / num_batches
        push!(log_likelihoods, log_likelihood)

        if verbose
            println("Adam epoch $epoch; train LL $log_likelihood")
            if val_data !== nothing && epoch % 20 == 0
                val_ll = loglikelihood(bpc, val_data; batch_size)
                println("[Validation] LL $(val_ll)")
            end
        end

    end

    PCs.cleanup_memory((flows, flows_mem), (node_aggr, node_aggr_mem), (edge_aggr, edge_aggr_mem))
    CUDA.unsafe_free!(shuffled_indices)

    log_likelihoods
end