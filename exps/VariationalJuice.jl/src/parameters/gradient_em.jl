using ProbabilisticCircuits: clear_input_node_mem, isfirst, islast


#########################
# Minibatch gradient EM
#########################

"Update parameters of the CuBitsProbCircuit by (minibatch) gradient EM"
function mini_batch_gradient_em(bpc::CuBitsProbCircuit, data::CuArray, num_epochs::Integer;
                                batch_size::Integer, lr::AbstractFloat, shuffle = :each_epoch,
                                mars_mem = nothing, flows_mem = nothing, node_aggr_mem = nothing, edge_aggr_mem = nothing,
                                mine = 2, maxe = 32, verbose = true, log_mode = "overprint",
                                eval_data::Union{CuArray,Nothing} = nothing, eval_interval = 0,
                                update_mode::Symbol = :multiplicative)

    @assert 0 < lr < 1
    @assert shuffle ∈ [:once, :each_epoch, :each_batch]
    @assert update_mode ∈ [:multiplicative, :additive, :scaled_multiplicative, :scaled_additive]

    num_examples = size(data, 1)
    num_nodes = length(bpc.nodes)
    num_edges = length(bpc.edge_layers_down.vectors)
    num_batches = num_examples ÷ batch_size # drop last incomplete batch

    @assert batch_size <= num_examples

    if verbose && log_mode == "overprint"
        println("Preparing to run mini-batch gradient EM...")
    end

    mars = prep_memory(mars_mem, (batch_size, num_nodes), (false, true))
    flows = prep_memory(flows_mem, (batch_size, num_nodes), (false, true))
    node_aggr = prep_memory(node_aggr_mem, (num_nodes,))
    edge_aggr = prep_memory(edge_aggr_mem, (num_edges,))

    shuffled_indices_cpu = Vector{Int32}(undef, num_examples)
    shuffled_indices = CuVector{Int32}(undef, num_examples)
    batches = [@view shuffled_indices[1+(b-1)*batch_size : b*batch_size]
                for b in 1:num_batches]

    do_shuffle() = begin
        randperm!(shuffled_indices_cpu)
        copyto!(shuffled_indices, shuffled_indices_cpu)
    end

    (shuffle == :once) && do_shuffle()

    log_likelihoods = Vector{Float32}()
    log_likelihoods_epoch = CUDA.zeros(Float32, num_batches, 1)

    last_test_ll = nothing

    for epoch = 1 : num_epochs

        @inbounds @views log_likelihoods_epoch .= zero(Float32)

        (shuffle == :each_epoch) && do_shuffle()

        for (batch_id, batch) in enumerate(batches)

            (shuffle == :each_batch) && do_shuffle()

            # Clear aggregated statistics
            @inbounds @views edge_aggr .= zero(Float32)
            clear_input_node_mem(bpc; rate = 0)

            PCs.probs_flows_circuit(flows, mars, edge_aggr, bpc, data, batch; mine, maxe)

            @views sum!(log_likelihoods_epoch[batch_id:batch_id, 1:1],
                        mars[1:batch_size, end:end])

            compute_grad_em_update_target(edge_aggr, node_aggr, bpc, lr, batch_size, update_mode)
            update_log_parameters(bpc, edge_aggr, node_aggr, update_mode, lr, batch_size)
            grad_em_update_input_node_params(bpc, lr, batch_size, update_mode)
        end

        log_likelihood = sum(log_likelihoods_epoch) / batch_size / num_batches
        push!(log_likelihoods, log_likelihood)

        # logging
        if verbose
            if eval_data !== nothing && eval_interval > 0 && epoch % eval_interval == 0
                test_ll = loglikelihood(bpc, eval_data; batch_size)
                if log_mode == "overprint"
                    overprint("Mini-batch gradient EM epoch $epoch/$num_epochs: train LL $log_likelihood - test LL $test_ll")
                else
                    println("Mini-batch gradient EM epoch $epoch/$num_epochs: train LL $log_likelihood - test LL $test_ll")
                end
                last_test_ll = test_ll
            else
                if log_mode == "overprint"
                    if last_test_ll !== nothing
                        overprint("Mini-batch gradient EM epoch $epoch/$num_epochs; train LL $log_likelihood - test LL $last_test_ll")
                    else
                        overprint("Mini-batch gradient EM epoch $epoch/$num_epochs; train LL $log_likelihood")
                    end
                else
                    println("Mini-batch gradient EM epoch $epoch/$num_epochs; train LL $log_likelihood")
                end
            end
        end

    end

    PCs.cleanup_memory((mars, mars_mem), (flows, flows_mem), (node_aggr, node_aggr_mem), 
                       (edge_aggr, edge_aggr_mem))

    log_likelihoods
end

function grad_em_unnorm_target_mul_kernel(edge_aggr, edges, node_aggr, lr, batch_size)
    edge_id = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x
    @inbounds if edge_id <= length(edges)
        edge = edges[edge_id]
        if edge isa SumEdge
            parent_id = edge.parent_id
            edge_aggr[edge_id] = edge.logp + lr * edge_aggr[edge_id] / batch_size
            CUDA.@atomic node_aggr[parent_id] = logsumexp(node_aggr[parent_id], edge_aggr[edge_id])
        end
    end
    nothing
end

function grad_em_unnorm_target_sc_mul_kernel(edge_aggr, edges, node_aggr, lr, batch_size)
    edge_id = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x
    @inbounds if edge_id <= length(edges)
        edge = edges[edge_id]
        if edge isa SumEdge
            parent_id = edge.parent_id
            CUDA.@atomic node_aggr[parent_id] += edge_aggr[edge_id]

            CUDA.sync_threads()

            edge_aggr[edge_id] /= node_aggr[parent_id]
            node_aggr[parent_id] = typemin(Float32)

            CUDA.sync_threads()

            edge_aggr[edge_id] = edge.logp + lr * edge_aggr[edge_id] / batch_size
            CUDA.@atomic node_aggr[parent_id] = logsumexp(node_aggr[parent_id], edge_aggr[edge_id])
        end
    end
    nothing
end

function grad_em_unnorm_target_add_kernel(edge_aggr, edges, node_aggr, lr, batch_size)
    edge_id = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x
    @inbounds if edge_id <= length(edges)
        edge = edges[edge_id]
        if edge isa SumEdge
            parent_id = edge.parent_id
            update_target = edge_aggr[edge_id] / (exp(edge.logp) + Float32(1e-8))
            edge_aggr[edge_id] = exp(edge.logp) + lr * update_target / batch_size
            CUDA.@atomic node_aggr[parent_id] += edge_aggr[edge_id]
        end
    end
    nothing
end

function grad_em_unnorm_target_sc_add_kernel(edge_aggr, edges, node_aggr, lr, batch_size)
    edge_id = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x
    @inbounds if edge_id <= length(edges)
        edge = edges[edge_id]
        if edge isa SumEdge
            parent_id = edge.parent_id
            CUDA.@atomic node_aggr[parent_id] += edge_aggr[edge_id]
        end
    end
    nothing
end

function compute_grad_em_update_target(edge_aggr, node_aggr, bpc, lr, batch_size, update_mode)
    edges = bpc.edge_layers_down.vectors
    args = (edge_aggr, edges, node_aggr, Float32(lr), Float32(batch_size))

    if update_mode == :multiplicative
        @inbounds @views node_aggr .= typemin(Float32)
        kernel = @cuda name="grad_em_unnorm_target_mul" launch=false grad_em_unnorm_target_mul_kernel(args...)
    elseif update_mode == :additive
        @inbounds @views node_aggr .= zero(Float32)
        kernel = @cuda name="grad_em_unnorm_target_add" launch=false grad_em_unnorm_target_add_kernel(args...)
    elseif update_mode == :scaled_multiplicative
        @inbounds @views node_aggr .= typemin(Float32)
        kernel = @cuda name="grad_em_unnorm_target_sc_mul" launch=false grad_em_unnorm_target_sc_mul_kernel(args...)
    elseif update_mode == :scaled_additive
        @inbounds @views node_aggr .= zero(Float32)
        kernel = @cuda name="grad_em_unnorm_target_sc_add" launch=false grad_em_unnorm_target_sc_add_kernel(args...)
    else
        error("Unknown update mode `$(update_mode)`")
    end

    threads = launch_configuration(kernel.fun).threads
    blocks = cld(length(edges), threads)
    kernel(args...; threads, blocks)

    nothing
end

function update_log_parameters_mul_kernel(edges_down, edges_up, _down2upedge, _node_aggr, _edge_aggr, lr, batch_size)
    node_aggr = Base.Experimental.Const(_node_aggr)
    edge_aggr = Base.Experimental.Const(_edge_aggr)
    down2upedge = Base.Experimental.Const(_down2upedge)

    edge_id_down = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    @inbounds if edge_id_down <= length(edges_down)
        edge_down = edges_down[edge_id_down]
        if edge_down isa SumEdge

            edge_id_up = down2upedge[edge_id_down]
            # only difference is the tag
            edge_up_tag = edges_up[edge_id_up].tag

            if !(isfirst(edge_up_tag) && islast(edge_up_tag))
                parent_id = edge_down.parent_id
                new_logp = edge_aggr[edge_id_down] - node_aggr[parent_id]

                edges_down[edge_id_down] = 
                    SumEdge(parent_id, edge_down.prime_id, edge_down.sub_id,
                            new_logp, edge_down.tag)

                edges_up[edge_id_up] = 
                    SumEdge(parent_id, edge_down.prime_id, edge_down.sub_id,
                            new_logp, edge_up_tag)
            end
        end
    end
    nothing
end

function update_log_parameters_add_kernel(edges_down, edges_up, _down2upedge, _node_aggr, _edge_aggr, lr, batch_size)
    node_aggr = Base.Experimental.Const(_node_aggr)
    edge_aggr = Base.Experimental.Const(_edge_aggr)
    down2upedge = Base.Experimental.Const(_down2upedge)

    edge_id_down = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    @inbounds if edge_id_down <= length(edges_down)
        edge_down = edges_down[edge_id_down]
        if edge_down isa SumEdge

            edge_id_up = down2upedge[edge_id_down]
            # only difference is the tag
            edge_up_tag = edges_up[edge_id_up].tag

            if !(isfirst(edge_up_tag) && islast(edge_up_tag))
                parent_id = edge_down.parent_id
                new_logp = log(edge_aggr[edge_id_down] / node_aggr[parent_id])

                edges_down[edge_id_down] = 
                    SumEdge(parent_id, edge_down.prime_id, edge_down.sub_id,
                            new_logp, edge_down.tag)

                edges_up[edge_id_up] = 
                    SumEdge(parent_id, edge_down.prime_id, edge_down.sub_id,
                            new_logp, edge_up_tag)
            end
        end
    end
    nothing
end

function update_log_parameters_sc_add_kernel(edges_down, edges_up, _down2upedge, _node_aggr, _edge_aggr, lr, batch_size)
    node_aggr = Base.Experimental.Const(_node_aggr)
    edge_aggr = Base.Experimental.Const(_edge_aggr)
    down2upedge = Base.Experimental.Const(_down2upedge)

    edge_id_down = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    @inbounds if edge_id_down <= length(edges_down)
        edge_down = edges_down[edge_id_down]
        if edge_down isa SumEdge

            edge_id_up = down2upedge[edge_id_down]
            # only difference is the tag
            edge_up_tag = edges_up[edge_id_up].tag

            if !(isfirst(edge_up_tag) && islast(edge_up_tag))
                parent_id = edge_down.parent_id
                inertia = one(Float32) - lr * node_aggr[parent_id] / batch_size
                old = inertia * exp(edge_down.logp)
                new = (one(Float32) - inertia) * edge_aggr[edge_id_down] / node_aggr[parent_id]
                new_logp = log(old + new)

                edges_down[edge_id_down] = 
                    SumEdge(parent_id, edge_down.prime_id, edge_down.sub_id,
                            new_logp, edge_down.tag)

                edges_up[edge_id_up] = 
                    SumEdge(parent_id, edge_down.prime_id, edge_down.sub_id,
                            new_logp, edge_up_tag)
            end
        end
    end
    nothing
end

function update_log_parameters(bpc, edge_aggr, node_aggr, update_mode, lr, batch_size)
    edges_down = bpc.edge_layers_down.vectors
    edges_up = bpc.edge_layers_up.vectors
    down2upedge = bpc.down2upedge
    @assert length(edges_down) == length(down2upedge) == length(edges_up)

    args = (edges_down, edges_up, down2upedge, node_aggr, edge_aggr, lr, batch_size)

    if update_mode == :multiplicative || update_mode == :scaled_multiplicative
        kernel = @cuda name="update_log_parameters_mul" launch=false update_log_parameters_mul_kernel(args...)
    elseif update_mode == :additive
        kernel = @cuda name="update_log_parameters_add" launch=false update_log_parameters_add_kernel(args...)
    elseif update_mode == :scaled_additive
        kernel = @cuda name="update_log_parameters_sc_add" launch=false update_log_parameters_sc_add_kernel(args...)
    else
        error("Unknown update mode `$(update_mode)`")
    end

    threads = launch_configuration(kernel.fun).threads
    blocks = cld(length(edges_down), threads)
    kernel(args...; threads, blocks)

    nothing
end

function grad_em_update_input_node_params_mul_kernel(nodes, input_node_ids, heap, lr, batch)

    node_id = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    @inbounds if node_id <= length(input_node_ids)
        orig_node_id::UInt32 = input_node_ids[node_id]
        inputnode = nodes[orig_node_id]::BitsInput
        grad_em_update_params_mul(dist(inputnode), heap, lr, batch)
    end
    nothing
end

function grad_em_update_input_node_params_sc_mul_kernel(nodes, input_node_ids, heap, lr, batch, pseudocount)

    node_id = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    @inbounds if node_id <= length(input_node_ids)
        orig_node_id::UInt32 = input_node_ids[node_id]
        inputnode = nodes[orig_node_id]::BitsInput
        grad_em_update_params_sc_mul(dist(inputnode), heap, pseudocount, lr, batch)
    end
    nothing
end

function grad_em_update_input_node_params_add_kernel(nodes, input_node_ids, heap, lr, batch)

    node_id = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    @inbounds if node_id <= length(input_node_ids)
        orig_node_id::UInt32 = input_node_ids[node_id]
        inputnode = nodes[orig_node_id]::BitsInput
        grad_em_update_params_add(dist(inputnode), heap, lr, batch)
    end
    nothing
end

function grad_em_update_input_node_params_sc_add_kernel(nodes, input_node_ids, heap, lr, batch, pseudocount)

    node_id = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    @inbounds if node_id <= length(input_node_ids)
        orig_node_id::UInt32 = input_node_ids[node_id]
        inputnode = nodes[orig_node_id]::BitsInput
        grad_em_update_params_sc_add(dist(inputnode), heap, pseudocount, lr, batch)
    end
    nothing
end

function grad_em_update_input_node_params(bpc, lr, batch, update_mode; pseudocount = 0.1)
    num_input_nodes = length(bpc.input_node_ids)

    if update_mode == :multiplicative
        args = (bpc.nodes, bpc.input_node_ids, bpc.heap, Float32(lr), Float32(batch))
        kernel = @cuda name="grad_em_update_input_node_params_mul" launch=false grad_em_update_input_node_params_mul_kernel(args...)
    elseif update_mode == :scaled_multiplicative
        args = (bpc.nodes, bpc.input_node_ids, bpc.heap, Float32(lr), Float32(batch), Float32(pseudocount))
        kernel = @cuda name="grad_em_update_input_node_params_sc_mul" launch=false grad_em_update_input_node_params_sc_mul_kernel(args...)
    elseif update_mode == :additive
        args = (bpc.nodes, bpc.input_node_ids, bpc.heap, Float32(lr), Float32(batch))
        kernel = @cuda name="grad_em_update_input_node_params_add" launch=false grad_em_update_input_node_params_add_kernel(args...)
    elseif update_mode == :scaled_additive
        args = (bpc.nodes, bpc.input_node_ids, bpc.heap, Float32(lr), Float32(batch), Float32(pseudocount))
        kernel = @cuda name="grad_em_update_input_node_params_sc_add" launch=false grad_em_update_input_node_params_sc_add_kernel(args...)
    else
        error("Unknown update mode `$(update_mode)`")
    end

    threads = launch_configuration(kernel.fun).threads
    blocks = cld(num_input_nodes, threads)
    kernel(args...; threads, blocks)

    nothing
end