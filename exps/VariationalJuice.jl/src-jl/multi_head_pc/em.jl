function mini_batch_em_for_multihead_pc(mhbpc::CuMultiHeadBitsProbCircuit, data::CuArray, head_mask::CuArray, num_epochs; 
                                        batch_size, pseudocount, soft_reg, soft_reg_width, 
                                        param_inertia, param_inertia_end = param_inertia, flow_memory = zero(Float32), shuffle = :each_epoch,
                                        mars_mem = nothing, flows_mem = nothing, node_aggr_mem = nothing, edge_aggr_mem = nothing,
                                        mine = 2, maxe = 32, debug = false, verbose = true, weights = nothing, lls_mem = nothing,
                                        eval_dataset = nothing, eval_interval = 0, init_clear_mem = true)

    if verbose
        println("Preparing for EM...")
    end

    bpc = mhbpc.bpc

    num_examples = size(data, 1)
    num_nodes = length(bpc.nodes)
    num_edges = length(bpc.edge_layers_down.vectors)
    num_batches = num_examples ÷ batch_size # drop last incomplete batch

    @assert batch_size <= num_examples
    @assert isnothing(weights) || length(weights) == num_examples

    mars = prep_memory(mars_mem, (batch_size, num_nodes), (false, true))
    flows = prep_memory(flows_mem, (batch_size, num_nodes), (false, true))
    node_aggr = prep_memory(node_aggr_mem, (num_nodes,))
    edge_aggr = prep_memory(edge_aggr_mem, (num_edges,))
    lls = prep_memory(lls_mem, (batch_size,), (false,))

    if edge_aggr_mem === nothing || init_clear_mem
        edge_aggr .= zero(Float32)
        PCs.clear_input_node_mem(bpc; rate = 0, debug)
    end

    shuffled_indices_cpu = Vector{Int32}(undef, num_examples)
    shuffled_indices = CuVector{Int32}(undef, num_examples)
    batches = [@view shuffled_indices[1+(b-1)*batch_size : b*batch_size]
                for b in 1:num_batches]

    do_shuffle() = begin
        randperm!(shuffled_indices_cpu)
        copyto!(shuffled_indices, shuffled_indices_cpu)
    end

    (shuffle == :once) && do_shuffle()

    Δparam_inertia = (param_inertia_end-param_inertia)/num_epochs

    log_likelihoods = Vector{Float32}()
    log_likelihoods_epoch = CUDA.zeros(Float32, num_batches, 1)

    for epoch = 1 : num_epochs

        log_likelihoods_epoch .= zero(Float32)
        CUDA.synchronize()
        
        (shuffle == :each_epoch) && do_shuffle()
        CUDA.synchronize()
        
        for (batch_id, batch) in enumerate(batches)
            (shuffle == :each_batch) && do_shuffle()

            if iszero(flow_memory)
                edge_aggr .= zero(Float32)
                PCs.clear_input_node_mem(bpc; rate = 0, debug)
            else
                # slowly forget old edge aggregates
                rate = max(zero(Float32), one(Float32) - (batch_size + pseudocount) / flow_memory)
                edge_aggr .*= rate
                PCs.clear_input_node_mem(bpc; rate)
            end
            
            multi_head_probs_flows_circuit(flows, mars, edge_aggr, mhbpc, data, head_mask, batch; 
                                        mine, maxe, soft_reg, soft_reg_width, debug, weights)
            extract_lls_from_root_nodes(mars, mhbpc, head_mask, batch, lls)
            @views sum!(log_likelihoods_epoch[batch_id:batch_id, 1], lls[1:batch_size])
            
            # to modify
            PCs.add_pseudocount(edge_aggr, node_aggr, bpc, pseudocount; debug)
            PCs.aggr_node_flows(node_aggr, bpc, edge_aggr; debug)
            PCs.update_params(bpc, node_aggr, edge_aggr; inertia = param_inertia, debug)
            
            PCs.update_input_node_params(bpc; pseudocount, inertia = param_inertia, debug)

        end
        
        log_likelihood = sum(log_likelihoods_epoch) / batch_size / num_batches
        push!(log_likelihoods, log_likelihood)

        param_inertia += Δparam_inertia

        if verbose
            overprint("Mini-batch EM epoch $epoch; train LL $log_likelihood")
        end
    end

    PCs.cleanup_memory((flows, flows_mem), (node_aggr, node_aggr_mem), (edge_aggr, edge_aggr_mem))
    CUDA.unsafe_free!(shuffled_indices)

    log_likelihoods
end