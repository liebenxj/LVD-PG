

function extract_lls_from_root_nodes(mars, mhbpc, head_mask, example_ids, lls)
    root_ids = Array(mhbpc.root_ids)
    lls .= typemin(Float32)
    num_examples = length(example_ids)
    if head_mask !== nothing
        for i = 1 : length(root_ids)
            lls[1:num_examples] .= PCs.logsumexp.(lls[1:num_examples], mars[1:num_examples,root_ids[i]] .+ log.(head_mask[example_ids,i]))
        end
        lls[1:num_examples] .-= log.(sum(head_mask[example_ids,:]; dims = 2))
    else
        for i = 1 : length(root_ids)
            lls[1:num_examples,i] .= mars[1:num_examples,root_ids[i]]
        end
    end
    lls
end

function eval_multi_head_pc(mars, mhbpc, data, example_ids; mine, maxe, soft_reg, soft_reg_width, debug = false)
    sum_agg_func(x::Float32, y::Float32) = 
        PCs.logsumexp(x, y)

    bpc = mhbpc.bpc

    init_mar!_with_reg(mars, bpc, data, example_ids; mine, maxe, soft_reg, soft_reg_width, debug)
    layer_start = 1
    for layer_end in bpc.edge_layers_up.ends
        PCs.layer_up(mars, bpc, layer_start, layer_end, length(example_ids); mine, maxe, sum_agg_func, debug)
        layer_start = layer_end + 1
    end
    nothing
end

import ProbabilisticCircuits: loglikelihoods # extend

function loglikelihoods(mhbpc::CuMultiHeadBitsProbCircuit, data::CuArray, head_mask::Union{CuMatrix,Nothing}; batch_size, mars_mem = nothing, 
                        mine = 2, maxe = 32)
    bpc = mhbpc.bpc
    num_examples = size(data, 1)
    num_nodes = length(bpc.nodes)

    mars = prep_memory(mars_mem, (batch_size, num_nodes), (false, true))
    if head_mask === nothing
        lls = prep_memory(nothing, (batch_size, length(mhbpc.root_ids)), (false, true))
        log_likelihoods = CUDA.zeros(Float32, num_examples, length(mhbpc.root_ids))
    else
        lls = prep_memory(nothing, (batch_size,), (false,))
        log_likelihoods = CUDA.zeros(Float32, num_examples)
    end    

    for batch_start = 1 : batch_size : num_examples

        batch_end = min(batch_start + batch_size - 1, num_examples)
        batch = batch_start : batch_end
        num_batch_examples = length(batch)

        eval_multi_head_pc(mars, mhbpc, data, batch; mine, maxe, soft_reg = 0.0, soft_reg_width = 1)
        lls = extract_lls_from_root_nodes(mars, mhbpc, head_mask, batch, lls)

        if head_mask === nothing
            log_likelihoods[batch_start:batch_end,:] .= lls[1:num_batch_examples,:]
        else
            log_likelihoods[batch_start:batch_end] .= lls[1:num_batch_examples]
        end
    end

    PCs.cleanup_memory(mars, mars_mem)

    log_likelihoods
end

import ProbabilisticCircuits: loglikelihood # extend

function loglikelihood(mhbpc::CuMultiHeadBitsProbCircuit, data::CuArray, head_mask::Union{CuMatrix,Nothing}; batch_size, mars_mem = nothing, 
                       mine = 2, maxe = 32)
    lls = loglikelihoods(mhbpc, data, head_mask; batch_size, mars_mem, mine, maxe)

    if head_mask === nothing
        reshape(sum(lls; dims = 1), (:,)) / size(lls, 1)
    else
        sum(lls) / length(lls)
    end
end