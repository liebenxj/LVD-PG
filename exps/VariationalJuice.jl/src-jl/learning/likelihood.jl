using StatsFuns
using ProbabilisticCircuits: BitsCategorical, BitsSum, BitsMul, BitsInput, SumEdge, MulEdge
using ProbabilisticCircuits: isfirst, islast, ispartial, isonlysubedge


soft_loglikelihood(d::BitsCategorical, value, heap, soft_reg::Float32, soft_reg_width::Int32) = begin
    logp = heap[d.heap_start + UInt32(value)]
    c_start = max(0, value - soft_reg_width ÷ 2)
    c_end = min(c_start + soft_reg_width - 1, d.num_cats - 1)
    sp = zero(Float32)
    for cat_idx = c_start : c_end
        sp += exp(heap[d.heap_start + cat_idx])
    end
    sp /= (c_end - c_start + 1)
    PCs.logsumexp(logp + log(one(Float32) - soft_reg), log(sp) + log(soft_reg))
end

function init_mar!_with_reg_kernel(mars, nodes, data, example_ids, heap, num_ex_threads::Int32, 
                                   node_work::Int32, soft_reg::Float32, soft_reg_width::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = one(Int32) + (node_batch - one(Int32)) * node_work 
    node_end = min(node_start + node_work - one(Int32), length(nodes))

    @inbounds if ex_id <= length(example_ids)
        for node_id = node_start:node_end

            node = nodes[node_id]
            
            mars[ex_id, node_id] = 
                if (node isa BitsSum)
                    -Inf32
                elseif (node isa BitsMul)
                    zero(Float32)
                else
                    orig_ex_id::Int32 = example_ids[ex_id]
                    inputnode = node::BitsInput
                    variable = inputnode.variable
                    value = data[orig_ex_id, variable]
                    if ismissing(value)
                        zero(Float32)
                    else
                        soft_loglikelihood(dist(inputnode), value, heap, soft_reg, soft_reg_width)
                    end
                end
        end
    end
    nothing
end

function init_mar!_with_reg_cat_kernel(mars, nodes, data, example_ids, heap, num_ex_threads::Int32, 
                                       node_work::Int32, soft_reg::Float32, soft_reg_width::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = one(Int32) + (node_batch - one(Int32)) * node_work 
    node_end = min(node_start + node_work - one(Int32), length(nodes))

    @inbounds if ex_id <= length(example_ids)
        for node_id = node_start:node_end

            node = nodes[node_id]
            
            mars[ex_id, node_id] = 
                if (node isa BitsSum)
                    -Inf32
                elseif (node isa BitsMul)
                    zero(Float32)
                else
                    orig_ex_id::Int32 = example_ids[ex_id]
                    inputnode = node::BitsInput
                    variable = inputnode.variable
                    value = data[orig_ex_id, variable]
                    d = dist(inputnode)
                    if ismissing(value)
                        zero(Float32)
                    elseif d isa BitsCategorical
                        logp = typemin(Float32)
                        for i = 0 : d.num_cats - 1
                            logp = PCs.logsumexp(logp, data[orig_ex_id, variable, i+1] + heap[d.heap_start+i])
                        end
                        logp
                    else
                        @assert false
                    end
                end
        end
    end
    nothing
end

function init_mar!_with_reg(mars, bpc, data, example_ids; mine, maxe, soft_reg, soft_reg_width, debug = false)
    num_examples = length(example_ids)
    num_nodes = length(bpc.nodes)

    mars .= zero(Float32)
    
    dummy_args = (mars, bpc.nodes, data, example_ids, bpc.heap, Int32(1), Int32(1), Float32(soft_reg), Int32(soft_reg_width))
    if data isa CuMatrix
        kernel = @cuda name="init_mar!_with_reg" launch=false init_mar!_with_reg_kernel(dummy_args...) 
    elseif data isa CuArray{Float32, 3}
        kernel = @cuda name="init_mar!_with_reg" launch=false init_mar!_with_reg_cat_kernel(dummy_args...) 
    else
        error("Unknown data type") 
    end
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, node_work = 
        balance_threads(num_nodes, num_examples, config; mine, maxe)
    
    args = (mars, bpc.nodes, data, example_ids, bpc.heap, 
            Int32(num_example_threads), Int32(node_work), Float32(soft_reg), Int32(soft_reg_width))
    
    if debug
        println("Node initialization")
        @show threads blocks num_example_threads node_work num_nodes num_examples
        CUDA.@time kernel(args...; threads, blocks)
    else
        kernel(args...; threads, blocks)
    end
    
    nothing
end

function eval_circuit_with_reg(mars, bpc, data, example_ids; mine, maxe, soft_reg, soft_reg_width, debug = false)
    sum_agg_func(x::Float32, y::Float32) = 
        PCs.logsumexp(x, y)

    init_mar!_with_reg(mars, bpc, data, example_ids; mine, maxe, soft_reg, soft_reg_width, debug)
    layer_start = 1
    for layer_end in bpc.edge_layers_up.ends
        PCs.layer_up(mars, bpc, layer_start, layer_end, length(example_ids); mine, maxe, sum_agg_func, debug)
        layer_start = layer_end + 1
    end
    nothing
end

function loglikelihoods_probcat(bpc::CuBitsProbCircuit, data::CuArray; batch_size, mars_mem = nothing, 
                                mine = 2, maxe = 32)
    num_examples = size(data, 1)
    num_nodes = length(bpc.nodes)

    mars = prep_memory(mars_mem, (batch_size, num_nodes), (false, true))

    log_likelihoods = CUDA.zeros(Float32, num_examples)

    for batch_start = 1 : batch_size : num_examples

        batch_end = min(batch_start + batch_size - 1, num_examples)
        batch = batch_start : batch_end
        num_batch_examples = length(batch)

        eval_circuit_with_reg(mars, bpc, data, batch; mine, maxe, soft_reg = 0.0, soft_reg_width = 1)

        log_likelihoods[batch_start:batch_end] .= @view mars[1:num_batch_examples, end]
    end

    PCs.cleanup_memory(mars, mars_mem)

    log_likelihoods
end

function loglikelihood_probcat(bpc::CuBitsProbCircuit, data::CuArray, weights = nothing; batch_size, mars_mem = nothing, 
                               mine = 2, maxe = 32)
    lls = loglikelihoods_probcat(bpc, data; batch_size, mars_mem, mine, maxe)

    if weights === nothing
        sum(lls) / length(lls)
    else
        sum(lls .* weights) / sum(weights)
    end
end