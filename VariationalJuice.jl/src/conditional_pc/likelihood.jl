

#####################################################
## Compute likelihoods using a matrix of parameters
#####################################################

function cpc_layer_up_kernel(mars, edge_mars, edges, params, edge2param, example_ids, num_ex_threads::Int32, 
                             layer_start::Int32, edge_work::Int32, layer_end::Int32, sum_agg_func)

    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    edge_batch, ex_id = fldmod1(threadid, num_ex_threads)

    edge_start = layer_start + (edge_batch - one(Int32)) * edge_work 
    edge_end = min(edge_start + edge_work - one(Int32), layer_end)

    @inbounds if ex_id <= length(example_ids)

        local acc::Float32    
        owned_node::Bool = false
        
        for edge_id = edge_start:edge_end

            edge = edges[edge_id]

            tag = edge.tag
            isfirstedge = isfirst(tag)
            islastedge = islast(tag)
            issum = edge isa SumEdge
            owned_node |= isfirstedge

            # compute probability coming from child
            child_prob = mars[ex_id, edge.prime_id]
            if edge.sub_id != 0
                child_prob += mars[ex_id, edge.sub_id]
            end
            if issum
                orig_ex_id = example_ids[ex_id]
                param_id = edge2param[edge_id] # use the specific parameter
                child_prob += params[orig_ex_id, param_id]
            end

            # record edge value
            edge_mars[ex_id, edge_id] = child_prob

            # accumulate probability from child
            if isfirstedge || (edge_id == edge_start)  
                acc = child_prob
            elseif issum
                acc = sum_agg_func(acc, child_prob)
            else
                acc += child_prob
            end

            # write to global memory
            if islastedge || (edge_id == edge_end)   
                pid = edge.parent_id

                if islastedge && owned_node
                    # no one else is writing to this global memory
                    mars[ex_id, pid] = acc
                else
                    if issum
                        CUDA.@atomic mars[ex_id, pid] = sum_agg_func(mars[ex_id, pid], acc)
                    else
                        CUDA.@atomic mars[ex_id, pid] += acc
                    end 
                end    
            end
        end
    end
    nothing
end

function cpc_layer_up(mars, edge_mars, bpc, params, edge2param, example_ids, layer_start, layer_end; mine, maxe, sum_agg_func, debug=false)
    edges = bpc.edge_layers_up.vectors
    num_edges = layer_end - layer_start + 1
    dummy_args = (mars, edge_mars, edges, params, edge2param, example_ids,
                  Int32(32), Int32(1), Int32(1), Int32(2), sum_agg_func)
    kernel = @cuda name="cpc_layer_up" launch=false cpc_layer_up_kernel(dummy_args...) 
    config = launch_configuration(kernel.fun)
    num_examples = length(example_ids)
    
    # configure thread/block balancing
    threads, blocks, num_example_threads, edge_work = 
        balance_threads(num_edges, num_examples, config; mine, maxe)
    
    args = (mars, edge_mars, edges, params, edge2param, example_ids,
            Int32(num_example_threads), Int32(layer_start), Int32(edge_work), Int32(layer_end), sum_agg_func)
    if debug
        println("Layer $layer_start:$layer_end")
        @show num_edges num_examples threads blocks num_example_threads edge_work
        CUDA.@time kernel(args...; threads, blocks)
    else
        kernel(args...; threads, blocks)
    end
    nothing
end

import ProbabilisticCircuits: init_mar!_kernel, init_mar! # extend

function cpc_init_mar!_kernel(mars, nodes, data, params, innode2ncumparam, node2inputid, num_inner_params, example_ids, heap, 
                              num_ex_threads::Int32, node_work::Int32, input_init_func, soft_reg::Float32, soft_reg_width::Int32)
    # this kernel follows the structure of the layer eval kernel, would probably be faster to 
    # have 1 thread process multiple examples, rather than multiple nodes 
    
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = one(Int32) + (node_batch - one(Int32)) * node_work 
    node_end = min(node_start + node_work - one(Int32), length(nodes))

    if ex_id <= length(example_ids)
        for node_id = node_start:node_end

            node = nodes[node_id]
            
            mars[ex_id, node_id] = 
                if (node isa BitsSum)
                    -Inf32
                elseif (node isa BitsMul)
                    zero(Float32)
                else # node isa BitsInput
                    orig_ex_id::Int32 = example_ids[ex_id]
                    input_node_id = node2inputid[node_id]

                    inputnode = node::BitsInput
                    variable = inputnode.variable

                    param_start_id = num_inner_params + innode2ncumparam[input_node_id] - num_parameters(dist(inputnode), false) + 1
                    
                    value = data[orig_ex_id, variable]
                    if ismissing(value)
                        @assert false
                    else
                        if soft_reg < Float32(1e-6)
                            loglikelihood(dist(inputnode), value, heap, params, orig_ex_id, param_start_id)
                        else
                            soft_loglikelihood(dist(inputnode), value, heap, params, orig_ex_id, param_start_id, soft_reg, soft_reg_width)
                        end
                    end
                end
        end
    end
    nothing
end

function cpc_init_mar!(mars, cbpc, data, params, example_ids, soft_reg, soft_reg_width; mine, maxe, input_init_func, debug=false)
    bpc = cbpc.bpc
    num_examples = length(example_ids)
    num_nodes = length(bpc.nodes)
    innode2ncumparam = cbpc.innode2ncumparam
    node2inputid = cbpc.node2inputid
    num_inner_params = length(cbpc.param2edge)
    
    dummy_args = (mars, bpc.nodes, data, params, innode2ncumparam, node2inputid, num_inner_params,
                  example_ids, bpc.heap, Int32(1), Int32(1), input_init_func, Float32(soft_reg), Int32(soft_reg_width))
    kernel = @cuda name="cpc_init_mar!" launch=false cpc_init_mar!_kernel(dummy_args...) 
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, node_work = 
        balance_threads(num_nodes, num_examples, config; mine, maxe)
    
    args = (mars, bpc.nodes, data, params, innode2ncumparam, node2inputid, num_inner_params, 
            example_ids, bpc.heap, Int32(num_example_threads), Int32(node_work), input_init_func, 
            Float32(soft_reg), Int32(soft_reg_width))
    if debug
        println("Node initialization")
        @show threads blocks num_example_threads node_work num_nodes num_examples
        CUDA.@time kernel(args...; threads, blocks)
    else
        kernel(args...; threads, blocks)
    end
    nothing
end

# run entire circuit
function eval_conditional_circuit(mars, edge_mars, cbpc::CuCondBitsProbCircuit, params, data, example_ids; 
                                  mine, maxe, debug=false, soft_reg = 0.0, soft_reg_width = 1)
    input_init_func(dist, heap) = 
        zero(Float32)

    sum_agg_func(x::Float32, y::Float32) =
        logsumexp(x, y)

    bpc = cbpc.bpc

    cpc_init_mar!(mars, cbpc, data, params, example_ids, Float32(soft_reg), Int32(soft_reg_width); mine, maxe, input_init_func, debug)
    layer_start = 1
    for layer_end in bpc.edge_layers_up.ends
        cpc_layer_up(mars, edge_mars, bpc, params, cbpc.edge2param, example_ids, layer_start, layer_end; mine, maxe, sum_agg_func, debug)
        layer_start = layer_end + 1
    end
    nothing
end

import ProbabilisticCircuits: loglikelihoods

function loglikelihoods(cbpc::CuCondBitsProbCircuit, data::CuArray, params::CuArray; batch_size::Integer,
                        mine = 2, maxe = 32, soft_reg = 0.0, soft_reg_width = 1, mars_mem = nothing, edge_mars_mem = nothing)

    n_examples = size(data, 1)
    n_nodes = num_nodes(cbpc)
    n_edges = num_edges(cbpc)

    mars = prep_memory(mars_mem, (batch_size, n_nodes), (false, true))
    edge_mars = prep_memory(edge_mars_mem, (batch_size, n_edges), (false, true))

    log_likelihoods = CUDA.zeros(Float32, n_examples)

    for batch_start = 1 : batch_size : n_examples
        batch_end = min(batch_start + batch_size - 1, n_examples)
        batch = batch_start : batch_end
        num_batch_examples = batch_end - batch_start + 1

        eval_conditional_circuit(mars, edge_mars, cbpc, params, data, batch; mine, maxe, soft_reg, soft_reg_width)
        @inbounds @views log_likelihoods[batch_start:batch_end] .= mars[1:num_batch_examples, end]
    end

    log_likelihoods
end
