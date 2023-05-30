


function gradients(mbpc::CuMetaBitsProbCircuit, data::CuMatrix, gradients::CuMatrix; batch_size, mine = 2, maxe = 32,
                   mars_mem = nothing, edge_mars_mem = nothing, flows_mem = nothing, edge_flows_mem = nothing,
                   input_edge_flows_mem = nothing, grads_mem = nothing)
    n_examples = size(data, 1)
    n_nodes = num_nodes(mbpc)
    n_edges = num_edges(mbpc)
    n_params = num_parameters(mbpc)
    @assert size(gradients, 2) == n_params

    mars = prep_memory(mars_mem, (batch_size, n_nodes), (false, true))
    edge_mars = prep_memory(edge_mars_mem, (batch_size, n_edges), (false, true))
    flows = prep_memory(flows_mem, (batch_size, n_nodes), (false, true))
    edge_flows = prep_memory(edge_flows_mem, (batch_size, n_edges), (false, true))
    input_edge_flows = prep_memory(input_edge_flows_mem, (batch_size, mbpc.num_input_params), (false, true))
    grads = prep_memory(grads_mem, (batch_size, n_params), (false, true))
    edge_aggr = nothing

    for batch_start = 1 : batch_size : n_examples
        batch_end = min(batch_start + batch_size - 1, n_examples)
        batch = batch_start : batch_end
        num_batch_examples = batch_end - batch_start + 1

        probs_flows_circuit(flows, edge_flows, input_edge_flows, mars, edge_mars, edge_aggr, mbpc, data, batch; mine, maxe)
        gradients_from_flows(edge_flows, edge_mars, mbpc, grads, num_batch_examples; mine, maxe)
        gradients_from_input_flows(input_edge_flows, mbpc, grads, data, batch; mine, maxe)
        
        @inbounds @views gradients[batch, :] .= grads[1:num_batch_examples, :]
    end
    nothing
end

function gradients_from_flows_kernel(edge_flows, edge_mars, edge_layers, param2edge, grads, num_examples, num_ex_threads::Int32, param_work::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    param_batch, ex_id = fldmod1(threadid, num_ex_threads)

    param_start = one(Int32) + (param_batch - one(Int32)) * param_work
    param_end = min(param_start + param_work - one(Int32), length(param2edge))

    @inbounds if ex_id <= num_examples
        for param_id = param_start : param_end
            edge_id = param2edge[param_id]
            edge = edge_layers[edge_id]::SumEdge
            grad = edge_flows[ex_id, edge_id] / exp(edge_mars[ex_id, edge_id] - edge.logp)
            grads[ex_id, param_id] = grad
        end
    end
    nothing
end

function gradients_from_flows(edge_flows, edge_mars, mbpc, grads, num_examples; mine = 2, maxe = 32)
    bpc = mbpc.bpc
    param2edge = mbpc.param2edge
    edge_layers = bpc.edge_layers_up.vectors

    dummy_args = (edge_flows, edge_mars, edge_layers, param2edge, grads, num_examples, Int32(1), Int32(1))
    kernel = @cuda name="gradients_from_flows" launch=false gradients_from_flows_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, param_work = 
        balance_threads(length(param2edge), num_examples, config; mine, maxe)

    args = (edge_flows, edge_mars, edge_layers, param2edge, grads, num_examples, Int32(num_example_threads), Int32(param_work))
    kernel(args...; threads, blocks)
end

function gradients_from_input_flows_kernel(input_edge_flows, nodes, grads, input_node_ids, innode2ncumparam, num_inner_params, data, example_ids, 
                                           num_ex_threads::Int32, node_work::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = one(Int32) + (node_batch - one(Int32)) * node_work
    node_end = min(node_start + node_work - one(Int32), length(input_node_ids))

    @inbounds if ex_id <= length(example_ids)
        for idx = node_start : node_end
            node_id = input_node_ids[idx]
            node = nodes[node_id]::BitsInput
            d = dist(node)

            nparams = num_parameters(d, false)
            if nparams > 0
                param_id_base = num_inner_params + innode2ncumparam[idx] - nparams
                for param_id = 1 : nparams
                    # TODO: this is hacky and doesn't work for other types of input node
                    grads[ex_id, param_id_base + param_id] = input_edge_flows[ex_id, innode2ncumparam[idx] - nparams + param_id]
                end
            end
        end
    end
    nothing
end

function gradients_from_input_flows(input_edge_flows, mbpc, grads, data, example_ids; mine = 2, maxe = 32)
    bpc = mbpc.bpc
    num_inner_params = length(mbpc.param2edge)
    innode2ncumparam = mbpc.innode2ncumparam
    input_node_ids = bpc.input_node_ids

    dummy_args = (input_edge_flows, bpc.nodes, grads, input_node_ids, innode2ncumparam, num_inner_params, data, example_ids, Int32(1), Int32(1))
    kernel = @cuda name="gradients_from_input_flows" launch=false gradients_from_input_flows_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, node_work = 
        balance_threads(length(input_node_ids), length(example_ids), config; mine, maxe)

    args = (input_edge_flows, bpc.nodes, grads, input_node_ids, innode2ncumparam, 
            num_inner_params, data, example_ids, Int32(num_example_threads), Int32(node_work))
    kernel(args...; threads, blocks)
end