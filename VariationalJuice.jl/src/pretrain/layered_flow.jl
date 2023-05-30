

function compute_head_flows_kernel(flows, head_flows, mars, head_mars, head_params, nodeheads,
                                   layer_start::Int32, layer_end::Int32, num_examples::Int32, num_ex_threads::Int32, node_work::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = layer_start + (node_batch - one(Int32)) * node_work
    node_end = min(node_start + node_work - one(Int32), layer_end)

    @inbounds if ex_id <= num_examples
        for node_id = node_start : node_end

            param_id = node_id - layer_start + 1
            orig_node_id = nodeheads[node_id]

            flow = exp(mars[ex_id, orig_node_id] + head_params[param_id] - head_mars[ex_id])
            
            CUDA.@atomic head_flows[param_id] += flow

            flows[ex_id, orig_node_id] = flow
        end
    end
    nothing
end

function compute_head_flows(lbpc, flows, head_flows, mars, head_mars, head_params, layer_id, num_examples; mine = 2, maxe = 32)
    layer_start = layer_id == 1 ? 1 : lbpc.node_layer_head.ends[layer_id-1] + 1
    layer_end = lbpc.node_layer_head.ends[layer_id]
    num_nodes = layer_end - layer_start + 1

    nodeheads = lbpc.node_layer_head.vectors

    dummy_args = (flows, head_flows, mars, head_mars, head_params, nodeheads, 
                  Int32(layer_start), Int32(layer_end), Int32(num_examples), Int32(1), Int32(1))
    kernel = @cuda name="compute_head_flows" launch=false compute_head_flows_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_ex_threads, node_work = 
        balance_threads(num_nodes, num_examples, config; mine, maxe)

    args = (flows, head_flows, mars, head_mars, head_params, nodeheads, 
            Int32(layer_start), Int32(layer_end), Int32(num_examples), Int32(num_ex_threads), Int32(node_work))
    kernel(args...; threads, blocks)
end

function layered_flows_circuit(data, flows, head_flows, mars, head_mars, head_params, edge_aggr, lbpc, 
                               example_ids, layer_id; mine, maxe, soft_reg, soft_reg_width, debug=false)
    num_examples = length(example_ids)
    dlayer_start_id = lbpc.down_layer_start_ids[layer_id]

    bpc = lbpc.bpc
    edges = bpc.edge_layers_down.vectors
    
    @inbounds @views flows .= zero(Float32)

    compute_head_flows(lbpc, flows, head_flows, mars, head_mars, head_params, layer_id, num_examples)
    
    layer_start = dlayer_start_id == 1 ? 1 : bpc.edge_layers_down.ends[dlayer_start_id-1] + 1
    for layer_end in bpc.edge_layers_down.ends[dlayer_start_id:end]

        PCs.layer_down(flows, edge_aggr, bpc, mars, layer_start, layer_end, num_examples; 
                       mine, maxe, debug)
        layer_start = layer_end + 1
    end

    input_flows_circuit_with_reg(flows, bpc, data, example_ids; mine, maxe, soft_reg, soft_reg_width, debug)

    nothing
end

function layer_wise_probs_flows_circuit(flows, head_flows, mars, edge_aggr, head_mars, head_params, lbpc, data, batch, layer_id; 
                                        mine, maxe, soft_reg, soft_reg_width)
    layered_eval_circuit_with_reg(mars, head_mars, head_params, lbpc, data, batch, layer_id; 
                                  mine, maxe, soft_reg, soft_reg_width, debug = false)
    layered_flows_circuit(data, flows, head_flows, mars, head_mars, head_params, edge_aggr, lbpc, 
                          batch, layer_id; mine, maxe, soft_reg, soft_reg_width, debug = false)
    nothing
end