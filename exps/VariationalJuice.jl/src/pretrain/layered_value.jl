
function compute_head_mars_kernel(mars, head_mars, head_params, nodeheads, layer_start::Int32, layer_end::Int32, num_examples::Int32, num_ex_threads::Int32, node_work::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = layer_start + (node_batch - one(Int32)) * node_work
    node_end = min(node_start + node_work - one(Int32), layer_end)

    @inbounds if ex_id <= num_examples
        
        local acc::Float32
        acc = -Inf32

        for node_id = node_start : node_end

            param_id = node_id - layer_start + 1
            orig_node_id = nodeheads[node_id]

            acc = PCs.logsumexp(acc, mars[ex_id, orig_node_id] + head_params[param_id])
        end
        
        CUDA.@atomic head_mars[ex_id] = PCs.logsumexp(head_mars[ex_id], acc)
    end
    nothing
end

function compute_head_mars(lbpc, mars, head_mars, head_params, num_examples, layer_id; mine, maxe)
    layer_start = layer_id == 1 ? 1 : lbpc.node_layer_head.ends[layer_id-1] + 1
    layer_end = lbpc.node_layer_head.ends[layer_id]
    num_nodes = layer_end - layer_start + 1

    nodeheads = lbpc.node_layer_head.vectors
    
    dummy_args = (mars, head_mars, head_params, nodeheads, Int32(layer_start), Int32(layer_end), Int32(num_examples), Int32(1), Int32(1))
    kernel = @cuda name="compute_head_mars" launch=false compute_head_mars_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_ex_threads, node_work = 
        balance_threads(num_nodes, num_examples, config; mine, maxe)

    args = (mars, head_mars, head_params, nodeheads, Int32(layer_start), Int32(layer_end), Int32(num_examples), Int32(num_ex_threads), Int32(node_work))
    kernel(args...; threads, blocks)
end

function layered_eval_circuit_with_reg(mars, head_mars, head_params, lbpc, data, example_ids, layer_id; 
                                       mine, maxe, soft_reg, soft_reg_width, debug = false)
    sum_agg_func(x::Float32, y::Float32) = 
        PCs.logsumexp(x, y)

    bpc = lbpc.bpc

    num_layers = lbpc.up_layer_end_ids[layer_id]
    @assert num_layers <= length(bpc.edge_layers_up.ends)

    @inbounds @views head_mars .= -Inf32

    init_mar!_with_reg(mars, bpc, data, example_ids; mine, maxe, soft_reg, soft_reg_width, debug)
    
    layer_start = 1
    for (up_layer_id, layer_end) in enumerate(bpc.edge_layers_up.ends)
        PCs.layer_up(mars, bpc, layer_start, layer_end, length(example_ids); mine, maxe, sum_agg_func, debug)
        layer_start = layer_end + 1
        
        if up_layer_id >= num_layers
            break
        end
    end
    
    compute_head_mars(lbpc, mars, head_mars, head_params, length(example_ids), layer_id; mine, maxe)
    
    nothing
end