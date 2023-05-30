
## Add pseudocount

function conditioned_add_pseudocount_kernel(edge_aggr, edges, _node_aggr, pseudocount)
    node_aggr = Base.Experimental.Const(_node_aggr)
    edge_id = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 
    @inbounds if edge_id <= length(edges)
        edge = edges[edge_id]
        if edge isa SumEdge
            parent_id = edge.parent_id
            if edge_aggr[edge_id] > Float32(1e-8)
                CUDA.@atomic edge_aggr[edge_id] += pseudocount / node_aggr[parent_id]
            end
        end
    end      
    nothing
end

function conditioned_add_pseudocount(edge_aggr, node_aggr, bpc, pseudocount; debug = false)
    PCs.count_siblings(node_aggr, bpc)
    edges = bpc.edge_layers_down.vectors
    args = (edge_aggr, edges, node_aggr, Float32(pseudocount))
    kernel = @cuda name="conditioned_add_pseudocount" launch=false conditioned_add_pseudocount_kernel(args...) 
    threads = launch_configuration(kernel.fun).threads
    blocks = cld(length(edges), threads)

    if debug
        println("Add pseudocount")
        @show threads blocks length(edges)
        CUDA.@time kernel(args...; threads, blocks)
    else
        kernel(args...; threads, blocks)
    end
    nothing
end

## Update parameters

function conditioned_update_params_kernel(edges_down, edges_up, _down2upedge, _node_aggr, _edge_aggr, inertia)
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
                parent_flow = node_aggr[parent_id]
                edge_flow = edge_aggr[edge_id_down]

                if edge_flow > Float32(1e-8)
                    old = inertia * exp(edge_down.logp)
                    new = (one(Float32) - inertia) * edge_flow / parent_flow 
                    new_log_param = log(old + new)

                    edges_down[edge_id_down] = 
                        SumEdge(parent_id, edge_down.prime_id, edge_down.sub_id, 
                                new_log_param, edge_down.tag)

                    edges_up[edge_id_up] = 
                        SumEdge(parent_id, edge_down.prime_id, edge_down.sub_id, 
                                new_log_param, edge_up_tag)
                end
            end
        end
    end      
    nothing
end

function conditioned_update_params(bpc, node_aggr, edge_aggr; inertia = 0, debug = false)
    edges_down = bpc.edge_layers_down.vectors
    edges_up = bpc.edge_layers_up.vectors
    down2upedge = bpc.down2upedge
    @assert length(edges_down) == length(down2upedge) == length(edges_up)

    args = (edges_down, edges_up, down2upedge, node_aggr, edge_aggr, Float32(inertia))
    kernel = @cuda name="conditioned_update_params" launch=false conditioned_update_params_kernel(args...) 
    threads = launch_configuration(kernel.fun).threads
    blocks = cld(length(edges_down), threads)
    
    if debug
        println("Update parameters")
        @show threads blocks length(edges_down)
        CUDA.@time kernel(args...; threads, blocks)
    else
        kernel(args...; threads, blocks)
    end
    nothing
end

## Update leaf parameters

function conditioned_update_params(d::BitsCategorical, heap, pseudocount, inertia)
    heap_start = d.heap_start
    num_cats = d.num_cats
    
    @inbounds begin
        # add pseudocount & accumulate node flow
        node_flow = zero(Float32)
        cat_pseudocount = pseudocount / Float32(num_cats)
        for i = 0 : num_cats-1
            node_flow += heap[heap_start+num_cats+i]
        end
        missing_flow = heap[heap_start+UInt32(2)*num_cats]
        node_flow += missing_flow + pseudocount
        
        if node_flow > Float32(1e-6)
            # update parameter
            for i = 0 : num_cats-1
                oldp = exp(heap[heap_start+i])
                old = inertia * oldp
                new = (one(Float32) - inertia) * (heap[heap_start+num_cats+i] + 
                        cat_pseudocount + missing_flow * oldp) / node_flow 
                new_log_param = log(old + new)
                heap[heap_start+i] = new_log_param
            end
        end
    end
    nothing
end

function conditioned_update_input_node_params_kernel(nodes, input_node_ids, heap, pseudocount, inertia)

    node_id = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    @inbounds if node_id <= length(input_node_ids)
        orig_node_id::UInt32 = input_node_ids[node_id]
        inputnode = nodes[orig_node_id]::BitsInput
        conditioned_update_params(dist(inputnode), heap, pseudocount, inertia)
    end
    nothing
end

function conditioned_update_input_node_params(bpc; pseudocount, inertia = 0, debug = false)
    num_input_nodes = length(bpc.input_node_ids)

    args = (bpc.nodes, bpc.input_node_ids, bpc.heap, Float32(pseudocount), Float32(inertia))
    kernel = @cuda name="conditioned_update_input_node_params" launch=false conditioned_update_input_node_params_kernel(args...)
    threads = launch_configuration(kernel.fun).threads
    blocks = cld(num_input_nodes, threads)

    if debug
        println("Update parameters of input nodes")
        @show threads blocks num_input_nodes
        CUDA.@time kernel(args...; threads, blocks)
    else
        kernel(args...; threads, blocks)
    end
    nothing
end

## Main function

function warmup_with_mini_batch_em(lbpc::CuLayeredBitsProbCircuit, data::CuArray; batch_size, num_epochs_per_layer,
                                   flow_memory = 0.0, mine = 2, maxe = 32, pseudocount = 0.1, soft_reg = 0.0, soft_reg_width = 7,
                                   param_inertia = 0.98, param_inertia_end = 0.98, layer_start = 1, layer_interval = 1, verbose = true,
                                   mars_mem = nothing, flows_mem = nothing, head_mars_mem = nothing, head_params_mem = nothing,
                                   head_flows_mem = nothing, node_aggr_mem = nothing, edge_aggr_mem = nothing)
    bpc = lbpc.bpc
    num_examples = size(data, 1)
    num_nodes = length(bpc.nodes)
    num_edges = length(bpc.edge_layers_down.vectors)
    num_batches = num_examples ÷ batch_size # drop last incomplete batch
    max_num_heads = maximum(lbpc.node_layer_head.ends .- vcat([1], lbpc.node_layer_head.ends[1:end-1])) + 1
    num_layers = length(lbpc.down_layer_start_ids)

    @assert batch_size <= num_examples
    @assert num_layers == length(lbpc.up_layer_end_ids)

    mars = prep_memory(mars_mem, (batch_size, num_nodes), (false, true))
    flows = prep_memory(flows_mem, (batch_size, num_nodes), (false, true))
    head_mars = prep_memory(head_mars_mem, (batch_size,))
    head_params = prep_memory(head_params_mem, (max_num_heads,))
    head_flows = prep_memory(head_flows_mem, (max_num_heads,))
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

    clear_aggr_mem(mem_rate = 0.0) = begin
        if iszero(mem_rate)
            edge_aggr .= zero(Float32)
            head_flows .= zero(Float32)
            PCs.clear_input_node_mem(bpc; rate = 0, debug = false)
        else
            # slowly forget old edge aggregates
            rate = max(zero(Float32), one(Float32) - (batch_size + pseudocount) * mem_rate)
            edge_aggr .*= rate
            head_flows .*= rate
            PCs.clear_input_node_mem(bpc; rate)
        end
    end

    param_inertia_init = param_inertia
    Δparam_inertia = (param_inertia_end - param_inertia) / num_epochs_per_layer

    log_likelihoods_layer = CUDA.zeros(Float32, num_batches)

    # Layer-wise pretraining
    layer_ids = collect(layer_start : layer_interval : num_layers)
    if !(num_layers in layer_ids)
        push!(layer_ids, num_layers)
    end
    for up_layer_id in layer_ids

        if verbose
            println("> Start initializing layer $(up_layer_id)/$(num_layers)")
        end

        clear_aggr_mem(0.0)

        param_inertia = param_inertia_init

        # Initialize the weights of the top (pseudo) layer uniformly
        layer_start = up_layer_id == 1 ? 1 : lbpc.node_layer_head.ends[up_layer_id-1] + 1
        layer_end = lbpc.node_layer_head.ends[up_layer_id]
        num_head_nodes = layer_end - layer_start + 1
        @inbounds @views head_params .= log(Float32(1 / (layer_end - layer_start + 1)))

        for epoch = 1 : num_epochs_per_layer

            do_shuffle() # shuffle data per epoch

            @inbounds @views log_likelihoods_layer .= zero(Float32)

            for (batch_id, batch) in enumerate(batches)

                clear_aggr_mem(flow_memory)

                layer_wise_probs_flows_circuit(flows, head_flows, mars, edge_aggr, head_mars, head_params, lbpc, data, batch, up_layer_id; 
                                               mine, maxe, soft_reg, soft_reg_width)
                
                @views sum!(log_likelihoods_layer[batch_id:batch_id], head_mars[1:batch_size])
                
                conditioned_add_pseudocount(edge_aggr, node_aggr, bpc, pseudocount)
                PCs.aggr_node_flows(node_aggr, bpc, edge_aggr)

                conditioned_update_params(bpc, node_aggr, edge_aggr; inertia = param_inertia)
                
                conditioned_update_input_node_params(bpc; pseudocount, inertia = param_inertia)
                
                oldp = exp.(head_params)
                newp = head_flows ./ sum(head_flows[1:num_head_nodes])
                head_params .= log.(oldp .* param_inertia .+ newp .* (one(Float32) - param_inertia))
            end

            if verbose
                CUDA.allowscalar() do
                    tr_ll = sum(log_likelihoods_layer) / num_batches / batch_size
                    println("  - Epoch $(epoch) - train LL: $(tr_ll)")
                end
            end
        end

        param_inertia += Δparam_inertia
    end

    nothing
end