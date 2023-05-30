
##########################
## Top-down probability ##
##########################

function td_prob_inner_kernel(td_probs, param_td_probs, edges, _down2upedge, _edge2param, _param2group, 
                              num_ex_threads::Int32, layer_start::Int32, edge_work::Int32, layer_end::Int32)

    down2upedge = Base.Experimental.Const(_down2upedge)
    edge2param = Base.Experimental.Const(_edge2param)
    param2group = Base.Experimental.Const(_param2group)
        
    threadid_block = threadIdx().x
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadid_block 

    edge_batch, ex_id = fldmod1(threadid, num_ex_threads)

    edge_start = layer_start + (edge_batch-one(Int32)) * edge_work 
    edge_end = min(edge_start + edge_work - one(Int32), layer_end) 

    local acc::Float32

    owned_node::Bool = false
    
    @inbounds if ex_id <= 1
        for edge_id = edge_start:edge_end

            edge = edges[edge_id]

            parent_id = edge.parent_id
            prime_id = edge.prime_id
            sub_id = edge.sub_id

            tag = edge.tag
            firstedge = isfirst(tag)
            lastedge = islast(tag)
            issum = edge isa SumEdge
            up_edge_id = down2upedge[edge_id]
            
            if firstedge
                partial = ispartial(tag)
                owned_node = !partial
            end
                
            edge_td_prob = td_probs[parent_id]

            if issum
                param_id = edge2param[up_edge_id]
                param_group_id = param2group[param_id]
                
                prob = exp(edge.logp)
                
                edge_td_prob *= prob
                param_td_probs[param_id] = edge_td_prob
            end

            if sub_id != 0
                if isonlysubedge(tag)
                    td_probs[sub_id] = edge_td_prob
                else
                    CUDA.@atomic td_probs[sub_id] += edge_td_prob
                end            
            end

            # accumulate td_probs from parents
            if firstedge || (edge_id == edge_start)  
                acc = edge_td_prob
            else
                acc += edge_td_prob
            end

            # write to global memory
            if lastedge || (edge_id == edge_end)   
                if lastedge && owned_node
                    # no one else is writing to this global memory
                    td_probs[prime_id] = acc
                else
                    CUDA.@atomic td_probs[prime_id] += acc
                end
            end
        end
    end

    nothing
end

function td_prob_inner_layer(td_probs, param_td_probs, mbpc, layer_start, layer_end; mine, maxe, debug=false)
    bpc = mbpc.bpc
    edges = bpc.edge_layers_down.vectors
    num_edges = layer_end-layer_start+1
    down2upedge = bpc.down2upedge
    edge2param = mbpc.edge2param
    param2group = mbpc.param2group
    num_examples = 1

    dummy_args = (td_probs, param_td_probs, edges,
                  down2upedge, edge2param, param2group, Int32(32), Int32(1), Int32(1), Int32(2))
    kernel = @cuda name="td_prob_inner" launch=false td_prob_inner_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    # configure thread/block balancing
    threads, blocks, num_example_threads, edge_work = 
        balance_threads(num_edges, num_examples, config; mine, maxe, contiguous_warps=true)
    
    args = (td_probs, param_td_probs, edges,
            down2upedge, edge2param, param2group, Int32(num_example_threads), 
            Int32(layer_start), Int32(edge_work), Int32(layer_end))
    if debug
        println("Layer $layer_start:$layer_end")
        @show threads blocks num_example_threads edge_work, num_edges num_examples
        CUDA.@time kernel(args...; threads, blocks)
    else
        kernel(args...; threads, blocks)
    end

    nothing
end

function td_prob_inner(td_probs, param_td_probs, mbpc::CuMetaBitsProbCircuit; mine = 2, maxe = 32, debug = false)
    bpc = mbpc.bpc

    init_samples() = begin 
        CUDA.allowscalar() do
            td_probs .= zero(Float32)
            td_probs[end] = one(Float32)
        end
    end
    if debug
        println("Initializing flows")
        CUDA.@time CUDA.@sync init_samples()
    else
        init_samples()
    end

    layer_start = 1
    for layer_end in bpc.edge_layers_down.ends
        td_prob_inner_layer(td_probs, param_td_probs, mbpc, layer_start, layer_end; 
                            mine, maxe, debug)
        layer_start = layer_end + 1
    end

    nothing
end

function td_prob(mbpc::CuMetaBitsProbCircuit)
    n_nodes = length(mbpc.bpc.nodes)
    n_pars = num_parameters(mbpc)

    td_probs = prep_memory(nothing, (n_nodes,))
    param_td_probs = prep_memory(nothing, (n_pars,))

    td_prob_inner(td_probs, param_td_probs, mbpc)

    Array(param_td_probs)
end

###############################################
## Conditioned (masked) top-down probability ##
###############################################

function cond_td_prob_inner_kernel(td_probs, marked_pc_idx1, marked_pc_idx2, node_probs, edges, num_examples::Int32, 
                                   num_ex_threads::Int32, layer_start::Int32, edge_work::Int32, layer_end::Int32)
        
    threadid_block = threadIdx().x
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadid_block 

    edge_batch, ex_id = fldmod1(threadid, num_ex_threads)

    edge_start = layer_start + (edge_batch - one(Int32)) * edge_work 
    edge_end = min(edge_start + edge_work - one(Int32), layer_end) 

    local acc::Float32

    owned_node::Bool = false
    
    @inbounds if ex_id <= num_examples
        for edge_id = edge_start:edge_end

            edge = edges[edge_id]

            parent_id = edge.parent_id
            prime_id = edge.prime_id
            sub_id = edge.sub_id

            if marked_pc_idx1[parent_id] > 0
                z_var_id = marked_pc_idx1[parent_id]
                z_cat_count = marked_pc_idx2[parent_id]
                td_probs[ex_id, parent_id] = node_probs[ex_id, z_var_id, z_cat_count]
            end

            tag = edge.tag
            firstedge = isfirst(tag)
            lastedge = islast(tag)
            issum = edge isa SumEdge
            
            if firstedge
                partial = ispartial(tag)
                owned_node = !partial
            end
                
            edge_td_prob = td_probs[ex_id, parent_id]

            if issum
                prob = exp(edge.logp)
                edge_td_prob *= prob
            end

            if sub_id != 0
                if isonlysubedge(tag)
                    td_probs[ex_id, sub_id] = edge_td_prob
                else
                    CUDA.@atomic td_probs[ex_id, sub_id] += edge_td_prob
                end            
            end

            # accumulate td_probs from parents
            if firstedge || (edge_id == edge_start)  
                acc = edge_td_prob
            else
                acc += edge_td_prob
            end

            # write to global memory
            if lastedge || (edge_id == edge_end)   
                if lastedge && owned_node
                    # no one else is writing to this global memory
                    td_probs[ex_id, prime_id] = acc
                else
                    CUDA.@atomic td_probs[ex_id, prime_id] += acc
                end
            end
        end
    end

    nothing
end

function cond_td_prob_inner_layer(mbpc, td_probs, marked_pc_idx1, marked_pc_idx2, node_probs, 
                                  layer_start, layer_end; mine = 2, maxe = 32)
    bpc = mbpc.bpc
    edges = bpc.edge_layers_down.vectors
    num_edges = layer_end-layer_start+1
    num_examples = size(td_probs, 1)

    dummy_args = (td_probs, marked_pc_idx1, marked_pc_idx2, node_probs, edges, Int32(num_examples),
                  Int32(32), Int32(1), Int32(1), Int32(2))
    kernel = @cuda name="cond_td_prob_inner" launch=false cond_td_prob_inner_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    # configure thread/block balancing
    threads, blocks, num_example_threads, edge_work = 
        balance_threads(num_edges, num_examples, config; mine, maxe, contiguous_warps=true)
    
    args = (td_probs, marked_pc_idx1, marked_pc_idx2, node_probs, edges, 
            Int32(num_examples), Int32(num_example_threads), 
            Int32(layer_start), Int32(edge_work), Int32(layer_end))
    
    kernel(args...; threads, blocks)

    nothing
end

function conditioned_td_prob_inner(mbpc::CuMetaBitsProbCircuit, td_probs, marked_pc_idx1, marked_pc_idx2, node_probs)
    bpc = mbpc.bpc

    # init td probs
    td_probs .= zero(Float32)

    layer_start = 1
    for layer_end in bpc.edge_layers_down.ends
        cond_td_prob_inner_layer(mbpc, td_probs, marked_pc_idx1, marked_pc_idx2, node_probs, 
                                 layer_start, layer_end; mine = 2, maxe = 32)
        layer_start = layer_end + 1
    end
end

function conditioned_td_prob(mbpc::CuMetaBitsProbCircuit, marked_pc_idx1, marked_pc_idx2, node_probs,
                             input_aggr_params; td_probs_mem = nothing)
    num_examples = size(node_probs, 1)
    n_nodes = length(mbpc.bpc.nodes)

    td_probs = prep_memory(td_probs_mem, (num_examples, n_nodes))

    input_aggr_params .= zero(Float32)

    conditioned_td_prob_inner(mbpc, td_probs, marked_pc_idx1, marked_pc_idx2, node_probs)

    gumbel_input_sample_down(td_probs, nothing, input_aggr_params, mbpc, 1 : num_examples; 
                             mine = 2, maxe = 32, norm_params = true)

    nothing
end