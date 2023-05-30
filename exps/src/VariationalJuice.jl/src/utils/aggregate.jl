

function aggr_param_statistics_inner_kernel(node_stats, par_statistics, edges, _edge2param, 
                                            num_ex_threads::Int32, layer_start::Int32, edge_work::Int32, layer_end::Int32)

    edge2param = Base.Experimental.Const(_edge2param)
        
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

            if edge isa SumEdge
                parent_id = edge.parent_id

                param_id = edge2param[edge_id]
                par_stats = par_statistics[param_id]
                CUDA.@atomic node_stats[parent_id] += par_stats
            end
        end
    end

    nothing
end

function aggr_param_statistics_layer(node_stats, par_statistics, mbpc, layer_start, layer_end; mine, maxe, debug=false)
    bpc = mbpc.bpc
    edges = bpc.edge_layers_up.vectors
    num_edges = layer_end-layer_start+1
    edge2param = mbpc.edge2param
    num_examples = 1

    dummy_args = (node_stats, par_statistics, edges,
                  edge2param, Int32(32), Int32(1), Int32(1), Int32(2))
    kernel = @cuda name="aggr_param_statistics_inner" launch=false aggr_param_statistics_inner_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    # configure thread/block balancing
    threads, blocks, num_example_threads, edge_work = 
        balance_threads(num_edges, num_examples, config; mine, maxe, contiguous_warps=true)
    
    args = (node_stats, par_statistics, edges,
            edge2param, Int32(num_example_threads), 
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

function aggregate_param_statistics_inner(node_stats, par_statistics, mbpc::CuMetaBitsProbCircuit; mine = 2, maxe = 32, debug = false)
    bpc = mbpc.bpc

    init_samples() = begin 
        node_stats .= zero(Float32)
    end
    if debug
        println("Initializing flows")
        CUDA.@time CUDA.@sync init_samples()
    else
        init_samples()
    end

    aggr_param_statistics_layer(node_stats, par_statistics, mbpc, 1, bpc.edge_layers_up.ends[end]; 
                                mine, maxe, debug)

    nothing
end

function aggregate_param_statistics(mbpc::CuMetaBitsProbCircuit, par_statistics)
    n_nodes = length(mbpc.bpc.nodes)
    n_pars = num_parameters(mbpc)

    node_stats = prep_memory(nothing, (n_nodes,))
    
    if par_statistics isa Vector{Float32}
        par_statistics = cu(par_statistics)
    end

    aggregate_param_statistics_inner(node_stats, par_statistics, mbpc)

    Array(node_stats)
end