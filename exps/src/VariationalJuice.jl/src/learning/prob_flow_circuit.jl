

##################################################################################
# Downward pass for input nodes
##################################################################################

# import ProbabilisticCircuits: input_flows_circuit # extend

function input_flows_circuit_kernel2(flows, nodes, input_node_ids, heap, 
                                     example_ids, num_ex_threads::Int32, node_work::Int32)

    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = one(Int32) + (node_batch - one(Int32)) * node_work
    node_end = min(node_start + node_work - one(Int32), length(input_node_ids))

    @inbounds if ex_id <= length(example_ids)
        for node_id = node_start : node_end
            orig_ex_id::Int32 = example_ids[ex_id]
            orig_node_id::UInt32 = input_node_ids[node_id]
            node_flow::Float32 = flows[ex_id, orig_node_id]
            inputnode = nodes[orig_node_id]::BitsInput
            variable = inputnode.variable
            flow(dist(inputnode), data, orig_ex_id, variable, node_flow, heap)
        end
    end
    nothing
end

function input_flows_circuit(flows, bpc, data::CuArray{Float32,3}, example_ids; mine, maxe, debug=false)
    num_examples = length(example_ids)
    num_input_nodes = length(bpc.input_node_ids)

    dummy_args = (flows, bpc.nodes, bpc.input_node_ids,
                  bpc.heap, data, example_ids, Int32(1), Int32(1))
    kernel = @cuda name="input_flows_circuit" launch=false input_flows_circuit_kernel2(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, node_work = 
        balance_threads(num_input_nodes, num_examples, config; mine, maxe)

    args = (flows, bpc.nodes, bpc.input_node_ids,
            bpc.heap, data, example_ids, Int32(num_example_threads), Int32(node_work))
    if debug
        println("Flows of input nodes")
        @show threads blocks num_example_threads node_work num_nodes num_examples
        CUDA.@time kernel(args...; threads, blocks)
    else
        kernel(args...; threads, blocks)
    end

    nothing
end

##################################################################################
# Init marginals
###################################################################################

import ProbabilisticCircuits: init_mar! # extend

function init_mar!_kernel2(mars, nodes, data, example_ids, heap, 
                           num_ex_threads::Int32, node_work::Int32, input_init_func)
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
                    loglikelihood(dist(inputnode), data, orig_ex_id, variable, heap)
                end
        end
    end
    nothing
end

function init_mar!(mars, bpc, data::CuArray{Float32,3}, example_ids; mine, maxe, input_init_func, debug=false)
    num_examples = length(example_ids)
    num_nodes = length(bpc.nodes)
    
    dummy_args = (mars, bpc.nodes, data, example_ids, bpc.heap, Int32(1), Int32(1), input_init_func)
    kernel = @cuda name="init_mar!2" launch=false init_mar!_kernel2(dummy_args...) 
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, node_work = 
        balance_threads(num_nodes, num_examples, config; mine, maxe)
    
    args = (mars, bpc.nodes, data, example_ids, bpc.heap, 
            Int32(num_example_threads), Int32(node_work), input_init_func)
    if debug
        println("Node initialization")
        @show threads blocks num_example_threads node_work num_nodes num_examples
        CUDA.@time kernel(args...; threads, blocks)
    else
        kernel(args...; threads, blocks)
    end
    nothing
end