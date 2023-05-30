using Test
using ProbabilisticCircuits: prep_memory, logsumexp

include("../../src/VariationalJuice.jl")
include("../helpers/simple_circuits.jl")


@testset "flows" begin

    pc = simple_3vars_circuit(; fixed = false)
    init_parameters(pc; perturbation = 0.4)

    mbpc = CuMetaBitsProbCircuit(pc)

    data = cu(UInt8.([1 2 3; 2 3 1]))

    n_nodes = num_nodes(mbpc)
    n_edges = num_edges(mbpc)
    mars = prep_memory(nothing, (2, n_nodes))
    edge_mars = prep_memory(nothing, (2, n_edges))
    flows = prep_memory(nothing, (2, n_nodes))
    edge_flows = prep_memory(nothing, (2, n_edges))
    input_edge_flows = prep_memory(nothing, (2, mbpc.num_input_params))
    edge_aggr = nothing

    probs_flows_circuit(flows, edge_flows, input_edge_flows, mars, edge_mars, 
                        edge_aggr, mbpc, data, 1:2; mine = 2, maxe = 32)

    CUDA.allowscalar() do 
        @test logsumexp(edge_mars[1, end-1], edge_mars[1, end]) ≈ mars[1, end]
        @test logsumexp(edge_mars[2, end-1], edge_mars[2, end]) ≈ mars[2, end]

        @test mars[1, 1] + mars[1, 2] ≈ edge_mars[1, 1]
        @test mars[1, 3] ≈ edge_mars[1, 2]
        @test mars[1, 5] + mars[1, 6] ≈ edge_mars[1, 3]
        @test mars[1, 7] ≈ edge_mars[1, 4]

        @test mars[2, 1] + mars[2, 2] ≈ edge_mars[2, 1]
        @test mars[2, 3] ≈ edge_mars[2, 2]
        @test mars[2, 5] + mars[2, 6] ≈ edge_mars[2, 3]
        @test mars[2, 7] ≈ edge_mars[2, 4]
    end

    CUDA.allowscalar() do 
        @test edge_flows[1,5] + edge_flows[1,6] ≈ flows[1,9]
        @test edge_flows[1,3] ≈ flows[1,8]
        @test edge_flows[1,4] ≈ flows[1,8]
        @test edge_flows[1,1] ≈ flows[1,4]
        @test edge_flows[1,2] ≈ flows[1,4]

        @test edge_flows[2,5] + edge_flows[2,6] ≈ flows[2,9]
        @test edge_flows[2,3] ≈ flows[2,8]
        @test edge_flows[2,4] ≈ flows[2,8]
        @test edge_flows[2,1] ≈ flows[2,4]
        @test edge_flows[2,2] ≈ flows[2,4]
    end

    CUDA.allowscalar() do 
        @test input_edge_flows[1,1] ≈ flows[1,1]
        @test input_edge_flows[1,2] ≈ zero(Float32)
        @test input_edge_flows[1,3] ≈ zero(Float32)
        @test input_edge_flows[1,4] ≈ zero(Float32)
        @test input_edge_flows[1,5] ≈ zero(Float32)
    end

end

@testset "flows with param set" begin

    pc = simple_3vars_circuit(; fixed = false)
    init_parameters(pc; perturbation = 0.4)

    mbpc = CuMetaBitsProbCircuit(pc)

    data = cu(UInt8.([1 2 3; 2 3 1]))

    pars = cu(zeros(Float32, 32))
    vectorize_parameters(mbpc, pars)
    parameters = cu(zeros(Float32, 2, 32))
    parameters[1,3:end] .= pars[3:end]
    parameters[2,3:end] .= pars[3:end]
    parameters[1:2,1:2] = cu(log.(Float32.([0.2 0.8; 0.5 0.5])))
    parameters[1,3:7] .= cu(log.(Float32.([0.2, 0.3, 0.1, 0.1, 0.3])))

    n_nodes = num_nodes(mbpc)
    n_edges = num_edges(mbpc)
    mars = prep_memory(nothing, (2, n_nodes))
    edge_mars = prep_memory(nothing, (2, n_edges))
    flows = prep_memory(nothing, (2, n_nodes))
    edge_flows = prep_memory(nothing, (2, n_edges))
    input_edge_flows = prep_memory(nothing, (2, mbpc.num_input_params))
    edge_aggr = nothing

    probs_flows_circuit(flows, edge_flows, input_edge_flows, mars, edge_mars, edge_aggr, 
                        mbpc, data, parameters, 1:2; mine = 2, maxe = 32)

    CUDA.allowscalar() do 
        @test edge_flows[1,5] + edge_flows[1,6] ≈ flows[1,9]
        @test edge_flows[1,3] ≈ flows[1,8]
        @test edge_flows[1,4] ≈ flows[1,8]
        @test edge_flows[1,1] ≈ flows[1,4]
        @test edge_flows[1,2] ≈ flows[1,4]

        @test edge_flows[2,5] + edge_flows[2,6] ≈ flows[2,9]
        @test edge_flows[2,3] ≈ flows[2,8]
        @test edge_flows[2,4] ≈ flows[2,8]
        @test edge_flows[2,1] ≈ flows[2,4]
        @test edge_flows[2,2] ≈ flows[2,4]
    end

    CUDA.allowscalar() do
        @test exp(edge_mars[1,5] - logsumexp(edge_mars[1,5], edge_mars[1,6])) ≈ edge_flows[1,5]
        @test exp(edge_mars[2,5] - logsumexp(edge_mars[2,5], edge_mars[2,6])) ≈ edge_flows[2,5]
    end

end

@testset "per-sample normalized flow tests" begin

    if isfile("../models/mnist_16.jpc")
        pc = read("../models/mnist_16.jpc", ProbCircuit, JpcFormat(), true)
        mbpc = CuMetaBitsProbCircuit(pc);

        n_pars = num_parameters(mbpc)

        pars = prep_memory(nothing, (n_pars,))
        vectorize_parameters(mbpc, pars)
        pars = reshape(pars, (1, :))

        data = cu(rand(0:255, (1, 28*28)))

        new_params = prep_memory(nothing, (1, n_pars))
        
        per_sample_normalized_flows(mbpc, data, pars, new_params; batch_size = 1)

        @test !any(isnan.(new_params))
    end

end