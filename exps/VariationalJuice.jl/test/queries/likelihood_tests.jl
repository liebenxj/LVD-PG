using Test
using ProbabilisticCircuits: prep_memory, logsumexp

include("../../src/VariationalJuice.jl")
include("../helpers/simple_circuits.jl")


@testset "likelihoods" begin

    pc = simple_3vars_circuit()
    init_parameters(pc; perturbation = 0.4)

    mbpc = CuMetaBitsProbCircuit(pc)

    data = cu(UInt8.([1 2 3; 2 3 1]))

    n_nodes = num_nodes(mbpc)
    n_edges = num_edges(mbpc)
    mars = prep_memory(nothing, (2, n_nodes))
    edge_mars = prep_memory(nothing, (2, n_edges))

    eval_circuit(mars, edge_mars, mbpc, data, 1:2; mine = 2, maxe = 32)

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

end

@testset "likelihoods with per-sample parameters" begin

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

    eval_circuit(mars, edge_mars, mbpc, parameters, data, 1:2; mine = 2, maxe = 32)

    CUDA.allowscalar() do
        @test exp(mars[1,4] + parameters[1,1]) + exp(mars[1,8] + parameters[1,2]) ≈ exp(mars[1,9])
        @test exp(mars[2,4] + parameters[2,1]) + exp(mars[2,8] + parameters[2,2]) ≈ exp(mars[2,9])

        @test mars[1,1] ≈ parameters[1,4]
        @test mars[2,1] ≈ parameters[2,5]
    end

end