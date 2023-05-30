using Test
using ProbabilisticCircuits: prep_memory, logsumexp

include("../../src/VariationalJuice.jl")
include("../helpers/simple_circuits.jl")


@testset "likelihoods with per-sample parameters" begin

    pc = simple_3vars_circuit(; fixed = false)
    init_parameters(pc; perturbation = 0.4)

    cbpc = CuCondBitsProbCircuit(pc)

    data = cu(UInt8.([1 2 3; 2 3 1]))

    parameters = cu(zeros(Float32, 2, 32))
    parameters[1,3:end] .= log(0.2)
    parameters[2,3:end] .= log(0.2)
    parameters[1:2,1:2] = cu(log.(Float32.([0.2 0.8; 0.5 0.5])))
    parameters[1,3:7] .= cu(log.(Float32.([0.2, 0.3, 0.1, 0.1, 0.3])))

    n_nodes = num_nodes(cbpc)
    n_edges = num_edges(cbpc)
    mars = prep_memory(nothing, (2, n_nodes))
    edge_mars = prep_memory(nothing, (2, n_edges))

    eval_conditional_circuit(mars, edge_mars, cbpc, parameters, data, 1:2; mine = 2, maxe = 32)

    CUDA.allowscalar() do
        @test exp(mars[1,4] + parameters[1,1]) + exp(mars[1,8] + parameters[1,2]) ≈ exp(mars[1,9])
        @test exp(mars[2,4] + parameters[2,1]) + exp(mars[2,8] + parameters[2,2]) ≈ exp(mars[2,9])

        @test mars[1,1] ≈ parameters[1,4]
        @test mars[2,1] ≈ parameters[2,5]
    end

end