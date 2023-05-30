using Test
using ProbabilisticCircuits: prep_memory, logsumexp

include("../../src/VariationalJuice.jl")
include("../helpers/simple_circuits.jl")


@testset "flows with param set" begin

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
    flows = prep_memory(nothing, (2, n_nodes))
    edge_flows = prep_memory(nothing, (2, n_edges))
    input_edge_flows = prep_memory(nothing, (2, cbpc.num_input_params))
    edge_aggr = nothing

    conditional_circuit_flows(flows, edge_flows, input_edge_flows, mars, edge_mars, edge_aggr, 
                              cbpc, data, parameters, 1:2; mine = 2, maxe = 32)

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


@testset "per-sample flows" begin

    pc = simple_3vars_circuit(; fixed = false)
    init_parameters(pc; perturbation = 0.4)

    cbpc = CuCondBitsProbCircuit(pc)

    data = cu(vcat([UInt8.([1 2 3; 2 3 1]) for i = 1 : 100]...))

    parameters = cu(zeros(Float32, 200, 32))
    parameters[:,3:end] .= log(0.2)
    parameters[:,3:end] .= log(0.2)
    for i = 1 : 200
        parameters[i,1:2] = cu(log.(Float32.([0.2 0.8])))
        parameters[i,3:7] .= cu(log.(Float32.([0.2, 0.3, 0.1, 0.1, 0.3])))
    end

    param_flows = cu(zeros(Float32, 200, 32))

    parameters2 = deepcopy(Array(parameters))

    @test_nowarn per_sample_flows(cbpc, data, parameters, param_flows; batch_size = 32)
    @test_nowarn per_sample_flows(cbpc, data, parameters, param_flows; batch_size = 32, normalize_flows = true)

    @test all(sum(param_flows[:,1:2]; dims = 2) .≈ 1.0)
    @test all(sum(param_flows[:,3:7]; dims = 2) .≈ 1.0)
    @test all(sum(param_flows[:,8:12]; dims = 2) .≈ 1.0)
    @test all(sum(param_flows[:,13:17]; dims = 2) .≈ 1.0)

    @test all(parameters2 .≈ Array(parameters))
end