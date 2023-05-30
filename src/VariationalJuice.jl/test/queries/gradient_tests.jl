using Test
using ProbabilisticCircuits: prep_memory, logsumexp

include("../../src/VariationalJuice.jl")
include("../helpers/simple_circuits.jl")


@testset "gradients" begin

    pc = simple_3vars_circuit(; fixed = false)
    init_parameters(pc; perturbation = 0.4)

    mbpc = CuMetaBitsProbCircuit(pc)

    data = cu(UInt8.([1 2 3; 2 3 1]))

    n_params = num_parameters(mbpc)
    gs = prep_memory(nothing, (2, n_params))

    gradients(mbpc, data, gs; batch_size = 2)

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
        @test gs[1,1] ≈ edge_flows[1,5] / exp(edge_mars[1,5] - pc.params[1])
        @test gs[1,2] ≈ edge_flows[1,6] / exp(edge_mars[1,6] - pc.params[2])

        @test gs[2,1] ≈ edge_flows[2,5] / exp(edge_mars[2,5] - pc.params[1])
        @test gs[2,2] ≈ edge_flows[2,6] / exp(edge_mars[2,6] - pc.params[2])
    end

end