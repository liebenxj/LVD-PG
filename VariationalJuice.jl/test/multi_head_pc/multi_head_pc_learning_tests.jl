using Test
using ProbabilisticCircuits: prep_memory, logsumexp, JpcFormat
using Statistics: mean
using MLDatasets

include("../../src/VariationalJuice.jl")
include("../helpers/simple_circuits.jl")


@testset "Multi-head PC likelihood & flow" begin
    pcs = simple_multihead_circuit(; num_cats = 5);
    pcs[1].inputs[1].inputs[1].dist.logps .= log.([0.1,0.2,0.3,0.3,0.1]);
    pcs[1].params .= log.([0.2,0.8]);
    pcs[2].params .= log.([0.6,0.4]);

    mhbpc = CuMultiHeadBitsProbCircuit(pcs);

    data = cu([0 0 0; 2 0 0]);
    head_mask = cu(Float32.([1.0 0.0; 0.0 1.0]));

    lls = multihead_loglikelihoods_probcat(mhbpc, data, head_mask; batch_size = 2)

    CUDA.allowscalar() do 
        @test lls[1] ≈ log(0.1*0.2*0.2*0.2+0.2^3*0.8)
        @test lls[2] ≈ log(0.3*0.2*0.2*0.6+0.2^3*0.4)
    end

    flows = prep_memory(nothing, (2, length(mhbpc.bpc.nodes)), (false, true));
    mars = prep_memory(nothing, (2, length(mhbpc.bpc.nodes)), (false, true));
    edge_aggr = prep_memory(nothing, (length(mhbpc.bpc.edge_layers_up.vectors),));
    example_ids = 1 : 2;

    multi_head_probs_flows_circuit(flows, mars, edge_aggr, mhbpc, data, head_mask, example_ids; 
                                   mine = 2, maxe = 32, soft_reg = 0.01, soft_reg_width = 3)

    CUDA.allowscalar() do
        @test flows[1,9] ≈ 1.0
        @test flows[1,10] ≈ 0.0
        @test all(flows[:,11] .≈ 0.0)
        @test flows[2,9] ≈ 0.0
        @test flows[2,10] ≈ 1.0

        @test flows[1,4] ≈ exp(mars[1,1] + mars[1,2] + mars[1,3] + log(Float32(0.2)) - mars[1,9])
        @test flows[1,8] ≈ exp(mars[1,5] + mars[1,6] + mars[1,7] + log(Float32(0.8)) - mars[1,9])
        @test flows[2,4] ≈ exp(mars[2,1] + mars[2,2] + mars[2,3] + log(Float32(0.6)) - mars[2,10])
        @test flows[2,8] ≈ exp(mars[2,5] + mars[2,6] + mars[2,7] + log(Float32(0.4)) - mars[2,10])
    end
end