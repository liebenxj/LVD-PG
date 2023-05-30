using Test
using ProbabilisticCircuits: prep_memory, logsumexp, JpcFormat
using Statistics: mean

include("../../src/VariationalJuice.jl")
include("../helpers/simple_circuits.jl")


@testset "Parameter mapping" begin

    if isfile("../models/mnist_16.jpc")
        pc2 = read("../models/mnist_16.jpc", ProbCircuit, JpcFormat(), true)
        pc1, pc2topc1 = convert_to_latent_pc(pc2; get_mapping = true)

        init_parameters(pc1; perturbation = 0.0)

        mbpc1 = CuMetaBitsProbCircuit(pc1);
        mbpc2 = CuMetaBitsProbCircuit(pc2);

        params1 = prep_memory(nothing, (num_parameters(mbpc1),))
        params2 = prep_memory(nothing, (num_parameters(mbpc2),))
        vectorize_parameters(mbpc1, params1)
        vectorize_parameters(mbpc2, params2)

        params1 = reshape(params1, (1, :))
        params2 = reshape(params2, (1, :))

        updated_params1 = map_pc_parameters(mbpc1, mbpc2, params1, params2; pc2topc1)

        params2 = Array(params2)
        @test updated_params1[1,1] ≈ params2[1,1]
    end

end