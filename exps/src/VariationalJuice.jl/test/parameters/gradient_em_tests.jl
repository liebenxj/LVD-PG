using Test
using CUDA
using ProbabilisticCircuits

include("../../src/VariationalJuice.jl")
include("../helpers/simple_circuits.jl")


@testset "gradient EM" begin

    pc = simple_3vars_cat_circuit()

    bpc = CuBitsProbCircuit(pc)

    data = cu(UInt8.([1 2 3; 2 3 1; 1 1 4; 2 1 4]))

    lls = mini_batch_gradient_em(bpc, data, 10; batch_size = 4, lr = 0.1)

    @test lls[2] > lls[1]

end
