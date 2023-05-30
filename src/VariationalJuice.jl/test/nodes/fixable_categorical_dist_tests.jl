using Test

include("../../src/VariationalJuice.jl")
include("../helpers/simple_circuits.jl")


@testset "fixable categorical dist" begin

    n1 = PCs.PlainInputNode(1, FixableCategorical(10, true))
    n2 = PCs.PlainInputNode(1, FixableCategorical(10, false))

    d1 = dist(n1)
    d2 = dist(n2)

    # @test num_parameters_node(n1, false) == 0
    @test num_parameters_node(n2, false) == 10

    @test all(init_params(d1, 0.2).logps .== d1.logps)

    heap = zeros(Float32, 0)

    bd1 = bits(d1, heap)
    bd2 = bits(d2, heap)

    @test num_parameters(bd1) == 0
    @test num_parameters(bd2) == 10

    @test d1.logps[1] ≈ heap[bd1.heap_start]
    @test d2.logps[1] ≈ heap[bd2.heap_start]

end