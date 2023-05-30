using Test

include("../src/VariationalJuice.jl")
include("helpers/simple_circuits.jl")


@testset "bits circuit" begin

    pc = simple_3vars_circuit()

    mbpc = CuMetaBitsProbCircuit(pc)

    @test all(Array(mbpc.edge2param) .== UInt32.([0, 0, 0, 0, 1, 2]))
    @test all(Array(mbpc.param2edge) .== UInt32.([5, 6]))
    @test all(Array(mbpc.param2group) .== UInt32.([1, 1]))
    @test all(Array(mbpc.innode2ncumparam) .== UInt32.([0, 0, 0, 0, 0, 0]))
    @test mbpc.num_input_params == 0

    pc = simple_3vars_circuit(; fixed = false)

    mbpc = CuMetaBitsProbCircuit(pc)

    @test all(Array(mbpc.innode2ncumparam) .== UInt32.([5, 10, 15, 20, 25, 30]))
    @test mbpc.num_input_params == 30

end