using Test

include("../../src/VariationalJuice.jl")
include("../helpers/simple_circuits.jl")


@testset "bits circuit" begin

    pc = simple_3vars_circuit()

    cbpc = CuCondBitsProbCircuit(pc)

    @test all(Array(cbpc.edge2param) .== UInt32.([0, 0, 0, 0, 1, 2]))
    @test all(Array(cbpc.param2edge) .== UInt32.([5, 6]))
    @test all(Array(cbpc.param2group) .== UInt32.([1, 1]))
    @test all(Array(cbpc.innode2ncumparam) .== UInt32.([0, 0, 0, 0, 0, 0]))
    @test cbpc.num_input_params == 0

    pc = simple_3vars_circuit(; fixed = false)

    cbpc = CuCondBitsProbCircuit(pc)

    @test all(Array(cbpc.innode2ncumparam) .== UInt32.([5, 10, 15, 20, 25, 30]))
    @test cbpc.num_input_params == 30

end