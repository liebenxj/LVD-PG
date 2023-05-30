using Test
using CUDA

include("../../src/VariationalJuice.jl")
include("../helpers/simple_circuits.jl")


@testset "log_softmax & its gradient test" begin

    pc = simple_3vars_circuit()

    cbpc = CuCondBitsProbCircuit(pc)

    parameters = cu(zeros(Float32, 2, 2))
    CUDA.allowscalar() do 
        parameters[1,1] = 2.0
        parameters[1,2] = 4.0
        parameters[2,1] = 8.0
        parameters[2,2] = 2.0
    end

    log_softmax(cbpc, parameters)

    @test all(sum(exp.(parameters); dims = 2) .≈ 1.0)

    grads = (cu(zeros(Float32, 2, 2)))
    CUDA.allowscalar() do 
        grads[1,1] = 1.0
        grads[1,2] = -2.0
        grads[2,1] = 2.0
        grads[2,2] = -3.0
    end

    log_softmax_grad(cbpc, parameters, grads)

    @test all(grads .≈ cu(Float32[1.119203 -1.1192029; 2.9975274 -2.9975274]))

    @test check_params_normalized(cbpc, parameters; logspace = true, edge_only = true)
end

@testset "check_params_normalized" begin

    # pc = simple_3vars_cat_circuit()
    pc = simple_multihead_circuit()[1]

    cbpc = CuCondBitsProbCircuit(pc)

    nparams = num_parameters(cbpc)
    parameters = cu(rand(Float32, 2, nparams))

    log_softmax(cbpc, parameters)

    @test check_params_normalized(cbpc, parameters; logspace = true, edge_only = false)

    parameters .= exp.(parameters)

    @test check_params_normalized(cbpc, parameters; logspace = false, edge_only = false)
    
end