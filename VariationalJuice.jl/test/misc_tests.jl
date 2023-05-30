using Test
using CUDA

include("../src/VariationalJuice.jl")
include("helpers/simple_circuits.jl")


@testset "parameter transfer and vectorization" begin

    pc = simple_3vars_circuit()

    mbpc = CuMetaBitsProbCircuit(pc)

    @test num_parameters(mbpc) == 2

    parameters = cu(zeros(Float32, 2))
    vectorize_parameters(mbpc, parameters)

    @test all(Array(parameters) .== Float32.([log(0.5), log(0.5)]))

    CUDA.allowscalar() do
        parameters[1] = log(0.4)
        parameters[2] = log(0.6)

        update_parameters(mbpc, parameters)

        @test mbpc.bpc.edge_layers_up[5].logp ≈ log(0.4)
        @test mbpc.bpc.edge_layers_up[6].logp ≈ log(0.6)
        @test mbpc.bpc.edge_layers_down[1].logp ≈ log(0.4)
        @test mbpc.bpc.edge_layers_down[2].logp ≈ log(0.6)
    end


    pc = simple_3vars_circuit(; fixed = false)
    init_parameters(pc; perturbation = 0.2)

    mbpc = CuMetaBitsProbCircuit(pc)

    @test num_parameters(mbpc) == 32

    parameters = cu(zeros(Float32, 32))
    vectorize_parameters(mbpc, parameters)

    CUDA.allowscalar() do
        @test parameters[1] ≈ mbpc.bpc.edge_layers_up[5].logp
        @test parameters[3] ≈ pc.inputs[1].inputs[1].dist.logps[1]
        @test parameters[18] ≈ pc.inputs[2].inputs[1].dist.logps[1]
    end
    
    CUDA.allowscalar() do
        parameters[1] = 0.0
        parameters[3] = 0.1
        parameters[18] = 0.2

        update_parameters(mbpc, parameters)

        @test mbpc.bpc.edge_layers_up[5].logp ≈ 0.0
        n = mbpc.bpc.nodes[mbpc.bpc.input_node_ids[1]]
        @test mbpc.bpc.heap[n.dist.heap_start] ≈ 0.1
        n = mbpc.bpc.nodes[mbpc.bpc.input_node_ids[4]]
        @test mbpc.bpc.heap[n.dist.heap_start] ≈ 0.2
    end

end

@testset "normalize parameters test" begin

    pc = simple_3vars_circuit()

    mbpc = CuMetaBitsProbCircuit(pc)

    parameters = cu(zeros(Float32, 2, 2))
    CUDA.allowscalar() do 
        parameters[1,1] = 2.0
        parameters[1,2] = 4.0
        parameters[2,1] = 8.0
        parameters[2,2] = 2.0
    end

    normalize_parameters(mbpc, parameters; is_log_params = false)

    CUDA.allowscalar() do 
        @test exp(parameters[1,1]) ≈ 0.333333
        @test exp(parameters[1,2]) ≈ 0.666667
        @test exp(parameters[2,1]) ≈ 0.8
        @test exp(parameters[2,2]) ≈ 0.2
    end

    pc = simple_3vars_circuit(; fixed = false)
    
    mbpc = CuMetaBitsProbCircuit(pc)

    parameters = cu(zeros(Float32, 32))
    vectorize_parameters(mbpc, parameters)
    parameters = reshape(parameters, 1, 32)
    CUDA.allowscalar() do 
        parameters[1,3] = log(2.0)
        parameters[1,4] = log(4.0)
        parameters[1,5] = log(2.0)
        parameters[1,6] = log(4.0)
        parameters[1,7] = log(2.0)
    end

    normalize_parameters(mbpc, parameters; is_log_params = true)

    CUDA.allowscalar() do
        @test exp(parameters[1,3]) ≈ 0.14285715
        @test exp(parameters[1,4]) ≈ 0.2857143
        @test exp(parameters[1,5]) ≈ 0.14285715
        @test exp(parameters[1,6]) ≈ 0.2857143
        @test exp(parameters[1,7]) ≈ 0.14285715
    end
end