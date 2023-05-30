using Test
using ProbabilisticCircuits: prep_memory, logsumexp, JpcFormat

include("../../src/VariationalJuice.jl")
include("../helpers/simple_circuits.jl")


@testset "KLDs" begin

    pc = simple_3vars_circuit(; fixed = false)
    init_parameters(pc; perturbation = 0.4)

    mbpc = CuMetaBitsProbCircuit(pc)

    pars = cu(zeros(Float32, 32))
    vectorize_parameters(mbpc, pars)
    parameters1 = cu(zeros(Float32, 2, 32))
    parameters1[1,:] .= pars
    parameters1[2,:] .= pars
    parameters2 = cu(zeros(Float32, 2, 32))
    parameters2[1,:] .= pars
    parameters2[2,:] .= pars

    n_nodes = num_nodes(mbpc)
    n_edges = num_edges(mbpc)
    klds = prep_memory(nothing, (2, n_nodes))
    edge_klds = prep_memory(nothing, (2, n_edges))

    kld(klds, edge_klds, mbpc, parameters1, parameters2, 1:2; mine = 2, maxe = 32)

    @test all(klds .≈ zero(Float32))
    @test all(edge_klds .≈ zero(Float32))

    parameters1[1,:] .= pars
    parameters1[2,:] .= pars
    parameters1[1:2,1:2] = cu(log.(Float32.([0.2 0.8; 0.5 0.5])))
    
    parameters2[1,:] .= pars
    parameters2[2,:] .= pars
    parameters2[1:2,1:2] = cu(log.(Float32.([0.4 0.6; 0.9 0.1])))

    kld(klds, edge_klds, mbpc, parameters1, parameters2, 1:2; mine = 2, maxe = 32)

    CUDA.allowscalar() do
        @test klds[1,end] ≈ 0.2 * log(0.2 / 0.4) + 0.8 * log(0.8 / 0.6)
        @test klds[2,end] ≈ 0.5 * log(0.5 / 0.9) + 0.5 * log(0.5 / 0.1)
    end

    parameters1[1,:] .= pars
    parameters1[2,:] .= pars
    parameters1[1:2,1:2] = cu(log.(Float32.([0.2 0.8; 0.5 0.5])))
    
    parameters2[1,:] .= pars
    parameters2[2,:] .= pars
    parameters2[1:2,1:2] = cu(log.(Float32.([0.2 0.8; 0.5 0.5])))

    parameters1[1,3:7] = cu(log.(Float32.([0.1, 0.2, 0.3, 0.2, 0.2])))
    parameters2[1,3:7] = cu(log.(Float32.([0.2, 0.1, 0.1, 0.3, 0.3])))

    parameters1[2,3:7] = cu(log.(Float32.([0.1, 0.2, 0.3, 0.2, 0.2])))
    parameters2[2,3:7] = cu(log.(Float32.([0.2, 0.1, 0.2, 0.3, 0.2])))

    kld(klds, edge_klds, mbpc, parameters1, parameters2, 1:2; mine = 2, maxe = 32)

    CUDA.allowscalar() do
        @test klds[1,end] ≈ sum(exp.(parameters1[1,3:7]) .* (parameters1[1,3:7] .- parameters2[1,3:7])) * 0.2
        @test klds[2,end] ≈ sum(exp.(parameters1[2,3:7]) .* (parameters1[2,3:7] .- parameters2[2,3:7])) * 0.5
    end

end

@testset "KLDs backward" begin

    pc = simple_3vars_circuit(; fixed = false)
    init_parameters(pc; perturbation = 0.4)

    mbpc = CuMetaBitsProbCircuit(pc)

    pars = cu(zeros(Float32, 32))
    vectorize_parameters(mbpc, pars)
    parameters1 = cu(zeros(Float32, 2, 32))
    parameters1[1,:] .= pars
    parameters1[2,:] .= pars
    parameters2 = cu(zeros(Float32, 2, 32))
    parameters2[1,:] .= pars
    parameters2[2,:] .= pars

    n_nodes = num_nodes(mbpc)
    n_edges = num_edges(mbpc)
    klds = prep_memory(nothing, (2, n_nodes))
    edge_klds = prep_memory(nothing, (2, n_edges))
    grads = prep_memory(nothing, (2, n_nodes))
    edge_grads1 = prep_memory(nothing, (2, n_edges))
    edge_grads2 = prep_memory(nothing, (2, n_edges))
    input_edge_grads1 = prep_memory(nothing, (2, mbpc.num_input_params))
    input_edge_grads2 = prep_memory(nothing, (2, mbpc.num_input_params))

    kld_forward_backward(klds, edge_klds, grads, edge_grads1, edge_grads2, 
                         input_edge_grads1, input_edge_grads2, mbpc,
                         parameters1, parameters2, 1:2; mine = 2, maxe = 32)

    CUDA.allowscalar() do
        @test edge_grads1[1,5] ≈ exp(parameters1[1,1])
        @test edge_grads1[1,6] ≈ exp(parameters1[1,2])
        @test edge_grads1[1,5] ≈ -edge_grads2[1,5]
        @test edge_grads1[1,6] ≈ -edge_grads2[1,6]
        @test all(edge_grads1[:, 1:4] .≈ 0.0)
        @test all(edge_grads2[:, 1:4] .≈ 0.0)
    end

    parameters1[1,:] .= pars
    parameters1[2,:] .= pars
    parameters1[1:2,1:2] = cu(log.(Float32.([0.2 0.8; 0.5 0.5])))
    
    parameters2[1,:] .= pars
    parameters2[2,:] .= pars
    parameters2[1:2,1:2] = cu(log.(Float32.([0.4 0.6; 0.9 0.1])))

    kld_forward_backward(klds, edge_klds, grads, edge_grads1, edge_grads2, 
                         input_edge_grads1, input_edge_grads2, mbpc,
                         parameters1, parameters2, 1:2; mine = 2, maxe = 32)

    CUDA.allowscalar() do # gradients w.r.t. log-parameters
        @test edge_grads1[1, 5] ≈ 0.2 * (log(0.2) - log(0.4) + 1)
        @test edge_grads2[1, 5] ≈ -0.2
        @test edge_grads1[1, 6] ≈ 0.8 * (log(0.8) - log(0.6) + 1)
        @test edge_grads2[1, 6] ≈ -0.8
    end

end

@testset "KLD gaussian" begin

    pc = simple_gaussian_circuit()
    init_parameters(pc; perturbation = 0.4)

    mbpc = CuMetaBitsProbCircuit(pc)

    pars = cu(zeros(Float32, 14))
    vectorize_parameters(mbpc, pars)
    parameters1 = cu(zeros(Float32, 2, 14))
    parameters1[1,:] .= pars
    parameters1[2,:] .= pars
    parameters2 = cu(zeros(Float32, 2, 14))
    parameters2[1,:] .= pars
    parameters2[2,:] .= pars

    n_nodes = num_nodes(mbpc)
    n_edges = num_edges(mbpc)
    klds = prep_memory(nothing, (2, n_nodes))
    edge_klds = prep_memory(nothing, (2, n_edges))
    grads = prep_memory(nothing, (2, n_nodes))
    edge_grads1 = prep_memory(nothing, (2, n_edges))
    edge_grads2 = prep_memory(nothing, (2, n_edges))
    input_edge_grads1 = prep_memory(nothing, (2, mbpc.num_input_params))
    input_edge_grads2 = prep_memory(nothing, (2, mbpc.num_input_params))

    kld(klds, edge_klds, mbpc, parameters1, parameters2, 1:2; mine = 2, maxe = 32, debug = false)
    
    @test all(klds .≈ 0.0)

    kld_forward_backward(klds, edge_klds, grads, edge_grads1, edge_grads2, 
                         input_edge_grads1, input_edge_grads2, mbpc,
                         parameters1, parameters2, 1:2; mine = 2, maxe = 32)
    
end

@testset "KLD inner nodes tests" begin

    pc = fully_factorized_categorical_fixed(; num_vars = 2, num_cats = 8)

    mbpc = CuMetaBitsProbCircuit(pc);

    num_pars = num_parameters(mbpc)
    pars = cu(zeros(Float32, num_pars))
    vectorize_parameters(mbpc, pars)

    pars1 = cu(zeros(Float32, 1, num_pars))
    pars1[1,:] .= pars
    pars1[1,1:8] .= cu(rand(Float32, 8))
    pars1[1,1:8] ./= sum(pars1[1,1:8])
    pars1[1,9:16] .= cu(rand(Float32, 8))
    pars1[1,9:16] ./= sum(pars1[1,9:16])
    pars1[1,1:16] .= log.(pars1[1,1:16])

    pars2 = cu(zeros(Float32, 1, num_pars))
    pars2[1,:] .= pars

    klds, kld_reuse = kld(mbpc, pars1, pars2);
    grad1, grad2, grad_reuse = kld_backward(mbpc, pars1, pars2; reuse = kld_reuse, get_reuse_grad = true);

    klds, edge_klds = kld_reuse;
    grads, edge_grads1, edge_grads2, input_edge_grads1, input_edge_grads2 = grad_reuse;

    @test klds[1,9] ≈ sum(exp.(pars1[1:8]) .* (pars1[1:8] .- pars2[1:8]))
    @test klds[1,18] ≈ sum(exp.(pars1[9:16]) .* (pars1[9:16] .- pars2[9:16]))

    CUDA.allowscalar() do 
        for i = 1 : 8
            @test exp(pars1[1,i]) ≈ grads[1,i]
            @test edge_grads1[1,i] ≈ exp(pars1[1,i]) * (pars1[1,i] - pars2[1,i] + 1)
        end
        for i = 9 : 16
            @test exp(pars1[1,i]) ≈ grads[1,i+1]
            @test edge_grads1[1,i] ≈ exp(pars1[1,i]) * (pars1[1,i] - pars2[1,i] + 1)
        end
    end

end

@testset "KLD with target" begin

    if isfile("../../../exps/pc-vae/pretrain_decoder/models/mnist_4.jpc")
        for _ = 1 : 10
            pc = read("../../../exps/pc-vae/pretrain_decoder/models/mnist_4.jpc", ProbCircuit, JpcFormat(), true)
            pc = convert_to_latent_pc(pc)

            mbpc = CuMetaBitsProbCircuit(pc);
            num_pars = num_parameters(mbpc)
            pars2 = cu(zeros(Float32, num_pars))
            vectorize_parameters(mbpc, pars2)

            init_parameters(pc; perturbation = 0.5)
            mbpc = CuMetaBitsProbCircuit(pc);
            pars1 = cu(zeros(Float32, num_pars))
            vectorize_parameters(mbpc, pars1)

            pars1 = reshape(pars1, (1, :))
            pars2 = reshape(pars2, (1, :))

            newton_step_size = 0.01
            newton_nsteps = 4

            kld_old, new_params, _ = kld_with_update_target(mbpc, pars1, pars2; newton_step_size, newton_nsteps)

            kld_new, _ = kld(mbpc, new_params, pars2)

            mid_params = log.(0.9 .* exp.(pars1) .+ 0.1 .* exp.(pars2))
            kld_mid, _ = kld(mbpc, mid_params, pars2)

            @test kld_new[1] < kld_old[1]
            @test kld_mid[1] < kld_old[1]
        end
    end

end