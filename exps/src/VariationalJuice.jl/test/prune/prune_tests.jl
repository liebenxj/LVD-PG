using Test
using ProbabilisticCircuits: prep_memory, logsumexp, JpcFormat
using Statistics: mean
using MLDatasets

include("../../src/VariationalJuice.jl")


@testset "Pruning" begin
    mnist_train_cpu = collect(transpose(reshape(MNIST(Tx=UInt8, split=:train).features, 28*28, :)));
    mnist_test_cpu = collect(transpose(reshape(MNIST(Tx=UInt8, split=:test).features, 28*28, :)));
    mnist_train_gpu = cu(mnist_train_cpu);
    mnist_test_gpu = cu(mnist_test_cpu);

    trunc_train = cu(mnist_train_cpu .÷ 2^4);
    pc = hclt(trunc_train, 32; num_cats = 256, input_type = Categorical);
    init_parameters(pc; perturbation = 0.4);
    bpc = CuBitsProbCircuit(pc);

    PCs.mini_batch_em(bpc, mnist_train_gpu, 10; batch_size = 512, pseudocount = 0.1, param_inertia = 0.1, param_inertia_end = 0.9);
    update_parameters(bpc);
    
    pc_pruned = prune_pc(pc, mnist_train_gpu; batch_size = 256, keep_frac = 0.1)

    bpc = CuBitsProbCircuit(pc_pruned);
    ll = loglikelihood(bpc, mnist_train_gpu; batch_size = 256)

    @test ll > -720
end