using Test
using ProbabilisticCircuits: prep_memory, logsumexp, JpcFormat
using Statistics: mean
using MLDatasets

include("../../src/VariationalJuice.jl")


@testset "Adam" begin
    mnist_train_cpu = collect(transpose(reshape(MNIST(Tx=UInt8, split=:train).features, 28*28, :)));
    mnist_test_cpu = collect(transpose(reshape(MNIST(Tx=UInt8, split=:test).features, 28*28, :)));
    mnist_train_gpu = cu(mnist_train_cpu);
    mnist_test_gpu = cu(mnist_test_cpu);

    trunc_train = cu(mnist_train_cpu .÷ 2^4);
    pc = hclt(trunc_train, 32; num_cats = 256, input_type = Categorical);
    init_parameters(pc; perturbation = 0.4);
    bpc = CuBitsProbCircuit(pc);

    PCs.mini_batch_em(bpc, mnist_train_gpu, 10; batch_size = 512, pseudocount = 0.1, param_inertia = 0.1, param_inertia_end = 0.9)

    lls = adam(bpc, mnist_train_gpu, 10; batch_size = 256, lr = 1e-3)

    @test lls[end] > -820

    # loglikelihood(bpc, mnist_test_gpu; batch_size = 512)
end