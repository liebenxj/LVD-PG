using Test
using ProbabilisticCircuits: prep_memory, logsumexp, JpcFormat
using Statistics: mean
using MLDatasets

include("../../src/VariationalJuice.jl")


@testset "Layered bit circuit tests" begin
    mnist_train_cpu = collect(transpose(reshape(MNIST(Tx=UInt8, split=:train).features, 28*28, :)));
    mnist_test_cpu = collect(transpose(reshape(MNIST(Tx=UInt8, split=:test).features, 28*28, :)));
    mnist_train_gpu = cu(mnist_train_cpu);
    mnist_test_gpu = cu(mnist_test_cpu);

    trunc_train = cu(mnist_train_cpu .÷ 2^4);
    pc = hclt(trunc_train, 4; num_cats = 256, input_type = Categorical);
    init_parameters_by_logits(pc; mval = 2.0);
    
    lbpc = CuLayeredBitsProbCircuit(pc; min_group_nvars = 5);

    warmup_with_mini_batch_em(lbpc, mnist_train_gpu; batch_size = 256, num_epochs_per_layer = 10,
                              pseudocount = 0.1, soft_reg = 0.0, soft_reg_width = 7, 
                              param_inertia = 0.1, param_inertia_end = 0.9, layer_interval = 100)

    mini_batch_em(lbpc.bpc, mnist_train_gpu, 10; batch_size = 256, pseudocount = 0.1, param_inertia = 0.1, param_inertia_end = 0.9)
end
