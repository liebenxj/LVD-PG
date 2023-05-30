using Test
using ProbabilisticCircuits: prep_memory, logsumexp, JpcFormat
using Statistics: mean

include("../../src/VariationalJuice.jl")
include("../helpers/simple_circuits.jl")


@testset "Gumbel-softmax samples" begin

    pc = simple_3vars_circuit(; fixed = false, num_cats = 2)
    init_parameters(pc; perturbation = 0.4)

    mbpc = CuMetaBitsProbCircuit(pc)

    pars = cu(zeros(Float32, 14))
    vectorize_parameters(mbpc, pars)
    parameters = cu(zeros(Float32, 2, 14))
    parameters[1,:] .= pars
    parameters[2,:] .= pars

    n_nodes = num_nodes(mbpc)
    n_edges = num_edges(mbpc)
    td_probs = prep_memory(nothing, (2, n_nodes))
    edge_td_probs = prep_memory(nothing, (2, n_edges))
    edge_probs = prep_memory(nothing, (2, length(mbpc.param2edge)))
    cum_probs = prep_memory(nothing, (2, mbpc.num_param_groups))
    input_aggr_params = prep_memory(nothing, (2, 3, 2))
    temperature = 1.0

    gumbel_sample(td_probs, edge_td_probs, edge_probs, cum_probs, parameters, input_aggr_params,
                  mbpc, temperature, 1:2; mine = 2, maxe = 32)

    CUDA.allowscalar() do
        @test sum(edge_probs[1,:]) ≈ cum_probs[1,1]
        @test edge_probs[1,1] / cum_probs[1,1] ≈ td_probs[1,4]
        @test edge_probs[2,1] / cum_probs[2,1] ≈ td_probs[2,4]
        @test edge_probs[1,2] / cum_probs[1,1] ≈ td_probs[1,8]
        @test edge_probs[2,2] / cum_probs[2,1] ≈ td_probs[2,8]
        for i = 1 : 3
            @test td_probs[1,i] ≈ td_probs[1,4]
            @test td_probs[2,i] ≈ td_probs[2,4]
        end
        for i = 5 : 7
            @test td_probs[1,i] ≈ td_probs[1,8]
            @test td_probs[2,i] ≈ td_probs[2,8]
        end
        for i = 1 : 2
            for j = 1 : 3
                @test sum(input_aggr_params[i, j, :]) ≈ one(Float32)
            end
        end
    end

end

@testset "Gumbel-softmax sample backward" begin

    pc = simple_3vars_circuit(; fixed = false, num_cats = 2)
    init_parameters(pc; perturbation = 0.4)

    mbpc = CuMetaBitsProbCircuit(pc)

    pars = cu(zeros(Float32, 14))
    vectorize_parameters(mbpc, pars)
    parameters = cu(zeros(Float32, 2, 14))
    parameters[1,:] .= pars
    parameters[2,:] .= pars

    n_nodes = num_nodes(mbpc)
    n_edges = num_edges(mbpc)
    td_probs = prep_memory(nothing, (2, n_nodes))
    edge_td_probs = prep_memory(nothing, (2, n_edges))
    edge_probs = prep_memory(nothing, (2, length(mbpc.param2edge)))
    cum_probs = prep_memory(nothing, (2, mbpc.num_param_groups))
    cum_grads = prep_memory(nothing, (2, mbpc.num_param_groups))
    input_aggr_params = prep_memory(nothing, (2, 3, 2))
    temperature = 1.0

    grads = prep_memory(nothing, (2, n_nodes))
    param_grads = prep_memory(nothing, (2, n_edges))
    input_aggr_grads = prep_memory(nothing, (2, 3, 2))

    gumbel_sample(td_probs, edge_td_probs, edge_probs, cum_probs, parameters, input_aggr_params,
                  mbpc, temperature, 1:2; mine = 2, maxe = 32)

    # set gradients
    CUDA.allowscalar() do
        input_aggr_grads[:,:,:] .= zero(Float32)
        input_aggr_grads[1,1,1] = 1.0
        input_aggr_grads[1,1,2] = -1.0
    end

    gumbel_sample_backward(td_probs, grads, param_grads, edge_td_probs, edge_probs, cum_probs, cum_grads, parameters, input_aggr_params,
                           input_aggr_grads, mbpc, temperature, 1:2; mine = 2, maxe = 32)

    CUDA.allowscalar() do
        @test param_grads[1,3] ≈ td_probs[1,1]
        @test param_grads[1,4] ≈ -td_probs[1,1]
        @test param_grads[1,5] ≈ 0.0
        @test param_grads[1,6] ≈ 0.0

        @test grads[1,1] ≈ exp.(parameters)[1,3] * input_aggr_grads[1,1,1] + exp.(parameters)[1,4] * input_aggr_grads[1,1,2]

        @test grads[1,4] ≈ grads[1,1] + grads[1,2] + grads[1,3]
        @test grads[1,8] ≈ grads[1,5] + grads[1,6] + grads[1,7]
        
        @test grads[1,9] ≈ grads[1,4] * edge_probs[1,1] / cum_probs[1,1] + grads[1,8] * edge_probs[1,2] / cum_probs[1,1]

        @test grads[1,8] * td_probs[1,8] + grads[1,4] * td_probs[1,4] ≈ cum_grads[1,1]

        norm_edge_prob1 = edge_probs[1,1] / cum_probs[1,1]
        norm_edge_prob2 = edge_probs[1,2] / cum_probs[1,1]
        @test param_grads[1,1] ≈ (grads[1,4] * td_probs[1,9] - cum_grads[1,1]) * norm_edge_prob1 / temperature
        @test param_grads[1,2] ≈ (grads[1,8] * td_probs[1,9] - cum_grads[1,1]) * norm_edge_prob2 / temperature
    end

end

@testset "Gumbel-softmax new tests" begin

    pc = fully_factorized_categorical_fixed(; num_vars = 3, num_cats = 8);

    mbpc = CuMetaBitsProbCircuit(pc);

    npars = num_parameters(mbpc)
    pars = cu(zeros(Float32, npars))
    vectorize_parameters(mbpc, pars)
    pars = reshape(pars, 1, npars)
    par_buffer_size = param_buffer_size(pc)
    temperature = 1.0

    cat_params, gumbel_reuse = gumbel_sample(mbpc, pars, 3, par_buffer_size; temperature);
    td_probs, edge_td_probs, edge_probs, cum_probs = gumbel_reuse;

    @test td_probs[1,9] ≈ 1.0
    @test td_probs[1,18] ≈ 1.0
    @test td_probs[1,27] ≈ 1.0
    @test cum_probs[1,1] ≈ sum(edge_probs[1,1:8])
    @test cum_probs[1,2] ≈ sum(edge_probs[1,9:16])
    @test cum_probs[1,3] ≈ sum(edge_probs[1,17:24])
    @test td_probs[1,1] ≈ edge_probs[1,1] / cum_probs[1,1]
    @test cat_params[1,1,1] ≈ td_probs[1,1]
    @test cat_params[1,1,2] ≈ td_probs[1,2]
    
end