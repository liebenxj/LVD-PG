using Test
using CUDA
using JLD
using HDF5
using MLDatasets

include("../../src/VariationalJuice.jl")


@testset "soft reg tests" begin

    pc_file_name = "../artifacts/pc.jpc.gz"
    data_file_name = "../artifacts/patch_data.jld"

    if isfile(pc_file_name) && isfile(data_file_name)

        pc = read(pc_file_name, ProbCircuit)

        train_data = reshape(JLD.load(data_file_name, "trn"), (:, 3*4*4))
        test_data = reshape(JLD.load(data_file_name, "val"), (:, 3*4*4))

        # train_data = cu(collect(transpose(reshape(MNIST.traintensor(UInt8), 28*28, :))))[:,241:280]
        # pc = hclt(train_data .÷ 2^4, 8; num_cats = 256, pseudocount = 0.1, input_type = Categorical)

        num_tr_samples = size(train_data, 1)

        cbpc = CuCondBitsProbCircuit(pc)
        nparams = num_parameters(cbpc)

        batch_size = 128

        n_nodes = num_nodes(cbpc)
        n_edges = num_edges(cbpc)
        input_node_ids = cbpc.bpc.input_node_ids
        mars_mem = prep_memory(nothing, (batch_size, n_nodes), (false, true))
        edge_mars_mem = prep_memory(nothing, (batch_size, n_edges), (false, true))
        flows_mem = prep_memory(nothing, (batch_size, n_nodes), (false, true))
        edge_flows_mem = prep_memory(nothing, (batch_size, n_edges), (false, true))
        input_edge_flows_mem = prep_memory(nothing, (batch_size, cbpc.num_input_params), (false, true))
        edge_groups_mem = prep_memory(nothing, (batch_size, maximum(cbpc.param2group)), (false, true))

        # test controlled EM update
        tr_data = cu(train_data[1:batch_size,:])
        parameters = cu(rand(Float32, 1, nparams))
        log_softmax(cbpc, parameters) # normalize parameters
        update_parameters_to_bpc(cbpc, parameters)

        param_flows = CUDA.zeros(Float32, batch_size, nparams)
        mean_flows = CUDA.zeros(Float32, 1, nparams)
        lls_buffer = CUDA.zeros(Float32, 1)

        lls = Vector{Float32}()

        lr = 0.1
        
        for iter = 1 : 40

            batch_params = repeat(parameters, batch_size, 1)
            per_sample_flows(cbpc, tr_data, batch_params, param_flows; batch_size, normalize_flows = false, 
                             mars_mem, edge_mars_mem, flows_mem, edge_flows_mem, input_edge_flows_mem, edge_groups_mem, 
                             soft_reg = 0.1, soft_reg_width = 3)
            ll_new = sum(mars_mem[:,end]) / batch_size
            push!(lls, ll_new)
            # println(ll_new)
            
            CUDA.sum!(mean_flows, param_flows)

            # normalize all flows
            param2group = cbpc.param2group
            num_groups = maximum(param2group)
            edge_groups_mem1 = prep_memory(nothing, (1, num_groups), (false, true))
            edge_groups_mem1 .= zero(Float32)
            normalize_parameters(cbpc, mean_flows; logspace = false, pseudocount = 0.0, edge_groups_mem = edge_groups_mem1)
            
            parameters .= log.(exp.(parameters) .* Float32(one(Float32) - lr) .+ mean_flows .* Float32(lr))
        end

        @test lls[end] > -170.0
    end

end