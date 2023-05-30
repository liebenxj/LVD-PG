using Test
using CUDA
using JLD
using HDF5
using MLDatasets

include("../../src/VariationalJuice.jl")


@testset "custom EM tests" begin

    # pc_file_name = "../artifacts/patch.jpc.gz"
    # pc_file_name = "../artifacts/pc-nn.jpc.gz"
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

        parameters = cu(rand(Float32, 1, nparams))
        log_softmax(cbpc, parameters) # normalize parameters

        batch_size = 128
        lr = 0.1

        param_flows = CUDA.zeros(Float32, batch_size, nparams)
        mean_flows = CUDA.zeros(Float32, 1, nparams)
        lls_buffer = CUDA.zeros(Float32, 1)

        n_nodes = num_nodes(cbpc)
        n_edges = num_edges(cbpc)
        input_node_ids = cbpc.bpc.input_node_ids
        mars_mem = prep_memory(nothing, (batch_size, n_nodes), (false, true))
        edge_mars_mem = prep_memory(nothing, (batch_size, n_edges), (false, true))
        flows_mem = prep_memory(nothing, (batch_size, n_nodes), (false, true))
        edge_flows_mem = prep_memory(nothing, (batch_size, n_edges), (false, true))
        input_edge_flows_mem = prep_memory(nothing, (batch_size, cbpc.num_input_params), (false, true))
        edge_groups_mem = prep_memory(nothing, (batch_size, maximum(cbpc.param2group)), (false, true))

        update_parameters_to_bpc(cbpc, parameters)

        CUDA.allowscalar(true)
        
        for _ = 1 : 5
            m = cu(randperm!(collect(1:batch_size)))
            tr_data = cu(train_data[m,:])

            # test likelihood
            lls1 = loglikelihoods(cbpc, tr_data, repeat(parameters, batch_size, 1); batch_size)
            lls2 = PCs.loglikelihoods(cbpc.bpc, tr_data; batch_size)
            @test all(Array(lls1) .≈ Array(lls2))

            # test flows
            edge_aggr_mem1 = prep_memory(nothing, (n_edges,))
            conditional_circuit_flows(flows_mem, edge_flows_mem, input_edge_flows_mem, mars_mem, edge_mars_mem, edge_aggr_mem1, 
                                    cbpc, tr_data, repeat(parameters, batch_size, 1), 1:batch_size; mine = 2, maxe = 32)
            mars1 = deepcopy(Array(mars_mem))
            flows1 = deepcopy(Array(flows_mem))
            eflows1 = deepcopy(Array(edge_flows_mem))
            ceaggr = deepcopy(Array(edge_aggr_mem1))
            input_edge_flows1 = deepcopy(Array(input_edge_flows_mem))
            PCs.clear_input_node_mem(cbpc.bpc)
            PCs.probs_flows_circuit(flows_mem, mars_mem, nothing, cbpc.bpc, tr_data, 1:batch_size; mine = 2, maxe = 32)
            mars2 = deepcopy(Array(mars_mem))
            flows2 = deepcopy(Array(flows_mem))
            @test all(mars1 .≈ mars2)
            @test all(isapprox.(flows1, flows2; atol = 1e-3))

            # test edge_aggr
            edge_aggr_mem = prep_memory(nothing, (n_edges,))
            edge_aggr_mem .= zero(Float32)
            PCs.clear_input_node_mem(cbpc.bpc)
            PCs.probs_flows_circuit(flows_mem, mars_mem, edge_aggr_mem, cbpc.bpc, tr_data, 1:batch_size; mine = 2, maxe = 32)
            eaggr = deepcopy(Array(edge_aggr_mem))
            down2upedge = Array(cbpc.bpc.down2upedge)
            aggregated_eflow = sum(eflows1; dims = 1)
            @test all(eaggr .≈ aggregated_eflow[down2upedge[collect(1:n_edges)]])
            @test all(ceaggr .≈ eaggr)

            # test input_edge_aggr
            innode2ncumparam = Array(cbpc.innode2ncumparam)
            input_node_ids = Array(cbpc.bpc.input_node_ids)
            heap = Array(cbpc.bpc.heap)
            nodes = Array(cbpc.bpc.nodes)
            flag = true
            for i = 1 : length(input_node_ids)
                node = nodes[input_node_ids[i]]
                d = dist(node)
                iflows1 = heap[d.heap_start+d.num_cats:d.heap_start+2*d.num_cats-1]
                iflows2 = sum(input_edge_flows1[:,innode2ncumparam[i]-d.num_cats+1:innode2ncumparam[i]]; dims = 1)[1,:]
                flag = flag && all(iflows1 .≈ iflows2)
            end
            @test flag
        end

        # lls = PCs.mini_batch_em(cbpc.bpc, cu(train_data[1:batch_size,:]), 10; batch_size, pseudocount = 0.1, param_inertia = 0.9)

        # test controlled EM update
        tr_data = cu(train_data[1:batch_size,:])
        parameters = cu(rand(Float32, 1, nparams))
        log_softmax(cbpc, parameters) # normalize parameters
        update_parameters_to_bpc(cbpc, parameters)
        node_aggr_mem = prep_memory(nothing, (n_nodes,))
        edge_aggr_mem = prep_memory(nothing, (n_edges,))
        
        for iter = 1 : 40
            edge_aggr_mem .= zero(Float32)
            PCs.clear_input_node_mem(cbpc.bpc)
            PCs.probs_flows_circuit(flows_mem, mars_mem, edge_aggr_mem, cbpc.bpc, tr_data, 1:batch_size; mine = 2, maxe = 32)
            good_eaggr = deepcopy(Array(edge_aggr_mem))
            ll_origin = sum(mars_mem[:,end]) / batch_size
            # println("origin ", ll_origin)

            PCs.add_pseudocount(edge_aggr_mem, node_aggr_mem, cbpc.bpc, 0.0; debug = false)
            good_eaggr2 = deepcopy(Array(edge_aggr_mem))
            PCs.aggr_node_flows(node_aggr_mem, cbpc.bpc, edge_aggr_mem; debug = false)
            good_naggr = deepcopy(Array(node_aggr_mem))
            PCs.update_params(cbpc.bpc, node_aggr_mem, edge_aggr_mem; inertia = one(Float32) - lr, debug = false)

            PCs.update_input_node_params(cbpc.bpc; pseudocount = 0.0, inertia = one(Float32) - lr, debug = false)

            batch_params = repeat(parameters, batch_size, 1)
            per_sample_flows(cbpc, tr_data, batch_params, param_flows; batch_size, normalize_flows = false, 
                            mars_mem, edge_mars_mem, flows_mem, edge_flows_mem, input_edge_flows_mem, edge_groups_mem)
            ll_new = sum(mars_mem[:,end]) / batch_size
            # println("new ", ll_new)
            @test abs(ll_origin - ll_new) < 1.0
            
            CUDA.sum!(mean_flows, param_flows)

            edge2param = Array(cbpc.edge2param)
            down2upedge = Array(cbpc.bpc.down2upedge)
            idxs = edge2param[down2upedge[collect(1:n_edges)]]
            flag = true
            for j = 1 : length(idxs)
                if idxs[j] != 0
                    flag = flag && isapprox(mean_flows[idxs[j]], good_eaggr[j]; atol = 1e-3)
                end
            end
            if iter < 10
                @test flag
            end

            innode2ncumparam = Array(cbpc.innode2ncumparam)
            input_node_ids = Array(cbpc.bpc.input_node_ids)
            heap = Array(cbpc.bpc.heap)
            nodes = Array(cbpc.bpc.nodes)
            n_inner_params = length(cbpc.param2edge)
            flag = true
            for i = 1 : length(input_node_ids)
                node = nodes[input_node_ids[i]]
                d = dist(node)
                iflows1 = heap[d.heap_start+d.num_cats:d.heap_start+2*d.num_cats-1]
                iflows2 = Array(mean_flows[1,n_inner_params+innode2ncumparam[i]-d.num_cats+1:n_inner_params+innode2ncumparam[i]])
                flag = flag && all(isapprox.(iflows1, iflows2; atol = 1e-3))
            end
            if iter < 10
                @test flag
            end

            # normalize all flows
            param2group = cbpc.param2group
            num_groups = maximum(param2group)
            edge_groups_mem1 = prep_memory(nothing, (1, num_groups), (false, true))
            edge_groups_mem1 .= zero(Float32)
            backup_mean_flows = deepcopy(Array(mean_flows))
            normalize_parameters(cbpc, mean_flows; logspace = false, pseudocount = 0.0, edge_groups_mem = edge_groups_mem1)
            bad_eaggr = deepcopy(Array(edge_groups_mem1))[1,:]
            
            #=if !check_params_normalized(cbpc, mean_flows)
                using NPZ
                npzwrite("./1.npz", Dict("mean_flows" => Array(mean_flows), "backup_mean_flows" => backup_mean_flows))
                print("><<<<<<<<<<<<<<<<<<<")
            end=#

            edges_down = Array(cbpc.bpc.edge_layers_down.vectors)
            param2group = Array(cbpc.param2group)
            flag = true
            for j = 1 : length(idxs)
                if idxs[j] != 0
                    par_id = edges_down[j].parent_id
                    ff1 = (good_naggr[par_id] ≈ 0.0 && bad_eaggr[param2group[idxs[j]]] ≈ 0.0) || isapprox(bad_eaggr[param2group[idxs[j]]] / good_naggr[par_id], 1.0; atol = 1e-3)
                    ff2 = isapprox(mean_flows[idxs[j]], good_eaggr2[j] / good_naggr[par_id]; atol = 1e-3)
                    if !ff1 || !ff2
                        println(j, " ", ff1, " ", ff2, " ", bad_eaggr[param2group[idxs[j]]] / good_naggr[par_id])
                    end
                    flag = flag && ff1 && ff2
                end
            end
            if iter < 10
                @test flag
            end
            
            parameters .= log.(exp.(parameters) .* Float32(one(Float32) - lr) .+ mean_flows .* Float32(lr))
        end

    end
end