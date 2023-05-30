using Test
using ProbabilisticCircuits: prep_memory, logsumexp, JpcFormat

include("../../src/VariationalJuice.jl")


@testset "conditioned td prob tests" begin

    nc = 1
    height = 16
    width = 16

    num_hidden_cats = 3

    data = rand(0:9, 128, nc * height * width);

    pc, leaves = hclt_for_colored_images(cu(data), nc, height, width, num_hidden_cats; num_cats = 10, pseudocount = 0.1, 
                                         mode = "red_only", patch_size = 4, input_type = Categorical);
    init_parameters(pc; perturbation = 0.4)

    pc2 = bind_pc(leaves)

    mbpc = CuMetaBitsProbCircuit(pc2);

    marked_pc_idx1, marked_pc_idx2 = mark_nodes(mbpc, leaves);
    marked_pc_idx1 = cu(marked_pc_idx1);
    marked_pc_idx2 = cu(marked_pc_idx2);

    cat_params = rand(Float32, 3, size(leaves, 1), size(leaves, 2));
    cat_params ./= sum(cat_params; dims = 3);
    cat_params = cu(cat_params);

    input_aggr_params = CUDA.zeros(Float32, size(cat_params, 1), nc * height * width, 10);

    td_probs_mem = prep_memory(nothing, (3, length(mbpc.bpc.nodes)))

    conditioned_td_prob(mbpc, marked_pc_idx1, marked_pc_idx2, cat_params,
                        input_aggr_params; td_probs_mem)

    @test all((sum(input_aggr_params; dims = 3) .- 1.0) .< 1e-6)

    cat_params = Array(cat_params);
    input_aggr_params = Array(input_aggr_params);

    for sample_idx = 1 : size(cat_params, 1)
        for var_idx = 1 : nc * height * width
            probs = zeros(Float32, 10)
            target_idx = 0
            idxj = 0
            for i = 1 : size(cat_params, 2)
                if var_idx in randvars(leaves[i,1])
                    target_idx = i
                    n = leaves[i,1].inputs[1]
                    for j = 1 : num_inputs(n)
                        if var_idx in randvars(n.inputs[j])
                            idxj = j
                            break
                        end
                    end
                    break
                end
            end

            for k = 1 : size(cat_params, 3)
                probs .+= cat_params[sample_idx, target_idx, k] .* exp.(leaves[target_idx, k].inputs[1].inputs[idxj].dist.logps)
            end

            @test all(input_aggr_params[sample_idx, var_idx, :] .≈ probs)
        end
    end

end