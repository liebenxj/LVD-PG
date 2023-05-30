using PyCall
using NPZ
using Printf
using Statistics: mean
using Pickle
using ChowLiuTrees: topk_MST
using ProbabilisticCircuits: clt_edges2graphs

include("../../../VariationalJuice.jl/src-jl/LatentPCs.jl")
include("../../../VariationalJuice.jl/src/VariationalJuice.jl")

push!(PyVector(pyimport("sys")["path"]), "../")
push!(PyVector(pyimport("sys")["path"]), "./src")


np = pyimport("numpy")


py"""
from subsample import subsample_data_loader
"""


function rgb2ycrcb(imgs)
    new_imgs = zeros(UInt8, size(imgs)...)
    new_imgs[:,1,:,:] .= UInt8.(round.(clamp.(imgs[:,1,:,:] .* 0.2126 .+ imgs[:,2,:,:] .* 0.7152 .+ imgs[:,3,:,:] .* 0.0722, 0, 255)))
    new_imgs[:,2,:,:] .= UInt8.(round.(clamp.(imgs[:,1,:,:] .* -0.1146 .+ imgs[:,2,:,:] .* -0.3854 .+ imgs[:,3,:,:] .* 0.5 .+ 128, 0, 255)))
    new_imgs[:,3,:,:] .= UInt8.(round.(clamp.(imgs[:,1,:,:] .* 0.5 .+ imgs[:,2,:,:] .* -0.4542 .+ imgs[:,3,:,:] .* -0.0458 .+ 128, 0, 255)))
    new_imgs
end

function ycrcb2rgb(imgs)
    new_imgs = zeros(UInt8, size(imgs)...)
    new_imgs[:,1,:,:] .= UInt8.(round.(clamp.(imgs[:,1,:,:] .+ (imgs[:,2,:,:] .- 128) .* 0 .+ (imgs[:,3,:,:] .- 128) .* 1.5748, 0, 255)))
    new_imgs[:,2,:,:] .= UInt8.(round.(clamp.(imgs[:,1,:,:] .+ (imgs[:,2,:,:] .- 128) .* -0.1873 .+ (imgs[:,3,:,:] .- 128) .* -0.4681, 0, 255)))
    new_imgs[:,3,:,:] .= UInt8.(round.(clamp.(imgs[:,1,:,:] .+ (imgs[:,2,:,:] .- 128) .* 1.8556 .+ (imgs[:,3,:,:] .- 128) .* 0, 0, 255)))
    new_imgs
end



function to_patch_tuple(patch_idx::Integer, patch_hw::Integer)
    (patch_idx ÷ patch_hw + 1, patch_idx % patch_hw + 1)
end


function training_pg_top_level_pcs(image_size, fname_idx, patch_size, patch_idxs, clt_edges, train_loader, test_loader, num_independent_clusters, num_init_clusters, num_final_clusters, top_level_hclt_params)
    note = "id"
    base_dir = "../progressive_growing/temp/temp_imagenet$(image_size)"
    task_identifier = "$(note)_id$(num_independent_clusters)_init$(num_init_clusters)_final$(num_final_clusters)"
    ll_file_name = joinpath(base_dir,"logs/$(task_identifier)_parallel.log")
    top_level_pc_fname = joinpath(base_dir,"top_level_pcs/$(task_identifier)_$(fname_idx).jpc")



    println("======= subsampling images from dataloaders =======")
    num_patches = length(patch_idxs)
    t = @elapsed begin
        subsampled_tr_data = py"subsample_data_loader"(train_loader, top_level_hclt_params["num_tr_samples"]; shuffle = false, get_data = true)
        subsampled_ts_data = py"subsample_data_loader"(test_loader, top_level_hclt_params["num_ts_samples"]; shuffle = false, get_data = true)

        subsampled_tr_data = rgb2ycrcb(subsampled_tr_data)
        subsampled_ts_data = rgb2ycrcb(subsampled_ts_data)

        subsampled_tr_data = cu(subsampled_tr_data)
        subsampled_ts_data = cu(subsampled_ts_data)
    end
    @printf(" done (%.2fs)\n", t)




    println("======== generating pseudo dataset for the top_level_pc")
    num_patches = length(patch_idxs)
    patch_hw = 8
    n_clusters_y = 0


    patch_level_tr_data = []
    patch_level_ts_data = []
    n_clusters_y = 0
    for (idx, patch_idx) in enumerate(patch_idxs)
        patch_tuple = to_patch_tuple(patch_idx, patch_hw)
        x_s, x_e = (patch_tuple[1] - 1) * patch_size + 1, patch_tuple[1] * patch_size
        y_s, y_e = (patch_tuple[2] - 1) * patch_size + 1, patch_tuple[2] * patch_size
        patch_tr_lls = []
        patch_ts_lls = []
        for cluster_id = 1 : num_independent_clusters
            base_dir1 = joinpath(base_dir,"final_pcs/$(task_identifier)/$(cluster_id)")
            final_pc_fname = joinpath(base_dir1, "final_pc_$(fname_idx).jpc")
            pc = read_mhpc(final_pc_fname)

            num_clusters = length(pc)
            if idx == 1
                n_clusters_y += num_clusters
            end
            mhbpc = CuMultiHeadBitsProbCircuit(pc)
            tr_lls = zeros(Float32, size(subsampled_tr_data, 1), 1, num_clusters)
            ts_lls = zeros(Float32, size(subsampled_ts_data, 1), 1, num_clusters)
            tr_lls[:,1,:] .= Array(loglikelihoods(mhbpc, reshape(subsampled_tr_data[:,:,x_s:x_e,y_s:y_e],(:,3*(patch_size^2))), nothing; batch_size = 256))
            ts_lls[:,1,:] .= Array(loglikelihoods(mhbpc, reshape(subsampled_ts_data[:,:,x_s:x_e,y_s:y_e],(:,3*(patch_size^2))), nothing; batch_size = 256))
            if cluster_id == 1
                patch_tr_lls = tr_lls
                patch_ts_lls = ts_lls
            else
                patch_tr_lls = cat(patch_tr_lls,tr_lls,dims=3)
                patch_ts_lls = cat(patch_ts_lls,ts_lls,dims=3)
                if idx == 1
                    @assert size(patch_tr_lls,3) == n_clusters_y
                end
            end
        end

        if idx == 1
            patch_level_tr_data = patch_tr_lls
            patch_level_ts_data = patch_ts_lls
        else
            patch_level_tr_data = cat(patch_level_tr_data,patch_tr_lls,dims=2)
            patch_level_ts_data = cat(patch_level_ts_data,patch_ts_lls,dims=2)
        end
        @assert size(patch_level_tr_data,2) == idx

        for cluster_id1 = 1 : n_clusters_y
            min_bpd = -maximum(patch_level_tr_data[:,idx,cluster_id1]) / log(2.0) / 3 / patch_size^2
            mean_bpd = -mean(patch_level_tr_data[:,idx,cluster_id1]) / log(2.0) / 3 / patch_size^2
            max_bpd = -minimum(patch_level_tr_data[:,idx,cluster_id1]) / log(2.0) / 3 / patch_size^2
            overprint(@sprintf("  - Completed patch (%2d,%2d) + cluster %3d - min/mean/max bpd: %.2f/%.2f/%.2f",
                            patch_tuple[1], patch_tuple[2], cluster_id1, min_bpd, mean_bpd, max_bpd))
        end
    end


        
    CUDA.unsafe_free!(subsampled_tr_data)
    CUDA.unsafe_free!(subsampled_ts_data)


    

    println("> Total number of low-level PG clusters: $(n_clusters_y) <")

    patch_idxs_dict = Dict{Int,Int}()
    for (idx, patch_idx) in enumerate(patch_idxs)
        patch_idxs_dict[patch_idx] = idx
    end

    # Train the top-level PC
    get_leaf_pcs(patch_idx; num_hidden_cats) = begin
        map(1 : num_hidden_cats) do idx
            ps = rand(Float32, num_hidden_cats) .* 0.01
            ps[idx] += 1.0
            ps ./= sum(ps)
            PlainInputNode(patch_idxs_dict[patch_idx], Categorical(log.(ps)))
        end
    end

    get_edge_params(patch_idx1, patch_idx2; num_hidden_cats) = begin
        pairwise_margs = zeros(Float32, num_hidden_cats, num_hidden_cats) .+ 0.2
        pairwise_margs
    end

    print("> Constructing top-level PC...")
    t = @elapsed top_level_pc = customized_hclt(clt_edges, n_clusters_y; get_leaf_pcs, get_edge_params, parameterize_leaf_edge = true)
    init_parameters(top_level_pc; perturbation=0.4)
    @printf(" done (%.2fs)\n", t)



    print("> Moving PC to GPU...")
    t = @elapsed bpc = CuBitsProbCircuit(top_level_pc)
    @printf(" done (%.2fs)\n", t)

    
    println("> Training top-level PC...")
    mini_batch_em_with_reg(bpc, cu(patch_level_tr_data), top_level_hclt_params["num_epochs1"]; 
                            batch_size = top_level_hclt_params["batch_size"], 
                            param_inertia = top_level_hclt_params["param_inertia1"], 
                            param_inertia_end = top_level_hclt_params["param_inertia2"],
                            pseudocount = top_level_hclt_params["pseudocount"], 
                            soft_reg = 0.0, soft_reg_width = 3, ent_reg = 0.0, log_mode = "plain",
                            verbose = true, eval_dataset = cu(patch_level_ts_data), eval_interval = 3)
    mini_batch_em_with_reg(bpc, cu(patch_level_tr_data), top_level_hclt_params["num_epochs2"]; 
                            batch_size = top_level_hclt_params["batch_size"], 
                            param_inertia = top_level_hclt_params["param_inertia2"], 
                            param_inertia_end = top_level_hclt_params["param_inertia3"],
                            pseudocount = top_level_hclt_params["pseudocount"], 
                            soft_reg = 0.0, soft_reg_width = 5, ent_reg = 0.0, log_mode = "plain",
                            verbose = true, eval_dataset = cu(patch_level_ts_data), eval_interval = 3)
    update_parameters(bpc)
    
    tr_ll = loglikelihood_probcat(bpc, cu(patch_level_tr_data); batch_size = 256)
    ts_ll = loglikelihood_probcat(bpc, cu(patch_level_ts_data); batch_size = 256)
    tr_bpd = -tr_ll / log(2.0) / 3 / patch_size^2 / length(patch_idxs)
    ts_bpd = -ts_ll / log(2.0) / 3 / patch_size^2 / length(patch_idxs)
    @printf("  - top level model: %2d - train bpd: %.4f - test bpd: %.4f\n", fname_idx, tr_bpd, ts_bpd)

    
    # Store PC
    write(top_level_pc_fname, top_level_pc)

    # Store pretrained results
    open(ll_file_name, "a") do io
        write(io, @sprintf("\n ====== Top level bpd: fname %2d - (train %.4f,test %.4f) ====== \n", fname_idx, tr_bpd, ts_bpd))
    end
    
end




