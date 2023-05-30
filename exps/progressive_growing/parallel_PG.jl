using PyCall
using NPZ
using Printf
using Statistics: mean

include("../../VariationalJuice.jl/src/VariationalJuice.jl")
include("../../VariationalJuice.jl/src-jl/LatentPCs.jl")
push!(PyVector(pyimport("sys")["path"]), "./src")

py"""
from kmeans import train_kmeans_model, pred_kmeans_clusters
"""

np = pyimport("numpy")


function rgb2ycrcb(imgs)
    new_imgs = zeros(UInt8, size(imgs)...)
    new_imgs[:,1,:,:] .= UInt8.(round.(clamp.(imgs[:,1,:,:] .* 0.2126 .+ imgs[:,2,:,:] .* 0.7152 .+ imgs[:,3,:,:] .* 0.0722, 0, 255)))
    new_imgs[:,2,:,:] .= UInt8.(round.(clamp.(imgs[:,1,:,:] .* -0.1146 .+ imgs[:,2,:,:] .* -0.3854 .+ imgs[:,3,:,:] .* 0.5 .+ 128, 0, 255)))
    new_imgs[:,3,:,:] .= UInt8.(round.(clamp.(imgs[:,1,:,:] .* 0.5 .+ imgs[:,2,:,:] .* -0.4542 .+ imgs[:,3,:,:] .* -0.0458 .+ 128, 0, 255)))
    new_imgs
end

function main(; dataset, start_cid, end_cid, num_independent_clusters = 400, kwargs...)
    note = "id" #or conv: features from vq-vae2 with independent decoder
    trn_data = np.load("data/data_$(dataset)/data_trn.npy")
    val_data = np.load("data/data_$(dataset)/data_val.npy")
    yz_trn_features = np.load("data/data_$(dataset)/idfeat_trn.npy")
    yz_val_features = np.load("data/data_$(dataset)/idfeat_val.npy")


    trn_data = reshape(rgb2ycrcb(trn_data), (size(trn_data, 1), :))
    val_data = reshape(rgb2ycrcb(val_data), (size(val_data, 1), :))


    # Perform global KMeans clustering
    cls_file_name = "temp/temp_$(dataset)/global_indep_cls/clusters_$(num_independent_clusters)_$(note).npz"
    if !isfile(cls_file_name)
        print("> Global clustering into $(num_independent_clusters) clusters... ")
        t = @elapsed begin
            centroids = py"train_kmeans_model"(yz_trn_features, num_independent_clusters)
            trn_cls_ids = py"pred_kmeans_clusters"(centroids, yz_trn_features)
            val_cls_ids = py"pred_kmeans_clusters"(centroids, yz_val_features)
        end
        println(@sprintf("done (%.2fs)", t))

        NPZ.npzwrite(cls_file_name, Dict("trn_cls_ids" => trn_cls_ids, "val_cls_ids" => val_cls_ids))
    else
        println("> Loaded global cluster ids")
        data = NPZ.npzread(cls_file_name)
        trn_cls_ids = data["trn_cls_ids"]
        val_cls_ids = data["val_cls_ids"]
    end


    

    task_identifier = "$(note)_id$(num_independent_clusters)_init$(Dict(kwargs...)[:num_init_clusters])_final$(Dict(kwargs...)[:num_final_clusters])"
    ll_file_name = "temp/temp_$(dataset)/logs/$(task_identifier)_parallel.log"

    total_trn_bpd = 0.0
    total_val_bpd = 0.0
    for cid = start_cid : end_cid
        final_pc_fname = "temp/temp_$(dataset)/final_pcs/$(task_identifier)/$(cid)/final_pc_$(num_final_clusters).jpc"

        if isfile(final_pc_fname)
            println(">>> Existing mhpc #$(cid) <<<")
            continue
        end


        println(">>> Progressive growing #$(cid) <<<")
        trn_filter = (trn_cls_ids .== cid)
        val_filter = (val_cls_ids .== cid)

        trn_weight = size(trn_data[trn_filter,:],1)
        val_weight = size(val_data[val_filter,:],1)

        println("tr_weight: $(trn_weight) ts_weight: $(val_weight)")

        trn_bpd, val_bpd = progressive_growing(
            dataset,
            trn_data[trn_filter,:], 
            yz_trn_features[trn_filter,:], 
            val_data[val_filter,:], 
            yz_val_features[val_filter,:],
            cid,
            task_identifier;
            kwargs...
        )
        total_trn_bpd += trn_bpd
        mean_trn_bpd = total_trn_bpd / (cid - start_cid + 1)
        total_val_bpd += val_bpd
        mean_val_bpd = total_val_bpd / (cid - start_cid + 1)


        print(@sprintf("cid: %d trn: %.4f; val: %.4f mean(%.4f,%.4f) \n", cid, trn_bpd, val_bpd, mean_trn_bpd, mean_val_bpd))
        open(ll_file_name, "a") do io
            write(io, @sprintf("cid: %d trn: %.4f %d; val: %.4f %d mean(%.4f,%.4f)\n", cid, trn_bpd, trn_weight, val_bpd, val_weight, mean_trn_bpd, mean_val_bpd))
        end
    end
end


function progressive_growing(
                            dataset, trn_data, trn_features, val_data, val_features,
                            global_task_id, task_identifier;
                            num_init_clusters = 1, num_final_clusters = 5, num_latents = 16, max_grow_frac = 0.4, batch_size = 512,prune_threshold = 0.001
                            )
    num_trn_examples = size(trn_data, 1)
    num_val_examples = size(val_data, 1)

    num_vars = size(trn_data, 2)

    trn_data_gpu = cu(trn_data)
    val_data_gpu = cu(val_data)

    if !isdir("temp/temp_$(dataset)/init_pcs/$(task_identifier)")
        mkdir("temp/temp_$(dataset)/init_pcs/$(task_identifier)")
    end

    if !isdir("temp/temp_$(dataset)/final_pcs/$(task_identifier)")
        mkdir("temp/temp_$(dataset)/final_pcs/$(task_identifier)")
    end


    if !isdir("temp/temp_$(dataset)/logs/$(task_identifier)")
        mkdir("temp/temp_$(dataset)/logs/$(task_identifier)")
    end

    grow_ll_file_name = "temp/temp_$(dataset)/logs/$(task_identifier)/$(global_task_id).log"

    # Perform initial KMeans clustering
    # print("> Clustering all samples into $(num_init_clusters) clusters... ")
    t = @elapsed begin
        centroids = py"train_kmeans_model"(trn_features, num_init_clusters)
        trn_cls_ids = py"pred_kmeans_clusters"(centroids, trn_features)
        val_cls_ids = py"pred_kmeans_clusters"(centroids, val_features)
    end
    # println(@sprintf("done (%.2fs)", t))

    # Generate initial structure
    base_dir = "temp/temp_$(dataset)/init_pcs/$(task_identifier)/$(global_task_id)"
    if !isdir(base_dir)
        mkdir(base_dir)
    end
    base_dir1 = "temp/temp_$(dataset)/final_pcs/$(task_identifier)/$(global_task_id)"
    if !isdir(base_dir1)
        mkdir(base_dir1)
    end
    init_pc_fname = joinpath(base_dir, "init_pc_$(num_init_clusters).jpc")
    # final_pc_fname = joinpath(base_dir1, "final_pc_$(num_final_clusters).jpc")




    if !isfile(init_pc_fname)
        println("> Constructing initial multi-headed PC...")
        datasets = []
        for cid = 1 : num_init_clusters
            dataset = trn_data[trn_cls_ids .== cid, :]
            push!(datasets, dataset)
        end
        pcs = joined_hclt(datasets, num_latents; num_cats = 256, input_type = Categorical)
        pcs = pcs[1:num_init_clusters]
        init_parameters(pcs; perturbation = 0.4)

        write_mhpc(init_pc_fname, pcs)
    else
        println("> Loaded initial multi-headed PC")
        pcs = read_mhpc(init_pc_fname)
    end

    ##### Main loop #####
    mean_trn_bpd = 0.0
    mean_val_bpd = 0.0
    trn_bpd = 0.0
    val_bpd = 0.0


    ##choose which ckpt to save
    if num_final_clusters == 20
        c = [5,10,15,20]
    end
    if num_final_clusters == 10
        c = [4,7,10]
    end
    if num_final_clusters == 5
        c = [3,4,5]
    end
    if num_final_clusters == 4
        c = [4]
    end
    if num_final_clusters == 1
        c = [1]
    end


    final_pc_fnames = []
    for ci in c
        final_pc_fname = joinpath(base_dir1, "final_pc_$(ci).jpc")
        push!(final_pc_fnames,final_pc_fname)
    end
    final_pc_fname = final_pc_fnames[1]

    for iter = num_init_clusters : 2 * num_final_clusters
        println("==== Iteration $(iter) ====")

        num_clusters = length(pcs)

        ## Step 1: train the multi-head PC
        println("> Training multi-head PC...")
        mhbpc = CuMultiHeadBitsProbCircuit(pcs)
        trn_head_mask = zeros(Float32, num_trn_examples, num_clusters)
        ids = [CartesianIndex(i, j) for (i, j) in zip(collect(1:num_trn_examples), trn_cls_ids)]
        trn_head_mask[ids] .= one(Float32)
        trn_head_mask_gpu = cu(trn_head_mask)

        lls = mini_batch_em_for_multihead_pc(
            mhbpc, trn_data_gpu, trn_head_mask_gpu, 50;
            batch_size, pseudocount = 0.1, soft_reg = 0.0, soft_reg_width = 3, 
            param_inertia = 0.9, param_inertia_end = 0.99
        )
        update_parameters(mhbpc)


        per_sample_lls = Array(loglikelihoods(mhbpc, val_data_gpu, nothing; batch_size=128))
        mean_bpd = -mean(per_sample_lls) / log(2.0) / num_vars
        
        println("  - Train bpd: $(-lls[end] / log(2.0) / num_vars)")
        println("  - Test  bpd: $mean_bpd")
        println("  - Number of nodes: $(length(mhbpc.bpc.nodes) - 1)")
        println("  - Number of edges: $(length(mhbpc.bpc.edge_layers_up.vectors) - num_clusters)")

        ## Step 2: prune multi-head PC
        print("> Pruning multi-head PC...")
        t = @elapsed pcs = prune_pc(pcs, trn_data_gpu, trn_head_mask_gpu; batch_size, prune_threshold, mhbpc)
        println(@sprintf("done (%.2fs)", t))

        ## Step 3: evaluate and re-assign cluster ids
        mhbpc = CuMultiHeadBitsProbCircuit(pcs)
        per_sample_lls = Array(loglikelihoods(mhbpc, trn_data_gpu, nothing; batch_size))
        mean_trn_bpd = -mean(per_sample_lls) / log(2.0) / num_vars
        best_lls, _ = findmax(per_sample_lls; dims = 2)
        min_trn_bpd = -mean(best_lls) / log(2.0) / num_vars

        per_cluster_ll = map(1:num_clusters) do idx
            mean(per_sample_lls[trn_cls_ids .== idx, idx])
        end
        per_cluster_bpd = -per_cluster_ll / log(2.0) / num_vars
        per_cluster_weight = map(1:num_clusters) do idx
            size(per_sample_lls[trn_cls_ids .== idx, idx], 1)
        end
        trn_bpd = sum(per_cluster_bpd .* per_cluster_weight) / sum(per_cluster_weight)
        
        per_sample_lls_val = Array(loglikelihoods(mhbpc, val_data_gpu, nothing; batch_size=128))
        mean_val_bpd = -mean(per_sample_lls_val) / log(2.0) / num_vars
        best_lls_val, _ = findmax(per_sample_lls_val; dims = 2)
        min_val_bpd = -mean(best_lls_val) / log(2.0) / num_vars


        per_cluster_weight_val = map(1:num_clusters) do idx
            sum(val_cls_ids .== idx)
        end
        
        per_cluster_ll_val = map(1:num_clusters) do idx
            if per_cluster_weight_val[idx] > 0
                mean(per_sample_lls_val[val_cls_ids .== idx, idx])
            else
                zero(Float32)
            end
        end

        per_cluster_bpd_val = -per_cluster_ll_val / log(2.0) / num_vars
        val_bpd = sum(per_cluster_bpd_val .* per_cluster_weight_val) / sum(per_cluster_weight_val)

        println("  - Weighted bpd: ($(trn_bpd),$(val_bpd))")
        println("  - Overall average bpd: ($(mean_trn_bpd),$(mean_val_bpd))")
        println("  - Number of nodes: $(length(mhbpc.bpc.nodes) - 1)")
        println("  - Number of edges: $(length(mhbpc.bpc.edge_layers_up.vectors) - num_clusters)")
        
        open(grow_ll_file_name, "a") do io
            write(io, @sprintf("trn: %.4f  val: %.4f n_cls:%d \n", trn_bpd, val_bpd, length(pcs)))
        end
        
        if length(c) == 1
            if length(pcs) >= c[1] && !isfile(final_pc_fname) 
                write_mhpc(final_pc_fname, pcs)
                break
            end
        else
            for i = 2 : length(c) - 1
                if length(pcs) >= c[i] && length(pcs) < c[i+1]
                    final_pc_fname = final_pc_fnames[i]
                    break
                end
            end

            if length(pcs) >= c[end]
                final_pc_fname = final_pc_fnames[end]
            end

            if length(pcs) >= c[1]
                if !isfile(final_pc_fname)
                    write_mhpc(final_pc_fname, pcs)
                end
            end

            if length(pcs) >= c[end]
                for final_pc_fname in final_pc_fnames
                    if !isfile(final_pc_fname)
                        write_mhpc(final_pc_fname, pcs)
                    end
                end
                break
            end
        end
                
        
        
        per_sample_lls[ids] .+= 1.0 # Reluctant to switch cluster id
        _, mxcls = findmax(per_sample_lls; dims = 2)
        trn_cls_ids = map(id -> id.I[2], mxcls[:,1])


        
        ## Step 4: decide clusters to grow
        grow_cls_true = []
        sorted_lls = sortperm(per_cluster_ll)
        thr = Int(round(num_trn_examples * 0.4))
        cnt = 0
        for i = 1 : num_clusters
            push!(grow_cls_true,sorted_lls[i])
            cnt += per_cluster_weight[sorted_lls[i]]
            if cnt > thr
                break
            end
        end

        if num_final_clusters <= 5
            target_n_clusters = length(grow_cls_true) + 1
        else
            thr1 = cnt ÷ 8000 #if dataset == "CIFAR10": 2500
            rand_min = length(grow_cls_true) + 1
            rand_max = min(num_final_clusters, 2*length(grow_cls_true), thr1)
            if rand_min > rand_max
                print("\n thr1:",thr1,"\n")
                for final_pc_fname in final_pc_fnames
                    if !isfile(final_pc_fname)
                        write_mhpc(final_pc_fname, pcs)
                    end
                end
                break
            end
            target_n_clusters = rand_min + Int(round(rand()*(rand_max-rand_min)))
        end

        grow_n_clusters = target_n_clusters - length(grow_cls_true)
        grow_cls = grow_cls_true[1:grow_n_clusters]


        ## Step 5-1: grow the multi-head PC
        print("> Growing the multi-head PC...")
        t = @elapsed pcs = begin
            filter = zeros(Bool, num_trn_examples)
            for cluster in grow_cls
                filter .|= (trn_cls_ids .== cluster)
            end
            head_mask = zeros(Float32, num_trn_examples, num_clusters)
            ids = [CartesianIndex(i, j) for (i, j) in zip(collect(1:num_trn_examples), trn_cls_ids)]
            head_mask[ids] .= one(Float32)
            pcs = grow_heads_by_flows(
                pcs, trn_data_gpu[filter,:], cu(head_mask[filter,:]); 
                sigma = 0.2, node_selection_method = "percentage", 
                node_selection_args = Dict("grow_frac" => max(grow_n_clusters / (grow_n_clusters + num_clusters), 0.2)),
                batch_size
            )
            @assert length(pcs) == num_clusters + grow_n_clusters
            pcs
        end
        println(@sprintf("done (%.2fs)", t))

        ## Step 5-2: update cluster ids
        all_trn_features = []
        old_centroids = []
        all_trn_data = []
        print("> Updating cluster ids...\n")
        for (i, cluster) in enumerate(grow_cls_true)
            trn_filter = (trn_cls_ids .== cluster)
            if i == 1
                all_trn_data = trn_data[trn_filter,:]
                all_trn_features = trn_features[trn_filter,:]
                old_centroids = mean(trn_features[trn_filter,:],dims=1)
            else
                all_trn_data = cat(all_trn_data,trn_data[trn_filter,:],dims=1)
                all_trn_features = cat(all_trn_features,trn_features[trn_filter,:],dims=1)
                old_centroids = cat(old_centroids,mean(trn_features[trn_filter,:],dims=1),dims=1)
            end
        end

        centroids = py"train_kmeans_model"(all_trn_features, target_n_clusters, centroids=old_centroids)
        trn_filter = zeros(Bool, num_trn_examples)
        val_filter = zeros(Bool, num_val_examples)
        for cluster in grow_cls_true
            trn_filter .|= (trn_cls_ids .== cluster)
            val_filter .|= (val_cls_ids .== cluster)
        end
        cls_ids_trn = py"pred_kmeans_clusters"(centroids, trn_features[trn_filter,:])
        cls_ids_val = py"pred_kmeans_clusters"(centroids, val_features[val_filter,:])
        for j = 1 : target_n_clusters
            if j <= length(grow_cls_true)
                @views trn_cls_ids[trn_filter][cls_ids_trn .== j] .= grow_cls_true[j]
                @views val_cls_ids[val_filter][cls_ids_val .== j] .= grow_cls_true[j]
            else
                @views trn_cls_ids[trn_filter][cls_ids_trn .== j] .= j + num_clusters - length(grow_cls_true)
                @views val_cls_ids[val_filter][cls_ids_val .== j] .= j + num_clusters - length(grow_cls_true)
            end
        end
    end
    trn_bpd, val_bpd
end

start_cid = parse(Int, ARGS[1])
end_cid = parse(Int, ARGS[2])
num_independent_clusters = parse(Int, ARGS[3])
dataset = ARGS[4]
println("dataset: $(dataset)")
num_init_clusters = 2
num_final_clusters = 4


main(; dataset, start_cid, end_cid, num_independent_clusters, num_init_clusters = num_init_clusters, num_final_clusters = num_final_clusters, num_latents = 16, max_grow_frac = 0.4, batch_size = 256,
    prune_threshold = 1e-4)