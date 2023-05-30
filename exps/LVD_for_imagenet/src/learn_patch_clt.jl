using NPZ
using ChowLiuTrees: topk_MST
using ProbabilisticCircuits: clt_edges2graphs
using PyCall
using Pickle

include("../../../VariationalJuice.jl/src/VariationalJuice.jl")

push!(PyVector(pyimport("sys")["path"]), "../")
push!(PyVector(pyimport("sys")["path"]), "./src")

py"""
from extract_features import get_img_features, discretize_features, get_patch_hw, subsample_data_loader
"""

function get_ancestors(g::MetaDiGraph)
    num_nodes = length(vertices(g))
    edges = Vector{Tuple{UInt32,UInt32}}()

    ancestors = Dict{Int,Set}()
    
    function dfs(node_idx)
        out_neighbors = outneighbors(g, node_idx)
        
        for out_neighbor in out_neighbors
            if haskey(ancestors, out_neighbor)
                union!(ancestors[out_neighbor], deepcopy(ancestors[node_idx]))
            else
                ancestors[out_neighbor] = deepcopy(ancestors[node_idx])
            end
            push!(ancestors[out_neighbor], out_neighbor)
            push!(edges, (node_idx, out_neighbor))
            dfs(out_neighbor)
        end
    end
        
    root_node_idx = findall(x->x==0, indegree(g))[1]
    ancestors[root_node_idx] = Set{Int}([root_node_idx])
    dfs(root_node_idx)
    
    ancestors, edges
end

function get_descendents(g::MetaDiGraph)
    num_nodes = length(vertices(g))
    descendents = Dict{Int,Set}()

    function dfs(node_idx)
        out_neighbors = outneighbors(g, node_idx)
        des = Set{Int}([node_idx])
        for out_neighbor in out_neighbors
            dfs(out_neighbor)
            union!(des, descendents[out_neighbor])
        end
        descendents[node_idx] = des
    end

    root_node_idx = findall(x->x==0, indegree(g))[1]
    dfs(root_node_idx)

    descendents
end

function get_siblings(g::MetaDiGraph)
    num_nodes = length(vertices(g))

    siblings = Dict{Int,Set}()
    
    function dfs(node_idx)
        out_neighbors = outneighbors(g, node_idx)
        
        for out_neighbor in out_neighbors
            d = Set{Int}()
            for sib in out_neighbors
                if sib != out_neighbor
                    push!(d, sib)
                end
            end
            siblings[out_neighbor] = d

            dfs(out_neighbor)
        end
    end
        
    root_node_idx = findall(x->x==0, indegree(g))[1]
    siblings[root_node_idx] = Set{Int}()
    dfs(root_node_idx)
    
    siblings
end

function learn_clt_for_top_pc(model, device, train_loader, patch_size; patch_hw, patch_n, patch_x1, patch_y1, n_clusters, base_dir, 
                            patch_kernel_size = 3, minimum_num_tr_samples = 100000, source_model = "VQVAE", from_py = false)

    if from_py
        patch_x1 += 1
        patch_y1 += 1
    end

    file_name = joinpath(base_dir, "clt_edges.pkl")
    if isfile(file_name)
        clt_edges, idxs_mapping = Pickle.load(file_name)
    else
        patch_cluster_idxs = nothing
        idxs_mapping = []

        # Get a subsampled data loader
        sampled_train_loader = py"subsample_data_loader"(train_loader, minimum_num_tr_samples; shuffle = false)

        for patch_x = patch_x1 : patch_x1 + patch_n - 1
            for patch_y = patch_y1 : patch_y1 + patch_n - 1
                target_patch = (patch_x, patch_y)
                println(" - extracting features of patch ($(patch_x), $(patch_y))...")
                visible_patches = []
                for i = -(patch_kernel_size-1) ÷ 2 : (patch_kernel_size-1) ÷ 2
                    for j = -(patch_kernel_size-1) ÷ 2 : (patch_kernel_size-1) ÷ 2
                        if patch_x + i <= 0 || patch_x + i > patch_hw || patch_y + j <= 0 || patch_y + j > patch_hw
                            continue
                        end
                        push!(visible_patches, (patch_x + i, patch_y + j))
                    end
                end
                features = py"get_img_features"(
                    model, device; data_loader = sampled_train_loader, target_patch = target_patch, 
                    visible_patches = visible_patches, source_model = source_model
                )
                if source_model in ["MAE", "pixel"]
                    disc_features, = py"discretize_features"(features, n_clusters; eval_features_set = [features], method = "Kmeans")
                else
                    disc_features = features
                end
                if patch_cluster_idxs === nothing
                    patch_cluster_idxs = disc_features
                else
                    patch_cluster_idxs = hcat(patch_cluster_idxs, disc_features)
                end

                push!(idxs_mapping, (patch_x - 1) * patch_hw + patch_y)
            end
        end
        idxs_mapping .-= 1

        print("> Learning CLT... ")
        MI = pairwise_MI_chunked(cu(patch_cluster_idxs); chunk_size = 16, pseudocount = 0.1)
        clt_edges = topk_MST(-MI; num_trees = 1, dropout_prob = 0.0)[1]
        store(file_name, (clt_edges, idxs_mapping))
    end

    clt = clt_edges2graphs(clt_edges; shape = :directed)

    ancestors, clt_edges = get_ancestors(clt)
    siblings = get_siblings(clt)
    descendents = get_descendents(clt)

    # Apply edge mapping
    for i = 1 : length(clt_edges)
        edge = clt_edges[i]
        clt_edges[i] = (idxs_mapping[edge[1]], idxs_mapping[edge[2]])
    end
    ancestors_dict = Dict{Int,Vector}()
    for i = 1 : length(ancestors)
        ancestors_dict[idxs_mapping[i]] = map(x->idxs_mapping[x], collect(ancestors[i]))
    end
    descendents_dict = Dict{Int,Vector}()
    for i = 1 : length(descendents)
        descendents_dict[idxs_mapping[i]] = map(x->idxs_mapping[x], collect(descendents[i]))
    end

    idxs_mapping, clt_edges, ancestors_dict, descendents_dict
end