using NPZ

function mar_cont(x::Vector, y::Vector, h::Integer, w::Integer)
    cont = zeros(Float32, h, w)
    Threads.@threads for i = 1 : size(x, 1)
        xid, yid = x[i], y[i]
        cont[xid, yid] += 1
    end
    cont
end

function init_lvd_pcs(base_folder_name::String; num_vars::Integer, num_cats::Integer, n_clusters::Integer,
                      init_scale::Float32 = 2.0, leaf_noise_factor::Float32 = 0.1, sum_noise_factor::Float32 = 0.04)
    @assert isdir(base_folder_name)
    region_graph_fname = joinpath(base_folder_name, "region_graph.json")
    metadata_fname = joinpath(base_folder_name, "metadata.npz")

    rnode_root = parse_rg_from_file(region_graph_fname)
    metadata = NPZ.npzread(metadata_fname)
    node_lvs = metadata["lvs"]
    leaf_xs = metadata["xs"]

    @assert size(leaf_xs, 2) == num_vars
    @assert maximum(leaf_xs) <= num_cats
    @assert maximum(node_lvs) <= n_clusters
    
    f_input(n::RegionGraph)::Vector{ProbCircuit} = begin
        v = randvar(n)
        n_xs = leaf_xs[:,v]
        n_hs = node_lvs[:,n.node_id]

        cont = mar_cont(n_hs, n_xs, n_clusters, num_cats)
        cont ./= sum(cont; dims = 2)
        randvals = exp.(rand(n_clusters, num_cats) .* -init_scale)
        randvals ./= sum(randvals; dims = 2)
        probs = (1.0 - leaf_noise_factor) .* cont .+ leaf_noise_factor .* randvals

        map(1:n_clusters) do i
            PCs.PlainInputNode(v, PCs.Categorical(log.(probs[i,:])))
        end
    end
    f_partition(n::RegionGraph, ins)::Vector{ProbCircuit} = begin
        map(1:n_clusters) do i
            chs = map(cs -> cs[i], ins)
            multiply(chs...)
        end
    end
    f_inner(n::RegionGraph, ins)::Vector{ProbCircuit} = begin
        probs = zeros(Float32, n_clusters, num_children(n) * n_clusters)
        n_nodeid = n.node_id
        for i = 1 : num_children(n)
            c = children(n)[i]
            ch_nodeid = children(c)[1].node_id
            cont = mar_cont(node_lvs[:,n_nodeid], node_lvs[:,ch_nodeid], n_clusters, n_clusters)
            probs[:,(i-1)*n_clusters+1:i*n_clusters] .= cont
        end
        probs ./= sum(probs; dims = 2)
        randvals = exp.(rand(n_clusters, num_children(n) * n_clusters) .* -init_scale)
        randvals ./= sum(randvals; dims = 2)
        probs = (1.0 - sum_noise_factor) .* cont .+ sum_noise_factor .* randvals

        all_chs = reduce(vcat, ins)
        map(1:n_clusters) do i
            m = summate(all_chs)
            m.params .= log.(probs[i,:])
            m
        end
    end
    pcs = foldup_aggregate(rnode_root, f_input, f_partition, f_inner, Vector{ProbCircuit})

    root_probs = zeros(Float32, num_children(pcs[1]))
    for n in pcs
        root_probs .+= exp.(n.params)
    end
    root_probs ./= sum(root_probs)

    pcs[1].params .= log.(root_probs)
    pcs[1]
end