using Graphs: SimpleGraph, bfs_tree, center, add_edge!


function customized_hclt(clt_edges, num_hidden_cats; get_leaf_pcs::Union{Function,Nothing} = nothing, 
                         get_edge_params::Union{Function,Nothing} = nothing, parameterize_leaf_edge = true)

    clt = edges2graphs(clt_edges)

    # Construct the CLT circuit bottom-up
    node_seq = bottom_up_order(clt)
    for curr_node in node_seq
        out_neighbors = outneighbors(clt, curr_node)
        
        # meaning: `pcs' of leaf CLT nodes refer to a collection of marginal distribution Pr(X);
        #          `pcs' of an inner CLT node (corr. var Y) is a collection of joint distributions
        #              over itself and its child vars (corr. var X_1, ..., X_k): Pr(Y)Pr(X_1|Y)...Pr(X_k|Y)

        var = get_prop(clt, curr_node, :var)
        
        if length(out_neighbors) == 0
            # Leaf node
            pcs = get_leaf_pcs(var; num_hidden_cats)
            set_prop!(clt, curr_node, :pcs, pcs)
        else
            # Inner node
            ch_pcs = Vector{Vector{ProbCircuit}}()
            for child_node in out_neighbors
                pcs = get_prop(clt, child_node, :pcs)
                if parameterize_leaf_edge || length(outneighbors(clt, child_node)) > 0
                    ch_var = get_prop(clt, child_node, :var)
                    pcs = [summate(pcs...) for _ = 1 : num_hidden_cats]
                    if !isnothing(get_edge_params)
                        latent_params = get_edge_params(var, ch_var; num_hidden_cats)
                        for i = 1 : num_hidden_cats
                            norm_params = latent_params[i,:] ./ sum(latent_params[i,:])
                            pcs[i].params .= log.(norm_params)
                        end
                    end
                end
                push!(ch_pcs, pcs)
            end
            push!(ch_pcs, get_leaf_pcs(var; num_hidden_cats))

            pcs = [multiply([ch_pc[cat_idx] for ch_pc in ch_pcs]...) for cat_idx = 1 : num_hidden_cats]
            set_prop!(clt, curr_node, :pcs, pcs)
        end
    end
    
    summate(get_prop(clt, node_seq[end], :pcs))
end

function customized_hclt_p2(clt_edges, num_hidden_cats, num_latents; get_leaf_pcs::Union{Function,Nothing} = nothing, 
                         get_edge_params::Union{Function,Nothing} = nothing, parameterize_leaf_edge = true)

    clt = edges2graphs(clt_edges)

    # Construct the CLT circuit bottom-up
    node_seq = bottom_up_order(clt)
    for curr_node in node_seq
        out_neighbors = outneighbors(clt, curr_node)
        
        # meaning: `pcs' of leaf CLT nodes refer to a collection of marginal distribution Pr(X);
        #          `pcs' of an inner CLT node (corr. var Y) is a collection of joint distributions
        #              over itself and its child vars (corr. var X_1, ..., X_k): Pr(Y)Pr(X_1|Y)...Pr(X_k|Y)

        var = get_prop(clt, curr_node, :var)
        
        if length(out_neighbors) == 0
            # Leaf node
            pcs = get_leaf_pcs(var; num_hidden_cats)
            pcs = [summate(pcs...) for _ = 1 : num_latents]
            num_hidden_cats = num_latents
            set_prop!(clt, curr_node, :pcs, pcs)
        else
            # Inner node
            ch_pcs = Vector{Vector{ProbCircuit}}()
            for child_node in out_neighbors
                pcs = get_prop(clt, child_node, :pcs)
                if parameterize_leaf_edge || length(outneighbors(clt, child_node)) > 0
                    ch_var = get_prop(clt, child_node, :var)
                    pcs = [summate(pcs...) for _ = 1 : num_latents]
                    if !isnothing(get_edge_params)
                        latent_params = get_edge_params(var, ch_var; num_hidden_cats)
                        for i = 1 : num_latents
                            norm_params = latent_params[i,:] ./ sum(latent_params[i,:])
                            pcs[i].params .= log.(norm_params)
                        end
                    end
                end
                push!(ch_pcs, pcs)
            end
            push!(ch_pcs, get_leaf_pcs(var; num_hidden_cats))

            pcs = [multiply([ch_pc[cat_idx] for ch_pc in ch_pcs]...) for cat_idx = 1 : num_latents]
            set_prop!(clt, curr_node, :pcs, pcs)
        end
    end
    
    summate(get_prop(clt, node_seq[end], :pcs))
end





function edges2graphs(edgepair)
    vars = sort(collect(Set(append!(first.(edgepair), last.(edgepair)))))
    num_vars = length(vars)

    var2idx = Dict{Int,Int}()
    for (idx, var) in enumerate(vars)
        var2idx[var] = idx
    end

    MStree = SimpleGraph(num_vars)
    for edge in edgepair
        add_edge!(MStree, var2idx[edge[1]], var2idx[edge[2]])
    end

    clt = MetaDiGraph(bfs_tree(MStree, center(MStree)[1]))

    for i = 1 : num_vars
        set_prop!(clt, i, :var, vars[i])
    end
    clt
end