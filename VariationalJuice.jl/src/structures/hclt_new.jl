using ChowLiuTrees
using Graphs: nv


function hclt_new(data, num_hidden_cats; 
                  num_cats = nothing,
                  shape = :directed,
                  input_type = Literal,
                  pseudocount = 0.1)
    
    clt_edges = learn_chow_liu_tree(data; pseudocount, Float=Float32)
    clt = PCs.clt_edges2graphs(clt_edges; shape)
    
    if num_cats === nothing
        num_cats = maximum(data) + 1
    end
    hclt_from_clt_new(clt, num_cats, num_hidden_cats; input_type)
end


function hclt_from_clt_new(clt, num_cats, num_hidden_cats; input_type = Literal)
    
    num_vars = nv(clt)

    # meaning: `joined_leaves[i,j]` is a distribution of the hidden variable `i` having value `j` 
    # conditioned on the observed variable `i`
    joined_leaves = PCs.categorical_leaves(num_vars, num_cats, num_hidden_cats, input_type)
    
    # Construct the CLT circuit bottom-up
    node_seq = bottom_up_order(clt)
    for curr_node in node_seq
        out_neighbors = outneighbors(clt, curr_node)
        
        # meaning: `circuits' of leaf CLT nodes refer to a collection of marginal distribution Pr(X);
        #          `circuits' of an inner CLT node (corr. var Y) is a collection of joint distributions
        #              over itself and its child vars (corr. var X_1, ..., X_k): Pr(Y)Pr(X_1|Y)...Pr(X_k|Y)
        
        if length(out_neighbors) == 0
            # Leaf node

            # We do not add hidden variables for leaf nodes
            circuits = joined_leaves[curr_node, :]
            set_prop!(clt, curr_node, :circuits, circuits)
        else
            # Inner node
            
            # Each element in `child_circuits' represents the joint distribution of the child nodes, 
            # i.e., Pr(X_1)...Pr(X_k)
            child_circuits = [get_prop(clt, child_node, :circuits) for child_node in out_neighbors]
            if length(out_neighbors) > 1
                child_circuits = [multiply([child_circuit[cat_idx] for child_circuit in child_circuits]) for cat_idx = 1 : num_hidden_cats]
                child_circuits = [summate(child_circuits...) for _ = 1 : num_hidden_cats]
            else
                child_circuits = child_circuits[1]
            end
            # Pr(X_1)...Pr(X_k) -> Pr(Y)Pr(X_1|Y)...Pr(X_k|Y)
            circuits = [summate(multiply.(child_circuits, joined_leaves[curr_node, :])) for cat_idx = 1 : num_hidden_cats]
            set_prop!(clt, curr_node, :circuits, circuits)
        end
    end
    
    get_prop(clt, node_seq[end], :circuits)[1] # A ProbCircuit node
end

function corr_hclt(data, num_hidden_cats; 
                   num_cats = nothing,
                   shape = :directed,
                   input_type = Literal,
                   pseudocount = 0.1,
                   mi_threshold_frac = 0.5,
                   num_cor_reps = 8)
    
    MI = pairwise_MI_chunked(data; chunk_size = 16, pseudocount)
    clt_edges = ChowLiuTrees.topk_MST(-Array(MI); num_trees = 1, dropout_prob = 0.0)[1]
    clt = PCs.clt_edges2graphs(clt_edges; shape)

    min_mi, max_mi = minimum(MI), maximum(MI)
    mi_threshold = min_mi + (max_mi - min_mi) * mi_threshold_frac
    
    if num_cats === nothing
        num_cats = maximum(data) + 1
    end
    high_order_hclt_from_clt_new(clt, MI, num_cats, num_hidden_cats; mi_threshold, num_cor_reps, input_type)
end

function high_order_hclt_from_clt_new(clt, MI, num_cats, num_hidden_cats; mi_threshold, num_cor_reps, input_type = Literal)
    
    num_vars = nv(clt)

    for i = 1 : size(MI, 1)
        MI[i, i] = zero(Float32)
    end

    # meaning: `joined_leaves[i,j]` is a distribution of the hidden variable `i` having value `j` 
    # conditioned on the observed variable `i`
    joined_leaves = PCs.categorical_leaves(num_vars, num_cats, num_hidden_cats, input_type)
    
    # Construct the CLT circuit bottom-up
    node_seq = bottom_up_order(clt)
    for curr_node in node_seq
        out_neighbors = outneighbors(clt, curr_node)
        
        # meaning: `circuits' of leaf CLT nodes refer to a collection of marginal distribution Pr(X);
        #          `circuits' of an inner CLT node (corr. var Y) is a collection of joint distributions
        #              over itself and its child vars (corr. var X_1, ..., X_k): Pr(Y)Pr(X_1|Y)...Pr(X_k|Y)
        
        if length(out_neighbors) == 0
            # Leaf node

            # We do not add hidden variables for leaf nodes
            circuits = joined_leaves[curr_node, :]
            set_prop!(clt, curr_node, :circuits, circuits)
        else
            # Inner node

            if length(out_neighbors) > 1
                child_groups = Vector()
                curr_MI = MI[out_neighbors, out_neighbors]
                added_vars = Set()
                while true
                    max_mi = maximum(curr_MI)
                    if max_mi < mi_threshold
                        break
                    end
                    var1, var2 = argmax(curr_MI).I
                    push!(child_groups, (out_neighbors[var1], out_neighbors[var2]))
                    curr_MI[var1,:] .= zero(Float32)
                    curr_MI[var2,:] .= zero(Float32)
                    curr_MI[:,var1] .= zero(Float32)
                    curr_MI[:,var2] .= zero(Float32)
                    push!(added_vars, out_neighbors[var1])
                    push!(added_vars, out_neighbors[var2])
                end
                for var in out_neighbors
                    if !(var in added_vars)
                        push!(child_groups, var)
                    end
                end

                group_pcs = map(1 : length(child_groups)) do group_idx
                    if child_groups[group_idx] isa Tuple
                        pcs1 = get_prop(clt, child_groups[group_idx][1], :circuits)
                        println(child_groups[group_idx][2], " ", out_neighbors)
                        pcs2 = get_prop(clt, child_groups[group_idx][2], :circuits)
                        map(1:num_hidden_cats) do idx
                            pairs = Vector()
                            push!(pairs, multiply(pcs1[idx], pcs2[idx]))
                            for _ = 1 : num_cor_reps
                                push!(pairs, multiply(pcs1[rand(1:num_hidden_cats)], pcs2[rand(1:num_hidden_cats)]))
                            end
                            summate(pairs...)
                        end
                    else
                        get_prop(clt, child_groups[group_idx], :circuits)
                    end
                end

                child_circuits = [summate(multiply([item[cat_idx] for item in group_pcs])) for cat_idx = 1 : num_hidden_cats]
            else
                child_circuits = get_prop(clt, out_neighbors[1], :circuits)
            end

            # Pr(X_1)...Pr(X_k) -> Pr(Y)Pr(X_1|Y)...Pr(X_k|Y)
            circuits = [summate(multiply.(child_circuits, joined_leaves[curr_node, :])) for cat_idx = 1 : num_hidden_cats]
            set_prop!(clt, curr_node, :circuits, circuits)
        end
    end
    
    get_prop(clt, node_seq[end], :circuits)[1] # A ProbCircuit node
end