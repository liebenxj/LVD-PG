

function categorical_clt(clt_edges; num_cats = 256, data = nothing, pseudocount = 0.1)

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
            pcs = map(0:num_cats-1) do cat_idx
                PlainInputNode(var, Indicator{UInt8}(UInt8(cat_idx)))
            end
            set_prop!(clt, curr_node, :pcs, pcs)
        else
            # Inner node
            ch_pcs = Vector{Vector{ProbCircuit}}()
            for child_node in out_neighbors
                pcs = get_prop(clt, child_node, :pcs)

                ch_var = get_prop(clt, child_node, :var)
                joint_cont = get_pairwise_cont(data, var, ch_var; num_cats, pseudocount)
                pcs = [summate(pcs...) for _ = 1 : num_cats]
                for i = 1 : num_cats
                    norm_params = joint_cont[i,:] ./ sum(joint_cont[i,:])
                    pcs[i].params .= log.(norm_params)
                end

                push!(ch_pcs, pcs)
            end
            pcs = map(0:num_cats-1) do cat_idx
                PlainInputNode(var, Indicator{UInt8}(UInt8(cat_idx)))
            end
            push!(ch_pcs, pcs)

            pcs = [multiply([ch_pc[cat_idx] for ch_pc in ch_pcs]...) for cat_idx = 1 : num_cats]
            set_prop!(clt, curr_node, :pcs, pcs)
        end
    end
    
    var = get_prop(clt, node_seq[end], :var)
    marg_cont = get_marg_cont(data, var; num_cats, pseudocount)
    n = summate(get_prop(clt, node_seq[end], :pcs))
    n.params .= log.(marg_cont ./ sum(marg_cont))
    n
end


function categorical_clt_group(clt_edges; num_cats = 256, num_groups = 16, data = nothing, pseudocount = 0.1)

    clt = edges2graphs(clt_edges)

    gsize = num_cats ÷ num_groups

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
            marg_cont = get_marg_cont(data, var; num_cats, pseudocount)
            pcs = map(0:num_groups-1) do g_idx
                ps = zeros(Float32, num_cats) .+ 1e-4
                ps[g_idx*gsize+1:(g_idx+1)*gsize] .= marg_cont[g_idx*gsize+1:(g_idx+1)*gsize]
                logps = log.(ps ./ sum(ps))
                PlainInputNode(var, Categorical(logps))
            end
            set_prop!(clt, curr_node, :pcs, pcs)
        else
            # Inner node
            ch_pcs = Vector{Vector{ProbCircuit}}()
            for child_node in out_neighbors
                pcs = get_prop(clt, child_node, :pcs)

                ch_var = get_prop(clt, child_node, :var)
                joint_cont = get_pairwise_cont(data, var, ch_var; num_cats, pseudocount)
                g_cont = zeros(Float32, num_groups, num_groups)
                for i = 0 : num_groups - 1
                    for j = 0 : num_groups - 1
                        g_cont[i+1,j+1] = sum(joint_cont[i*gsize+1:(i+1)*gsize, j*gsize+1:(j+1)*gsize])
                    end
                end
                pcs = [summate(pcs...) for _ = 1 : num_groups]
                for g_idx = 1 : num_groups
                    norm_params = g_cont[g_idx,:] ./ sum(g_cont[g_idx,:])
                    pcs[g_idx].params .= log.(norm_params)
                end

                push!(ch_pcs, pcs)
            end
            
            marg_cont = get_marg_cont(data, var; num_cats, pseudocount)
            pcs = map(0:num_groups-1) do g_idx
                ps = zeros(Float32, num_cats) .+ 1e-4
                ps[g_idx*gsize+1:(g_idx+1)*gsize] .= marg_cont[g_idx*gsize+1:(g_idx+1)*gsize]
                logps = log.(ps ./ sum(ps))
                PlainInputNode(var, Categorical(logps))
            end
            push!(ch_pcs, pcs)

            pcs = [multiply([ch_pc[cat_idx] for ch_pc in ch_pcs]...) for cat_idx = 1 : num_groups]
            set_prop!(clt, curr_node, :pcs, pcs)
        end
    end
    
    var = get_prop(clt, node_seq[end], :var)
    marg_cont = get_marg_cont(data, var; num_cats, pseudocount)
    g_cont = zeros(Float32, num_groups)
    for i = 0 : num_groups - 1
        g_cont[i+1] = sum(marg_cont[i*gsize+1:(i+1)*gsize])
    end
    n = summate(get_prop(clt, node_seq[end], :pcs))
    n.params .= log.(g_cont ./ sum(g_cont))
    n
end