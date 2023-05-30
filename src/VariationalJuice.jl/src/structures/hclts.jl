using ProbabilisticCircuits: Literal, Categorical, clt_edges2graphs, Var, bottom_up_order
using ChowLiuTrees: learn_chow_liu_tree
using MetaGraphs: get_prop, set_prop!, MetaDiGraph, vertices, indegree, outneighbors

import ProbabilisticCircuits: hclt # extend

function hclt(data, num_hidden_cats, variable_groups::Vector; num_cats = nothing, 
              input_type = Categorical, pseudocount = 0.1, shape = :directed)
    num_mapped_vars = length(variable_groups)
    num_examples = size(data, 1)

    mapped_data = CUDA.zeros(eltype(data), num_examples, num_mapped_vars)
    @inbounds for idx = 1 : num_mapped_vars
        mapped_data[:,idx] .= data[:,variable_groups[idx][1]]
    end

    clt_edges = learn_chow_liu_tree(mapped_data; pseudocount, Float = Float32)
    clt = clt_edges2graphs(clt_edges; shape)

    if num_cats === nothing
        num_cats = maximum(data) + 1
    end
    hclt_from_clt(clt, num_cats, num_hidden_cats, variable_groups; input_type)
end

function hclt_for_colored_images(data, nc, height, width, num_hidden_cats; num_cats = nothing, pseudocount = 0.1, 
                                 mode = "red_only", patch_size = 2, input_type = Categorical)
    num_examples = size(data, 1)
    
    if mode == "red_only"
        variable_groups = group_vars_by_grid(height, width, 1; patch_h = patch_size, patch_w = patch_size, global_offset = 0)
        num_mapped_vars = length(variable_groups)
        mapped_data = CUDA.zeros(eltype(data), num_examples, num_mapped_vars)
        @inbounds for idx = 1 : num_mapped_vars
            mapped_data[:,idx] .= data[:,variable_groups[idx][1]]
        end

        clt_edges = learn_chow_liu_tree(mapped_data; pseudocount, Float = Float32)
        clt = clt_edges2graphs(clt_edges; shape = :directed)

        if num_cats === nothing
            num_cats = maximum(data) + 1
        end
        pc, leaves = hclt_from_clt(clt, num_cats, num_hidden_cats, variable_groups; input_type, get_leaves = true)

    elseif mode == "green_only"
        variable_groups = group_vars_by_grid(height, width, 1; patch_h = patch_size, patch_w = patch_size, global_offset = height * width)
        num_mapped_vars = length(variable_groups)
        mapped_data = CUDA.zeros(eltype(data), num_examples, num_mapped_vars)
        @inbounds for idx = 1 : num_mapped_vars
            mapped_data[:,idx] .= data[:,variable_groups[idx][1]]
        end

        clt_edges = learn_chow_liu_tree(mapped_data; pseudocount, Float = Float32)
        clt = clt_edges2graphs(clt_edges; shape = :directed)

        if num_cats === nothing
            num_cats = maximum(data) + 1
        end
        pc, leaves = hclt_from_clt(clt, num_cats, num_hidden_cats, variable_groups; input_type, get_leaves = true)

    elseif mode == "blue_only"
        variable_groups = group_vars_by_grid(height, width, 1; patch_h = patch_size, patch_w = patch_size, global_offset = height * width * 2)
        num_mapped_vars = length(variable_groups)
        mapped_data = CUDA.zeros(eltype(data), num_examples, num_mapped_vars)
        @inbounds for idx = 1 : num_mapped_vars
            mapped_data[:,idx] .= data[:,variable_groups[idx][1]]
        end

        clt_edges = learn_chow_liu_tree(mapped_data; pseudocount, Float = Float32)
        clt = clt_edges2graphs(clt_edges; shape = :directed)

        if num_cats === nothing
            num_cats = maximum(data) + 1
        end
        pc, leaves = hclt_from_clt(clt, num_cats, num_hidden_cats, variable_groups; input_type, get_leaves = true)

    elseif mode == "cat_channels"
        variable_groups = group_vars_by_cgrid(height, width, nc; patch_h = patch_size, patch_w = patch_size, global_offset = 0)
        num_mapped_vars = length(variable_groups)
        mapped_data = CUDA.zeros(eltype(data), num_examples, num_mapped_vars)
        @inbounds for idx = 1 : num_mapped_vars
            mapped_data[:,idx] .= data[:,variable_groups[idx][1]]
        end

        clt_edges = learn_chow_liu_tree(mapped_data; pseudocount, Float = Float32)
        clt = clt_edges2graphs(clt_edges; shape = :directed)

        if num_cats === nothing
            num_cats = maximum(data) + 1
        end
        pc, leaves = hclt_from_clt(clt, num_cats, num_hidden_cats, variable_groups; input_type, get_leaves = true)
    else
        error("Not implemented")
    end

    pc, leaves
end

function hclt_from_clt(clt, num_cats, num_hidden_cats, variable_groups; input_type, get_leaves = false)
    num_mapped_vars = length(variable_groups)

    # meaning: `joined_leaves[i,j]` is a distribution of the hidden variable `i` having value `j` 
    # conditioned on the (mapped) observed variable `i`
    joined_leaves = categorical_leaves(num_mapped_vars, num_cats, num_hidden_cats, variable_groups, input_type)

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
                child_circuits = [summate(multiply([child_circuit[cat_idx] for child_circuit in child_circuits])) for cat_idx = 1 : num_hidden_cats]
            else
                child_circuits = child_circuits[1]
            end
            # Pr(X_1)...Pr(X_k) -> Pr(Y)Pr(X_1|Y)...Pr(X_k|Y)
            circuits = [summate(multiply.(child_circuits, joined_leaves[curr_node, :])) for cat_idx = 1 : num_hidden_cats]
            set_prop!(clt, curr_node, :circuits, circuits)
        end
    end
    
    pc = get_prop(clt, node_seq[end], :circuits)[1] # A ProbCircuit node

    if get_leaves
        pc, joined_leaves
    else
        pc
    end
end

function categorical_leaves(num_mapped_vars, num_cats, num_hidden_cats, variable_groups, input_type::Type{Categorical})
    leaves = Matrix{ProbCircuit}(undef, num_mapped_vars, num_hidden_cats)
    for idx = 1 : num_mapped_vars
        for rep = 1 : num_hidden_cats
            n = summate(multiply([PlainInputNode(var, Categorical(num_cats)) for var in variable_groups[idx][2]]...))
            leaves[idx, rep] = n
        end
    end
    leaves
end
function categorical_leaves(num_mapped_vars, num_cats, num_hidden_cats, variable_groups, input_type::Type{DiscreteLogistic})
    leaves = Matrix{ProbCircuit}(undef, num_mapped_vars, num_hidden_cats)
    for idx = 1 : num_mapped_vars
        for rep = 1 : num_hidden_cats
            ins = Vector{ProbCircuit}()
            for var in variable_groups[idx][2]
                node = summate([PlainInputNode(var, DiscreteLogistic(255)) for j = 1 : 10]...)
                push!(ins, node)
            end
            n = summate(multiply(ins...))
            leaves[idx, rep] = n
        end
    end
    leaves
end


function diff_categorical_leaves(num_mapped_vars, num_cats, num_hidden_cats, variable_groups, input_type::Type{DiscreteLogistic})
    leaves = Matrix{ProbCircuit}(undef, num_mapped_vars, num_hidden_cats)
    for idx = 1 : num_mapped_vars
        for rep = 1 : num_hidden_cats
            ins = Vector{ProbCircuit}()
            for var in variable_groups[idx][2]
                node = summate([PlainInputNode(var, DiscreteLogistic(255)) for j = 1 : 10]...)
                push!(ins, node)
            end
            n = summate(multiply(ins...))
            leaves[idx, rep] = n
        end
    end
    leaves
end

function group_vars_by_grid(height, width, cdepth; patch_h = 2, patch_w = 2, global_offset = 0)
    @assert height % patch_h == 0
    @assert width % patch_w == 0

    hw2var(h, w, offset) = Var((h - 1) * width + w + offset + global_offset)

    variable_groups = Vector{Tuple{Var,Set{Var}}}()
    for c = 1 : cdepth
        c_offset = height * width * (c - 1)
        for h = 1 : patch_h : height
            for w = 1 : patch_w : width
                first_var = hw2var(h, w, c_offset)
                var_set = Set{Var}()
                for x = 0 : patch_h - 1
                    for y = 0 : patch_w - 1
                        push!(var_set, hw2var(h+x, w+y, c_offset))
                    end
                end
                push!(variable_groups, (first_var, var_set))
            end
        end
    end

    all_variables = Set{Var}()
    for var_group in variable_groups
        @assert length(intersect(all_variables, var_group[2])) == 0
        union!(all_variables, var_group[2])
    end
    for i = 1 : height * width * cdepth
        @assert Var(global_offset + i) in all_variables
    end

    variable_groups
end

function group_vars_by_cgrid(height, width, cdepth; patch_h = 2, patch_w = 2, global_offset = 0)
    @assert height % patch_h == 0
    @assert width % patch_w == 0

    hw2var(h, w, offset) = Var((h - 1) * width + w + offset + global_offset)

    variable_groups = Vector{Tuple{Var,Set{Var}}}()
    for h = 1 : patch_h : height
        for w = 1 : patch_w : width
            first_var = hw2var(h, w, 0)
            var_set = Set{Var}()
            for x = 0 : patch_h - 1
                for y = 0 : patch_w - 1
                    for c = 1 : cdepth
                        c_offset = height * width * (c - 1)
                        push!(var_set, hw2var(h+x, w+y, c_offset))
                    end
                end
            end
            push!(variable_groups, (first_var, var_set))
        end
    end

    all_variables = Set{Var}()
    for var_group in variable_groups
        @assert length(intersect(all_variables, var_group[2])) == 0
        union!(all_variables, var_group[2])
    end
    for i = 1 : height * width * cdepth
        @assert Var(global_offset + i) in all_variables
    end

    variable_groups
end