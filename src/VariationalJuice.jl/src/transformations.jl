using ProbabilisticCircuits: JpcFormat, Var, PlainSumNode


function load_and_transform_pc_with_mapping(file_name::String = "models/mnist_4.jpc")
    pc2 = read(file_name, ProbCircuit, JpcFormat(), true)
    pc1, pc2topc1 = convert_to_latent_pc(pc2; get_mapping = true)
    pc1, pc2, pc2topc1
end

function load_and_transform_to_latent_pc(file_name::String; mode = "inputs", variable_groups = nothing,
                                         get_p_x_z = false)
    pc2 = read(file_name, ProbCircuit, JpcFormat(), true)

    z_nodes, node_mapping = select_latent_variables(pc2; mode, variable_groups)
    pc1, pc2topc1 = convert_to_latent_pc(pc2, z_nodes, node_mapping; get_mapping = true)

    if get_p_x_z
        z_nodes = mapreduce(permutedims, vcat, z_nodes)
        pc3 = bind_pc(z_nodes)

        pc1, pc2, pc2topc1, pc3, z_nodes
    else
        pc1, pc2, pc2topc1
    end
end

function to_gpu(pc1::ProbCircuit, pc2::ProbCircuit, pc2topc1)
    mbpc1 = CuMetaBitsProbCircuit(pc1)
    mbpc2 = CuMetaBitsProbCircuit(pc2)
    mbpc_mapping = compute_mbpc_mapping(mbpc1, mbpc2, pc2topc1)
    mbpc1, mbpc2, mbpc_mapping
end

#############################
## Select latent variables ##
#############################

function select_latent_variables(pc::ProbCircuit; mode = "inputs", variable_groups = nothing)
    if mode == "inputs"
        z_nodes = Vector{ProbCircuit}[Vector{ProbCircuit}() for i = 1 : num_variables(pc)]
        node_mapping = Dict{ProbCircuit,Tuple{Int,Int}}()
        foreach(pc) do n
            if isinput(n)
                var_idx = randvar(n)
                push!(z_nodes[var_idx], n)
                node_mapping[n] = (var_idx, length(z_nodes[var_idx]))
            end
        end
    elseif mode == "group"
        z_nodes = Vector{ProbCircuit}[Vector{ProbCircuit}() for i = 1 : length(variable_groups)]
        node_mapping = Dict{ProbCircuit,Tuple{Int,Int}}()

        scope_mapping = Dict()
        for i = 1 : length(variable_groups)
            minvar_in_scope = minimum(variable_groups[i][2])
            scope_mapping[minvar_in_scope] = i
        end

        foreach(pc) do n
            if issum(n) && isinput(n.inputs[1].inputs[1])
                minvar_in_scope = minimum(randvars(n))
                idx = scope_mapping[minvar_in_scope]
                push!(z_nodes[idx], n)
                node_mapping[n] = (idx, length(z_nodes[idx]))
            end
        end
    else
        error("Unknown mode")
    end
    z_nodes, node_mapping
end

function convert_to_latent_pc(pc::ProbCircuit, z_nodes, node_mapping; eps = 1e-6, get_mapping = false)

    old2new = Dict{ProbCircuit,ProbCircuit}()

    f_i(n) = begin
        if haskey(node_mapping, n)
            v, i = node_mapping[n]
            num_cats = length(z_nodes[v])
            d = FixableCategorical(num_cats, true)
            d.logps .= eps / num_cats
            d.logps[i] += (1 - eps)
            d.logps .= log.(d.logps)
            PlainInputNode(v, d), true
        else
            n, false
        end
    end
    f_m(n, ins) = begin
        ch_nodes = first.(ins)
        flag = all(last.(ins))
        if haskey(node_mapping, n)
            @assert false "Does not support mapping of product nodes"
        else
            new_n = multiply(ch_nodes...)
        end
        if flag 
            old2new[n] = new_n
        end
        new_n, flag
    end
    f_s(n, ins) = begin
        ch_nodes = first.(ins)
        flag = all(last.(ins))
        if haskey(node_mapping, n)
            v, i = node_mapping[n]
            num_cats = length(z_nodes[v])
            d = FixableCategorical(num_cats, true)
            d.logps .= eps / num_cats
            d.logps[i] += (1 - eps)
            d.logps .= log.(d.logps)
            new_n = PlainInputNode(v, d)
            flag = true
        else
            new_n = summate(ch_nodes...)
            new_n.params .= n.params
            if flag 
                old2new[n] = new_n
            end
        end
        new_n, flag
    end
    new_pc = foldup_aggregate(pc, f_i, f_m, f_s, Tuple{ProbCircuit,Bool})[1]

    if get_mapping
        new_pc, old2new
    else
        new_pc
    end
end

######################
## Bind ProbCircuit ##
######################

function bind_pc(nodes::Matrix)
    sum_nodes = Vector{ProbCircuit}()
    for i = 1 : size(nodes, 1)
        n = summate(nodes[i, :]...)
        push!(sum_nodes, n)
    end
    multiply(sum_nodes...)
end