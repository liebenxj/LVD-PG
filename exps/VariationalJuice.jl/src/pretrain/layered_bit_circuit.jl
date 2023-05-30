using ProbabilisticCircuits: FlatVectors


struct CuLayeredBitsProbCircuit 

    bpc::CuBitsProbCircuit

    node_layer_head::FlatVectors{<:CuVector{Int}}

    up_layer_end_ids::Vector{Int}

    down_layer_start_ids::Vector{Int}

end

function CuLayeredBitsProbCircuit(pc::ProbCircuit; min_group_nvars = 5)

    bpc = BitsProbCircuit(pc)

    upedgelayers = bpc.edge_layers_up.vectors
    downedgelayers = bpc.edge_layers_down.vectors
    nodes_map = bpc.nodes_map
    nodes_dict = Dict(zip(nodes_map, collect(1:length(nodes_map))))

    num_up_layers = length(bpc.edge_layers_up.ends)
    num_down_layers = length(bpc.edge_layers_down.ends)

    varsets, vars2headnodes, vtree = infer_vtree_and_heads(pc)
    varsets = filter(vs->(length(vs) >= min_group_nvars), varsets)

    layerheads = Vector{Int}[]
    up_layer_end_ids = Vector{Int}()
    for varset in varsets
        nodeset = Set{Int}()
        for n in vars2headnodes[varset]
            if haskey(nodes_dict, n)
                push!(nodeset, nodes_dict[n])
            else
                if num_inputs(n) == 1 && haskey(nodes_dict, n.inputs[1])
                    push!(nodeset, nodes_dict[n.inputs[1]])
                else
                    error("Not implemented.")
                end
            end
        end
        push!(layerheads, collect(nodeset))

        up_layer_end_id = num_up_layers
        while up_layer_end_id >= 1
            layer_start = up_layer_end_id == 1 ? 1 : bpc.edge_layers_up.ends[up_layer_end_id-1] + 1
            layer_end = bpc.edge_layers_up.ends[up_layer_end_id]
            layer_flag = false
            for edge_id = layer_start : layer_end
                if upedgelayers[edge_id].parent_id in nodeset
                    layer_flag = true
                    break
                end
            end
            if layer_flag
                break
            end
            up_layer_end_id -= 1
        end
        push!(up_layer_end_ids, up_layer_end_id)
    end
    node_layer_head = FlatVectors(layerheads)

    down_layer_start_ids = Vector{Int}()
    for layer_id = 1 : length(varsets)
        nodeset = Set(layerheads[layer_id])
        curr_layer_id = 0
        for down_layer_id = num_down_layers : -1 : 1
            layer_start = down_layer_id == 1 ? 1 : bpc.edge_layers_down.ends[down_layer_id-1] + 1
            layer_end = bpc.edge_layers_down.ends[down_layer_id]
            for edge_id = layer_start : layer_end
                if downedgelayers[edge_id].parent_id in nodeset
                    curr_layer_id = down_layer_id
                    break
                end
            end
            if curr_layer_id != 0
                break
            end
        end
        push!(down_layer_start_ids, curr_layer_id)
    end

    CuLayeredBitsProbCircuit(cu(bpc), cu(node_layer_head), up_layer_end_ids, down_layer_start_ids)

end

function infer_vtree_and_heads(pc::ProbCircuit)
    vars2vtree = Dict{BitSet,Vtree}()
    vars2headnodes = Dict{BitSet,Set{ProbCircuit}}()

    f_i(n) = begin
        vars = BitSet(randvar(n))
        if !haskey(vars2vtree, vars)
            vars2vtree[vars] = VtreeLeafNode(vars)
            vars2headnodes[vars] = Set{ProbCircuit}()
        end
        push!(vars2headnodes[vars], n)
        vars
    end
    f_m(n, ins) = begin
        vars = reduce(union, ins)
        if !haskey(vars2vtree, vars)
            vs = map(ins) do ch_vars
                vars2vtree[ch_vars]
            end
            vars2vtree[vars] = VtreeInnerNode(vs)
            vars2headnodes[vars] = Set{ProbCircuit}()
        end
        push!(vars2headnodes[vars], n)
        for c in inputs(n)
            @assert !(c in vars2headnodes[vars])
        end
        vars
    end
    f_s(n, ins) = begin
        vars = reduce(union, ins)
        @assert haskey(vars2vtree, vars)
        @assert haskey(vars2headnodes, vars)
        push!(vars2headnodes[vars], n)
        if n === pc
            println(length(vars))
        end
        for c in inputs(n)
            delete!(vars2headnodes[vars], c)
        end
        vars
    end
    root_vars = foldup_aggregate(pc, f_i, f_m, f_s, BitSet)

    vtree = vars2vtree[root_vars] # root vtree node
    varsets = bottom_up_varset_order(vtree)

    varsets, vars2headnodes, vtree
end

function bottom_up_varset_order(vtree::Vtree)
    varsets = Vector{BitSet}()

    dfs(v::Vtree) = begin
        if v isa VtreeInnerNode
            for cv in v.children
                dfs(cv)
            end
            push!(varsets, v.variables)
        end
    end
    dfs(vtree)

    varsets
end