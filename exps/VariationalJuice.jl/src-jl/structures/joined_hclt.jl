using MetaGraphs: MetaDiGraph, outneighbors, get_prop, set_prop!
using CUDA
using ProbabilisticCircuits
using ChowLiuTrees: learn_chow_liu_tree
using Printf

import Base: hash # extend

function myhash(rnode::PartitionNode)
    ch_scopes = map(rn -> rn.scope, rnode.children)
    first_ids = map(s -> first(s), ch_scopes)
    ch_scopes = ch_scopes[sortperm(first_ids)]
    hash(ch_scopes)
end

function clts2rgraph(clts::Vector{MetaDiGraph}, num_vars::Integer)
    scope2rnode = Dict{BitSet,RegionGraph}()
    for clt in clts
        var_seq = PCs.bottom_up_order(clt)
        for curr_var in var_seq
            ch_vars = outneighbors(clt, curr_var)

            rnode = if length(ch_vars) == 0
                scope = BitSet([curr_var])
                if !(scope in keys(scope2rnode))
                    scope2rnode[scope] = InputRegionNode(0, scope)
                end
                scope2rnode[scope]
            else
                scope = mapreduce(ch_var -> get_prop(clt, ch_var, :rnode).scope, union, ch_vars)
                scope = union(scope, BitSet([curr_var]))
                ch_rnodes = Vector{RegionGraph}(undef, length(ch_vars))
                map!(ch_var -> get_prop(clt, ch_var, :rnode), ch_rnodes, ch_vars)
                in_rnode = get!(scope2rnode, BitSet([curr_var])) do
                    InputRegionNode(0, BitSet([curr_var]))
                end
                push!(ch_rnodes, in_rnode)
                
                if scope in keys(scope2rnode)
                    inner_rn = scope2rnode[scope]
                    redundant = false
                    p_rnode = PartitionNode(0, ch_rnodes)
                    for i = 1 : length(inner_rn.children)
                        if myhash(inner_rn.children[i]) == myhash(p_rnode)
                            redundant = true
                            break
                        end
                    end
                    if !redundant
                        push!(inner_rn.children, p_rnode)
                    end
                    inner_rn
                else
                    rnode = InnerRegionNode(0, [PartitionNode(0, ch_rnodes)])
                    scope2rnode[scope] = rnode
                    rnode
                end
            end
            set_prop!(clt, curr_var, :rnode, rnode)
        end
    end
    scope2rnode[BitSet(collect(1:num_vars))]
end

function joined_hclt(datasets::Vector, num_hidden_cats; num_cats = nothing, shape = :directed,
                     input_type = Literal, pseudocount = 0.1)

    num_vars = size(datasets[1], 2)

    # Get all CLTs
    println("> Constructing CLTs...")
    clts = Vector{MetaDiGraph}()
    for (i, data) in enumerate(datasets)
        print(@sprintf("  - CLT #%03d/%03d... ", i, length(datasets)))
        t = @elapsed begin
            if data isa Array
                data = cu(data)
            end
            clt_edges = learn_chow_liu_tree(data; pseudocount = pseudocount, Float = Float32)
            clt = PCs.clt_edges2graphs(clt_edges; shape)
            push!(clts, clt)
        end
        println(@sprintf("done (%.2fs)", t))
    end

    # Construct region graph
    rnode = clts2rgraph(clts, num_vars)

    # Region graph 
    f_input(rn)::Vector{<:ProbCircuit} = begin
        if input_type == Categorical
            [PlainInputNode(randvar(rn), Categorical(num_cats)) for _ = 1 : num_hidden_cats]
        else
            error("Unknown input type $(input_type).")
        end
    end
    f_partition(rn, ins)::Vector{<:ProbCircuit} = begin
        [multiply([cs[i] for cs in ins]...) for i = 1 : num_hidden_cats]
    end
    f_inner(rn, ins)::Vector{<:ProbCircuit} = begin
        flattened_ins = reduce(vcat, ins)
        [summate(flattened_ins...) for _ = 1 : num_hidden_cats]
    end
    foldup_aggregate(rnode, f_input, f_partition, f_inner, Vector{<:ProbCircuit})
end