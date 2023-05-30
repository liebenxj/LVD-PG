using ProbabilisticCircuits: RegionGraph
using DirectedAcyclicGraphs
using JSON


struct PartitionNode <: RegionGraph
    node_id::Int
    scope::BitSet
    children::AbstractVector{T} where T <: RegionGraph
    PartitionNode(nid::Int, children::AbstractVector{T}) where T <: RegionGraph = begin
        s = BitSet()
        for ch in children
            @assert length(intersect(s, ch.scope)) == 0
            s = union(s, ch.scope)
        end
        new(nid, s, children)
    end
end

struct InnerRegionNode <: RegionGraph
    node_id::Int
    scope::BitSet
    children::AbstractVector{T} where T <: RegionGraph
    InnerRegionNode(nid::Int, children::AbstractVector{T}) where T <: RegionGraph = begin
        s = deepcopy(children[1].scope)
        for ch in children[2:end]
            @assert s == ch.scope
        end
        new(nid, s, children)
    end
end

struct InputRegionNode <: RegionGraph
    node_id::Int
    scope::BitSet
    InputRegionNode(nid::Int, scope::AbstractVector{T}) where T <: Integer = begin
        new(nid, BitSet(scope))
    end
    InputRegionNode(nid::Int, scope::BitSet) = begin
        new(nid, scope)
    end
end

#########
## I/O
#########

function parse_rg_from_file(file_name::String)
    d = JSON.parse(read(file_name, String))
    num_nodes = length(d)

    id2rnode = Dict{Int,RegionGraph}()
    for node_id = 0 : num_nodes - 1
        node_type = d[node_id]["node_type"]
        if node_type == "partition"
            ch_ids = d[node_id]["children"]
            chs = [id2rnode[id] for id in ch_ids]
            rn = PartitionNode(node_id + 1, chs)
        elseif node_type == "inner"
            ch_ids = d[node_id]["children"]
            chs = [id2rnode[id] for id in ch_ids]
            rn = InnerRegionNode(node_id + 1, chs)
        elseif node_type == "input"
            scope = d[node_id]["scope"]
            rn = InputRegionNode(node_id + 1, scope .+ 1)
        else
            error("Unknown node type $(node_type).")
        end
        id2rnode[node_id] = rn
    end

    id2rnode[num_nodes-1]
end

##############
## Methods
##############

import ProbabilisticCircuits: children, num_children, isinner, randvar, randvars # extend
import DirectedAcyclicGraphs: NodeType # extend

children(node::Union{PartitionNode,InnerRegionNode}) = node.children
children(node::InputRegionNode) = []

children(node::Union{PartitionNode,InnerRegionNode}) = length(node.children)
children(node::InputRegionNode) = 0

isinner(node::Union{PartitionNode,InnerRegionNode}) = true
isinner(node::InputRegionNode) = false

randvar(node::InputRegionNode) = first(node.scope)
randvars(node::RegionGraph) = collect(node.scope)

NodeType(node::Union{PartitionNode,InnerRegionNode}) = DirectedAcyclicGraphs.Inner
NodeType(node::InputRegionNode) = DirectedAcyclicGraphs.Leaf

##############
## Traverse
##############

import ProbabilisticCircuits: foreach # extend

function foreach(f::Function, node::RegionGraph, ::Nothing = nothing)
    foreach(f, node, Dict{RegionGraph,Nothing}())
end

function foreach(f::Function, node::RegionGraph, seen)
    get!(seen, node) do
        for c in node.children
            foreach(f, c, seen)
        end
        f(node)
        nothing
    end
    nothing
end

import ProbabilisticCircuits: foldup_aggregate # extend

function foldup_aggregate(node::RegionGraph, f_input::Function, f_partition::Function, f_inner::Function, ::Type{T}, ::Nothing = nothing) where {T}
    foldup_aggregate(node, f_input, f_partition, f_inner, T, Dict{RegionGraph,T}())
end

function foldup_aggregate(node::RegionGraph, f_input::Function, f_partition::Function, f_inner::Function, ::Type{T}, cache) where {T}
    get!(cache, node) do
        if node isa PartitionNode || node isa InnerRegionNode
            child_values = Vector{T}(undef, length(node.children))
            map!(c -> foldup_aggregate(c, f_input, f_partition, f_inner, T, cache)::T,
                 child_values, node.children)
            
            if node isa PartitionNode
                f_partition(node, child_values)::T
            else
                f_inner(node, child_values)::T
            end
        else
            f_input(node)::T
        end
    end
end