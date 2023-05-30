

abstract type Vtree end

mutable struct VtreeInnerNode <: Vtree
    variables::BitSet
    children::Vector{Vtree}
end

VtreeInnerNode(children::Vector{<:Vtree}) = begin
    variables = BitSet()
    chs = Vector{Vtree}()
    for ch in children
        union!(variables, ch.variables)
        push!(chs, ch)
    end
    VtreeInnerNode(variables, chs)
end

mutable struct VtreeLeafNode <: Vtree
    variables::BitSet
end