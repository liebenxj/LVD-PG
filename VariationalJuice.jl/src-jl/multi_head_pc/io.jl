
function write_mhpc(file::String, pcs::Vector{T}) where T <: ProbCircuit
    write(file, summate(pcs...))
end

function read_mhpc(file::String)
    pc = read(file, ProbCircuit)
    children(pc)
end