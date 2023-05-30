using ProbabilisticCircuits: PlainInputNode, Var, foldup_aggregate, BitsInput


function num_variables(pc::ProbCircuit)
    vars = Set{Var}()
    foreach(pc) do n
        if n isa PlainInputNode
            push!(vars, randvar(n))
        end
    end
    length(vars)
end
function num_variables(mbpc::CuMetaBitsProbCircuit)
    vars = Set{Var}()
    nodes = Array(mbpc.bpc.nodes)
    for n in nodes
        if n isa BitsInput
            push!(vars, n.variable)
        end
    end
    length(vars)
end

function convert_to_latent_pc(pc::ProbCircuit; eps = 1e-6, get_mapping = false)
    num_vars = num_variables(pc)
    num_inputs_per_var = zeros(Int32, num_vars)
    foreach(pc) do n
        if n isa PlainInputNode
            num_inputs_per_var[randvar(n)] += 1
        end
    end

    old2new = Dict{ProbCircuit,ProbCircuit}()

    input_node_counts = zeros(Int32, num_vars)
    f_i(n) = begin
        v = randvar(n)
        input_node_counts[v] += 1
        num_cats = num_inputs_per_var[v]
        d = FixableCategorical(num_cats, true)
        d.logps .= eps / num_cats
        d.logps[input_node_counts[v]] += (1 - eps)
        d.logps .= log.(d.logps)
        PlainInputNode(v, d)
    end
    f_m(n, ins) = begin
        new_n = multiply(ins...)
        old2new[n] = new_n
        new_n
    end
    f_s(n, ins) = begin
        new_n = summate(ins...)
        new_n.params .= n.params
        old2new[n] = new_n
        new_n
    end
    new_pc = foldup_aggregate(pc, f_i, f_m, f_s, ProbCircuit)

    if get_mapping
        new_pc, old2new
    else
        new_pc
    end
end