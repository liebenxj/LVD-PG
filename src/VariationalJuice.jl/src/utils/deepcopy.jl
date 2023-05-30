
import Main: deepcopy # extend

function deepcopy(n::ProbCircuit; depth, copy_leaves = false)
    if isinput(n)
        if copy_leaves
            PlainInputNode(randvar(n), deepcopy(dist(n)))
        else
            n
        end
    elseif depth == 0
        n
    else
        new_chs = map(inputs(n)) do c
            deepcopy(c; depth = depth - 1, copy_leaves)
        end
        if issum(n)
            new_n = summate(new_chs)
            new_n.params .= n.params
        else 
            @assert ismul(n)
            new_n = multiply(new_chs)
        end
        new_n
    end
end