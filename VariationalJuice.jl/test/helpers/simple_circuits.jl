include("../../src/VariationalJuice.jl")


function simple_3vars_circuit(; num_cats = 5, fixed = true)
    n11 = PCs.PlainInputNode(1, FixableCategorical(num_cats, fixed))
    n12 = PCs.PlainInputNode(1, FixableCategorical(num_cats, fixed))
    n21 = PCs.PlainInputNode(2, FixableCategorical(num_cats, fixed))
    n22 = PCs.PlainInputNode(2, FixableCategorical(num_cats, fixed))
    n31 = PCs.PlainInputNode(3, FixableCategorical(num_cats, fixed))
    n32 = PCs.PlainInputNode(3, FixableCategorical(num_cats, fixed))

    pc = summate(multiply(n11, n21, n31), multiply(n12, n22, n32))
    pc
end

function simple_3vars_cat_circuit(; num_cats = 5)
    n11 = PCs.PlainInputNode(1, Categorical(num_cats))
    n12 = PCs.PlainInputNode(1, Categorical(num_cats))
    n21 = PCs.PlainInputNode(2, Categorical(num_cats))
    n22 = PCs.PlainInputNode(2, Categorical(num_cats))
    n31 = PCs.PlainInputNode(3, Categorical(num_cats))
    n32 = PCs.PlainInputNode(3, Categorical(num_cats))

    pc = summate(multiply(n11, n21, n31), multiply(n12, n22, n32))
    pc
end

function simple_gaussian_circuit()
    n11 = PCs.PlainInputNode(1, Gaussian())
    n12 = PCs.PlainInputNode(1, Gaussian())
    n21 = PCs.PlainInputNode(2, Gaussian())
    n22 = PCs.PlainInputNode(2, Gaussian())
    n31 = PCs.PlainInputNode(3, Gaussian())
    n32 = PCs.PlainInputNode(3, Gaussian())

    pc = summate(multiply(n11, n21, n31), multiply(n12, n22, n32))
    pc
end

function fully_factorized_categorical_fixed(; num_vars = 3, num_cats = 8, eps = 1e-6)
    inputs = map(1:num_vars) do i
        ins = map(1:num_cats) do j
            d = FixableCategorical(num_cats, true)
            d.logps .= eps / num_cats
            d.logps[j] += 1 - eps
            d.logps .= log.(d.logps)
            PCs.PlainInputNode(i, d)
        end
        summate(ins...)
    end
    pc = summate(multiply(inputs...))
end

function simple_multihead_circuit(; num_cats = 5)
    n11 = PCs.PlainInputNode(1, Categorical(num_cats))
    n12 = PCs.PlainInputNode(1, Categorical(num_cats))
    n21 = PCs.PlainInputNode(2, Categorical(num_cats))
    n22 = PCs.PlainInputNode(2, Categorical(num_cats))
    n31 = PCs.PlainInputNode(3, Categorical(num_cats))
    n32 = PCs.PlainInputNode(3, Categorical(num_cats))

    n1 = multiply(n11, n21, n31)
    n2 = multiply(n12, n22, n32)
    pcs = [summate(n1, n2), summate(n1, n2)]
    pcs
end