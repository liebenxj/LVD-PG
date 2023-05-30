using Test

include("../../src/VariationalJuice.jl")


@testset "discrete logistic dist" begin

    n1 = PCs.PlainInputNode(1, DiscreteLogistic(10));
    n2 = PCs.PlainInputNode(1, DiscreteLogistic(10));

    pc = summate(n1, n2)
    init_parameters(pc; perturbation = 0.4)

    data = Matrix{Float32}(undef, 2, 1);
    data[1,1] = 0.3
    data[2,1] = 0.6

    bpc = CuBitsProbCircuit(pc);
    full_batch_em(bpc, cu(data), 10; batch_size = 2, pseudocount = 0.1);

    update_parameters(bpc)

    # init_parameters(pc; perturbation = 0.4)

    @test (pc.inputs[1].dist.mean < 0.5 && pc.inputs[2].dist.mean > 0.5) || (pc.inputs[1].dist.mean > 0.5 && pc.inputs[2].dist.mean < 0.5)

    #=lls = Vector{Float32}()
    for i = 0.0 : 0.1 : 1.0
        push!(lls, loglikelihood(pc.inputs[1].dist, i))
    end
    lls

    loglikelihood(pc.inputs[1].dist, 0.6)
    loglikelihood(pc.inputs[2].dist, 0.3)
    loglikelihood(pc.inputs[2].dist, 0.6)=#

end