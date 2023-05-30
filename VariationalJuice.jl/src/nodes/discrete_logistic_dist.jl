using ProbabilisticCircuits: InputDist

import ProbabilisticCircuits: num_parameters, init_params, loglikelihood, bits, unbits,
    flow, update_params, clear_memory


"Discrete Logistic distribution"
struct DiscreteLogistic <: InputDist
    mean::Float32
    logscale::Float32
    bin_size::Float32
end

DiscreteLogistic(num_bins; mean = 0.0, logscale = 0.0) = 
    DiscreteLogistic(Float32(mean), Float32(logscale), Float32(1 / num_bins))

num_parameters(n::DiscreteLogistic, _) = 2

init_params(d::DiscreteLogistic, perturbation::Float32) = begin
    mean = rand(Float32)
    logscale = -rand(Float32) * Float32(4.0)
    DiscreteLogistic(mean, logscale, d.bin_size)
end

sigmoid(x::Float32) = one(Float32) / (one(Float32) + exp(-x))

loglikelihood(d::DiscreteLogistic, value, _ = nothing) = begin
    invscale = exp(-d.logscale)
    x_plus = invscale * (value - d.mean + 0.5 * d.bin_size)
    x_minus = invscale * (value - d.mean - 0.5 * d.bin_size)

    # cdf_delta = sigmoid(x_plus) - sigmoid(x_minus)
    cdf_delta = one(Float32) / (one(Float32) + exp(-x_plus)) - one(Float32) / (one(Float32) + exp(-x_minus))

    log(max(cdf_delta, 1e-8))
end

"Bits representation of the Discrete Logistic distribution"
struct BitsDiscreteLogistic <: InputDist
    bin_size::Float32
    heap_start::UInt32
end

function bits(d::DiscreteLogistic, heap)
    heap_start = length(heap) + 1
    # use heap to allocate space for parameter learning
    append!(heap, [d.mean, d.logscale], zeros(eltype(heap), 3))
    BitsDiscreteLogistic(d.bin_size, heap_start)
end

function unbits(d::BitsDiscreteLogistic, heap)
    mean = heap[d.heap_start]
    logscale = heap[d.heap_start+1]
    DiscreteLogistic(mean, logscale, d.bin_size)
end

loglikelihood(d::BitsDiscreteLogistic, value, heap) = begin
    mean = heap[d.heap_start]
    logscale = heap[d.heap_start+1]

    invscale = exp(-logscale)
    x_plus = invscale * (value - mean + 0.5 * d.bin_size)
    x_minus = invscale * (value - mean - 0.5 * d.bin_size)

    # cdf_delta = sigmoid(x_plus) - sigmoid(x_minus)
    cdf_delta = one(Float32) / (one(Float32) + exp(-x_plus)) - one(Float32) / (one(Float32) + exp(-x_minus))

    log(max(cdf_delta, 1e-8))
end

function flow(d::BitsDiscreteLogistic, value, node_flow, heap)
    if ismissing(value)
        @assert false "Not implemented"
    else
        CUDA.@atomic heap[d.heap_start+2] += node_flow * value
        CUDA.@atomic heap[d.heap_start+3] += node_flow * value^2
        CUDA.@atomic heap[d.heap_start+4] += node_flow
    end
end

function update_params(d::BitsDiscreteLogistic, heap, pseudocount, inertia)
    heap_start = d.heap_start

    @inbounds begin
        mean = heap[heap_start+2] / heap[heap_start+4]
        var = heap[heap_start+3] / heap[heap_start+4] - mean^2
        scale = sqrt(Float32(3.0) * var) / Float32(π)
        logscale = log(max(scale, 1e-8))

        heap[heap_start] = mean
        heap[heap_start+1] = logscale
    end
    nothing
end

function clear_memory(d::BitsDiscreteLogistic, heap, rate)
    heap_start = d.heap_start
    heap[heap_start+2] *= rate
    heap[heap_start+3] *= rate
    heap[heap_start+4] *= rate
    nothing
end