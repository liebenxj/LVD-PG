using ProbabilisticCircuits: InputDist
using Random: rand


"A Gaussian input distribution"
struct Gaussian <: InputDist
    mu::Float32
    sigma::Float32
    Gaussian(mu::Float32 = zero(Float32), sigma::Float32 = one(Float32)) =
        new(mu, sigma)
end

import ProbabilisticCircuits: params, num_parameters # extend

mu(d::Gaussian) = d.mu
sigma(d::Gaussian) = d.sigma

params(d::Gaussian) = (mu(d), sigma(d))

num_parameters(n::Gaussian, independent::Bool = false) = 2

param_buffer_size(n::Gaussian) = 2

import ProbabilisticCircuits: init_params, loglikelihood # extend

init_params(d::Gaussian, perturbation) = begin
    mu = rand(Float32) * Float32(perturbation)
    sigma = one(Float32)
    Gaussian(mu, sigma)
end

loglikelihood(d::Gaussian, x, _ = nothing) =
    -( (x-d.mu)^2/(2*d.sigma^2) + Float32(0.5) * log(2*π) + log(d.sigma) )

struct BitsGaussian <: InputDist
    heap_start::UInt32
end

import ProbabilisticCircuits: bits, unbits, flow, update_params, clear_memory # extend

function bits(d::Gaussian, heap)
    heap_start = length(heap) + 1
    push!(heap, mu(d), sigma(d))
    append!(heap, zeros(Float32, 3)) # 3 statistics: \sum_{i} w_i, \sum_{i} w_i * x_i, \sum_{i} w_i * (x_i)^2
    BitsGaussian(heap_start)
end

function unbits(d::BitsGaussian, heap) 
    mu = heap[d.heap_start]
    sigma = heap[d.heap_start + 1]
    Gaussian(mu, sigma)
end

num_parameters(d::BitsGaussian, independent::Bool = false) = 2

loglikelihood(d::BitsGaussian, x, heap) = begin
    mu = heap[d.heap_start]
    sigma = heap[d.heap_start + 1]
    -( (x-mu)^2/(2*sigma^2) + Float32(0.5) * log(2*π) + log(sigma) )
end

loglikelihood(d::BitsGaussian, x, heap, params, ex_id, start_idx) = begin
    mu = params[ex_id, start_idx]
    sigma = params[ex_id, start_idx + 1]
    -( (x-mu)^2/(2*sigma^2) + Float32(0.5) * log(2*π) + log(sigma) )
end

function flow(d::BitsGaussian, x, node_flow, heap)
    heap_start = d.heap_start
    CUDA.@atomic heap[heap_start + 2] += node_flow
    CUDA.@atomic heap[heap_start + 3] += node_flow * x
    CUDA.@atomic heap[heap_start + 4] += node_flow * x^2
    nothing
end

function get_edge_flow(d::BitsGaussian, x, node_flow, param_id, heap)
    if param_id == 1
        node_flow
    elseif param_id == 2
        node_flow * x
    elseif param_id == 3
        node_flow * x^2
    else
        @assert false
    end
end

get_param(d::BitsGaussian, idx, heap) = 
    heap[d.heap_start + UInt32(idx) - one(UInt32)]

set_param(d::BitsGaussian, idx, param, heap) = begin
    heap[d.heap_start + UInt32(idx) - one(UInt32)] = param
    nothing
end

kl_div(d::BitsGaussian, heap, params1, params2, ex_id, start_idx, norm_params::Bool = true) = begin
    mu1 = params1[ex_id, start_idx]
    mu2 = params2[ex_id, start_idx]
    sigma1 = params1[ex_id, start_idx + 1]
    sigma2 = params2[ex_id, start_idx + 1]
    log(sigma2 / sigma1) + (sigma1^2 + (mu1 - mu2)^2) / (2 * sigma2^2) - Float32(0.5)
end

function get_edge_kld_grad(d::BitsGaussian, node_grad, params1, params2, ex_id, start_idx, norm_params::Bool = false)
    mu1 = params1[ex_id, start_idx]
    mu2 = params2[ex_id, start_idx]
    sigma1 = params1[ex_id, start_idx + 1]
    sigma2 = params2[ex_id, start_idx + 1]
    
    dmu1 = (mu1 - mu2) / sigma2^2
    dmu2 = -dmu1
    dsigma1 = sigma1 / sigma2^2 - 1 / sigma1
    dsigma2 = 1 / sigma2 - sigma1^2 / sigma2^3

    dmu1 *= node_grad
    dmu2 *= node_grad
    dsigma1 *= node_grad
    dsigma2 *= node_grad

    dmu1, dmu2, dsigma1, dsigma2
end

function clear_memory(d::BitsGaussian, heap, rate)
    heap_start = d.heap_start
    heap[heap_start + 2] += zero(Float32)
    heap[heap_start + 3] += zero(Float32)
    heap[heap_start + 4] += zero(Float32)
    nothing
end