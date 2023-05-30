using ProbabilisticCircuits: InputDist, loguniform, BitsCategorical


"A N-value categorical input distribution ranging over integers [0...N-1]"
struct FixableCategorical <: InputDist
    logps::Vector{Float32}
    fixed::Bool
end

FixableCategorical(num_cats::Integer, fixed::Bool) =
    FixableCategorical(loguniform(num_cats), fixed)

import ProbabilisticCircuits: logps, params, num_categories, num_parameters # extend

logps(d::FixableCategorical) = d.logps

params(d::FixableCategorical) = logps(d)

num_categories(d::FixableCategorical) = length(logps(d))

num_parameters(n::FixableCategorical, independent::Bool = false) = 
    ifelse(n.fixed, 0, num_categories(n) - (independent ? 1 : 0))

num_parameters(n::BitsCategorical, independent::Bool = false) = 
    n.num_cats - (independent ? 1 : 0)

param_buffer_size(n::FixableCategorical) = length(n.logps)

import ProbabilisticCircuits: init_params, loglikelihood # extend

init_params(d::FixableCategorical, perturbation) = begin
    if !d.fixed
        unnormalized_probs = map(rand(Float32, num_categories(d))) do x 
            Float32(1.0 - perturbation + x * 2.0 * perturbation)
        end
        logps = log.(unnormalized_probs ./ sum(unnormalized_probs))
        FixableCategorical(logps, false)
    else
        d
    end
end

loglikelihood(d::FixableCategorical, value, _ = nothing) =
    d.logps[1 + value] # "+1" since we assume categorical values start from 0

struct BitsFixableCategorical <: InputDist
    num_cats::UInt32
    heap_start::UInt32
    fixed::Bool
end

import ProbabilisticCircuits: bits, unbits, flow, update_params, clear_memory # extend

function bits(d::FixableCategorical, heap) 
    num_cats = num_categories(d)
    heap_start = length(heap) + 1
    if !d.fixed
        # use heap to store parameters and space for parameter learning
        append!(heap, logps(d), zeros(eltype(heap), num_cats + 1)) # the last value is used to maintain `missing` flows
    else
        append!(heap, logps(d))
    end
    BitsFixableCategorical(num_cats, heap_start, d.fixed)
end

function unbits(d::BitsFixableCategorical, heap) 
    logps = heap[d.heap_start : d.heap_start + d.num_cats - one(UInt32)]
    FixableCategorical(logps, d.fixed)
end

num_parameters(d::BitsFixableCategorical, independent::Bool = false) = 
    ifelse(d.fixed, 0, d.num_cats - (independent ? 1 : 0))

get_param(d::BitsFixableCategorical, idx, heap) = 
    heap[d.heap_start + UInt32(idx) - one(UInt32)]

get_param(d::BitsCategorical, idx, heap) = 
    heap[d.heap_start + UInt32(idx) - one(UInt32)]

set_param(d::BitsFixableCategorical, idx, param, heap) = begin
    heap[d.heap_start + UInt32(idx) - one(UInt32)] = param
    nothing
end

import ProbabilisticCircuits: loglikelihood, flow # extend

loglikelihood(d::BitsFixableCategorical, value, heap) =
    heap[d.heap_start + UInt32(value)]

loglikelihood(d::BitsFixableCategorical, data, ex_id, variable, heap) = begin
    p = zero(Float32)
    for i = 1 : d.num_cats
        p += data[ex_id, variable, i] * exp(heap[d.heap_start + i - 1])
    end
    log(p)
end

loglikelihood(d::BitsFixableCategorical, value, heap, params, ex_id, start_idx) = 
    if !d.fixed
        params[ex_id, start_idx + UInt32(value)]
    else
        heap[d.heap_start + UInt32(value)]
    end

loglikelihood(d::BitsCategorical, value, heap, params, ex_id, start_idx) = begin
    params[ex_id, start_idx + UInt32(value)]
end

soft_loglikelihood(d::BitsCategorical, value, heap, params, ex_id, start_idx, soft_reg, soft_reg_width) = begin
    logp = params[ex_id, start_idx + UInt32(value)]
    c_start = max(0, value - soft_reg_width ÷ 2)
    c_end = min(c_start + soft_reg_width - 1, d.num_cats - 1)
    sp = zero(Float32)
    for cat_idx = c_start : c_end
        sp += exp(params[ex_id, start_idx + cat_idx])
    end
    sp /= (c_end - c_start + 1)
    PCs.logsumexp(logp + log(one(Float32) - soft_reg), log(sp) + log(soft_reg))
end

function flow(d::BitsFixableCategorical, value, node_flow, heap)
    if !d.fixed
        if ismissing(value)
            CUDA.@atomic heap[d.heap_start+UInt32(2)*d.num_cats] += node_flow
        else
            CUDA.@atomic heap[d.heap_start+d.num_cats+UInt32(value)] += node_flow
        end
    end
    nothing
end

function flow(d::BitsFixableCategorical, data, ex_id, variable, node_flow, heap)
    if !d.fixed
        for i = 1 : d.num_cats
            CUDA.@atomic heap[d.heap_start+d.num_cats+i-1] += node_flow * data[ex_id, variable, i]
        end
    end
    nothing
end

function soft_flow_params(d::BitsCategorical, value, node_flow, params, input_edge_flows, ex_id, 
                          param_start, flow_start, soft_reg, soft_reg_width)
    c_start = max(0, value - soft_reg_width ÷ 2)
    c_end = min(c_start + soft_reg_width - 1, d.num_cats - 1)
    sp = zero(Float32)
    for cat_idx = c_start : c_end
        sp += exp(params[ex_id, param_start + cat_idx])
    end
    sp /= (c_end - c_start + 1)
    base = (one(Float32) - soft_reg) * exp(params[ex_id, param_start + UInt32(value)]) + soft_reg * sp

    main_frac = (one(Float32) - soft_reg) * exp(params[ex_id, param_start + value]) / base

    CUDA.@atomic input_edge_flows[ex_id, flow_start + UInt32(value)] += node_flow * main_frac
    for cat_idx = c_start : c_end
        CUDA.@atomic input_edge_flows[ex_id, flow_start + cat_idx] += soft_reg / 
            (c_end - c_start + 1) * exp(params[ex_id, param_start + cat_idx]) * node_flow / base
    end
    nothing
end

function get_edge_flow(d::BitsFixableCategorical, value, node_flow, param_id, heap)
    if value == param_id
        node_flow
    else
        zero(Float32)
    end
end

function get_edge_flow(d::BitsCategorical, value, node_flow, param_id, heap)
    if value == param_id
        node_flow
    else
        zero(Float32)
    end
end

kl_div(d::BitsFixableCategorical, heap, params1, params2, ex_id, start_idx, norm_params::Bool = true) = begin
    kld = zero(Float32)
    if !d.fixed
        if norm_params
            for i = 0 : ifelse(d.fixed, 0, d.num_cats-1)
                param1 = params1[ex_id, start_idx+i]
                param2 = params2[ex_id, start_idx+i]
                kld += param1 * (log(param1 + 1e-8) - log(param2 + 1e-8))
            end
        else
            for i = 0 : ifelse(d.fixed, 0, d.num_cats-1)
                param1 = params1[ex_id, start_idx+i]
                param2 = params2[ex_id, start_idx+i]
                kld += exp(param1) * (param1 - param2)
            end
        end
    end
    kld
end

compute_target_params(d::BitsFixableCategorical, heap, params1, params2, params1_target, newton_step_size, 
                      newton_nsteps, ex_id, start_idx, norm_params::Bool = true) = begin
    if !d.fixed
        if !norm_params
            for i = 0 : ifelse(d.fixed, 0, d.num_cats-1)
                params1_target[ex_id, start_idx+i] = params1[ex_id, start_idx+i]
            end
            # newton's method
            # TODO
            @assert false
        else
            @assert false
        end
    end
    nothing
end

function get_edge_kld_grad(d::BitsFixableCategorical, node_grad, param1, param2, param_id, params1, params2, norm_params::Bool = false)
    if norm_params
        node_grad * (log(param1 + 1e-8) - log(param2 + 1e-8) + one(Float32)), node_grad * param1 / (param2 + 1e-8)
    else
        node_grad * (param1 - param2 + one(Float32)), node_grad * exp(param1 - param2)
    end
end

function update_params(d::BitsFixableCategorical, heap, pseudocount, inertia)
    heap_start = d.heap_start
    num_cats = d.num_cats
    
    @inbounds if !d.fixed
        # add pseudocount & accumulate node flow
        node_flow = zero(Float32)
        cat_pseudocount = pseudocount / Float32(num_cats)
        for i = 0 : num_cats-1
            node_flow += heap[heap_start+num_cats+i]
        end
        missing_flow = heap[heap_start+UInt32(2)*num_cats]
        node_flow += missing_flow + pseudocount
        
        # update parameter
        for i = 0 : num_cats-1
            oldp = exp(heap[heap_start+i])
            old = inertia * oldp
            new = (one(Float32) - inertia) * (heap[heap_start+num_cats+i] + 
                    cat_pseudocount + missing_flow * oldp) / node_flow 
            new_log_param = log(old + new)
            heap[heap_start+i] = new_log_param
        end
    end
    nothing
end

function clear_memory(d::BitsFixableCategorical, heap, rate)
    if !d.fixed
        heap_start = d.heap_start
        num_cats = d.num_cats
        for i = 0 : num_cats-1
            heap[heap_start+num_cats+i] *= rate
        end
        heap[heap_start+2*num_cats] *= rate
    end
    nothing
end