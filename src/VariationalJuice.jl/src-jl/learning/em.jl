using Random
using StatsFuns
using DSP
using ProbabilisticCircuits: prep_memory


function apply_entropy_reg_kernel(edge_aggr, node_aggr, log_params, edges, td_probs, node_cum1, node_cum2, sib_count, up2downedge,
                                  layer_start::Int32, layer_end::Int32, ent_reg::Float32)
    up_edge_id = layer_start + ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 
    @inbounds if up_edge_id <= layer_end
        edge_id = up2downedge[up_edge_id]
        edge = edges[up_edge_id]
        parent_id = edge.parent_id
        prime_id = edge.prime_id
        sub_id = edge.sub_id

        edge_td_prob = td_probs[parent_id]
        if edge isa SumEdge
            prob = exp(edge.logp)
            edge_td_prob *= prob
        end

        CUDA.@atomic td_probs[prime_id] += edge_td_prob
        if sub_id > 0
            td_probs[sub_id] += edge_td_prob
        end
        
        if edge isa SumEdge
            log_params[edge_id] = log(edge_aggr[edge_id] / node_aggr[parent_id])
        end
    end

    CUDA.sync_threads()

    for _ = 1 : 1

        # init `node_cum1`
        @inbounds if up_edge_id <= layer_end
            edge_id = up2downedge[up_edge_id]
            edge = edges[up_edge_id]
            if edge isa SumEdge
                parent_id = edge.parent_id
                node_cum1[parent_id] = zero(Float32)
            end
        end

        CUDA.sync_threads()

        @inbounds if up_edge_id <= layer_end
            edge_id = up2downedge[up_edge_id]
            edge = edges[up_edge_id]
            if edge isa SumEdge
                parent_id = edge.parent_id

                logp = log_params[edge_id]

                y = ent_reg * td_probs[parent_id] * node_aggr[parent_id] * logp - edge_aggr[edge_id] / exp(logp)
                CUDA.@atomic node_cum1[parent_id] += y
            end
        end

        CUDA.sync_threads()

        for _ = 1 : 1

            @inbounds if up_edge_id <= layer_end
                edge_id = up2downedge[up_edge_id]
                edge = edges[up_edge_id]
                if edge isa SumEdge
                    parent_id = edge.parent_id
                    node_cum2[parent_id] = zero(Float32)
                end
            end
    
            CUDA.sync_threads()

            @inbounds if up_edge_id <= layer_end
                edge_id = up2downedge[up_edge_id]
                edge = edges[up_edge_id]
                if edge isa SumEdge
                    parent_id = edge.parent_id
        
                    logp = log_params[edge_id]
                    p_exp_divp = edge_aggr[edge_id] / exp(logp)
                    b = ent_reg * td_probs[parent_id]
                    step = (p_exp_divp - b * logp + node_cum1[parent_id] / (sib_count[parent_id] .+ one(Float32))) / (p_exp_divp + b + 1e-8)
                    
                    # log_params[edge_id] += step
                    CUDA.@atomic node_cum2[parent_id] += exp(log_params[edge_id])
                end
            end

            CUDA.sync_threads()

            @inbounds if up_edge_id <= layer_end
                edge_id = up2downedge[up_edge_id]
                edge = edges[up_edge_id]
                if edge isa SumEdge
                    parent_id = edge.parent_id

                    log_params[edge_id] -= log(node_cum2[parent_id])
                end
            end

            CUDA.sync_threads()

        end
    end
    nothing
end

function apply_entropy_reg(edge_aggr, node_aggr, bpc, log_params, td_probs, node_cum1, node_cum2, sib_count, ent_reg, up2downedge; debug)
    PCs.count_siblings(sib_count, bpc; debug)
    edges = bpc.edge_layers_up.vectors

    CUDA.allowscalar() do
        td_probs[:] .= zero(Float32)
        td_probs[end] = one(Float32)
        node_cum2 .= zero(Float32)
    end
    
    layer_start = 1
    for layer_end in bpc.edge_layers_down.ends
        args = (edge_aggr, node_aggr, log_params, edges, td_probs, node_cum1, node_cum2, 
                sib_count, up2downedge, Int32(layer_start), Int32(layer_end), Float32(ent_reg))
        kernel = @cuda name="apply_entropy_reg" launch=false apply_entropy_reg_kernel(args...)

        threads = launch_configuration(kernel.fun).threads
        blocks = cld(layer_end - layer_start + one(Int32), threads)
        kernel(args...; threads, blocks)

        layer_start = layer_end + 1
    end
    nothing
end

function update_params_with_reg_kernel(edges_down, edges_up, _down2upedge, _log_params, inertia)
    log_params = Base.Experimental.Const(_log_params)
    down2upedge = Base.Experimental.Const(_down2upedge)
    
    edge_id_down = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    @inbounds if edge_id_down <= length(edges_down)
        edge_down = edges_down[edge_id_down]
        if edge_down isa SumEdge 
            
            edge_id_up = down2upedge[edge_id_down]
            # only difference is the tag
            edge_up_tag = edges_up[edge_id_up].tag

            if !(isfirst(edge_up_tag) && islast(edge_up_tag))
                parent_id = edge_down.parent_id

                old = inertia * exp(edge_down.logp)
                new = (one(Float32) - inertia) * exp(log_params[edge_id_down])
                new_log_param = log(old + new)

                edges_down[edge_id_down] = 
                    SumEdge(parent_id, edge_down.prime_id, edge_down.sub_id, 
                            new_log_param, edge_down.tag)

                edges_up[edge_id_up] = 
                    SumEdge(parent_id, edge_down.prime_id, edge_down.sub_id, 
                            new_log_param, edge_up_tag)
            end
        end
    end      
    nothing
end

function update_params_with_reg(bpc, log_params; inertia = 0, debug = false)
    edges_down = bpc.edge_layers_down.vectors
    edges_up = bpc.edge_layers_up.vectors
    down2upedge = bpc.down2upedge
    @assert length(edges_down) == length(down2upedge) == length(edges_up)

    args = (edges_down, edges_up, down2upedge, log_params, Float32(inertia))
    kernel = @cuda name="update_params_with_reg" launch=false update_params_with_reg_kernel(args...) 
    threads = launch_configuration(kernel.fun).threads
    blocks = cld(length(edges_down), threads)
    
    if debug
        println("Update parameters")
        @show threads blocks length(edges_down)
        CUDA.@time kernel(args...; threads, blocks)
    else
        kernel(args...; threads, blocks)
    end
    nothing
end

function weighted_ll(bpc::CuBitsProbCircuit, data::CuArray, weights = nothing; batch_size = 256)
    if weights === nothing
        loglikelihoods(bpc, data; batch_size)
    else
        sum(loglikelihoods(bpc, data; batch_size) .* weights) / sum(weights)
    end
end

function overprint(str)  
    print("\u1b[1F")
    #Moves cursor to beginning of the line n (default 1) lines up   
    print(str)   #prints the new line
    print("\u1b[0K") 

    println() #prints a new line, i really don't like this arcane codes
end

function init_parameters_by_logits(pc::ProbCircuit; mval = 1.0)
    mval = Float32(mval)
    foreach(pc) do pn
        @inbounds if issum(pn)
            n_inputs = num_inputs(pn)
            if n_inputs == 1
                pn.params .= zero(Float32)
            else
                logps = rand(Float32, n_inputs) .* mval .- mval
                logps .-= StatsFuns.logsumexp(logps)
                pn.params .= logps
            end
        elseif isinput(pn)
            init_parameters_by_logits(pn; mval)
        end
    end
end

init_parameters_by_logits(n::PCs.PlainInputNode; mval) = begin
    d = init_parameters_by_logits(dist(n); mval)
    n.dist = d
end

init_parameters_by_logits(d::Categorical; mval, conv = true) = begin
    logps = rand(Float32, length(d.logps)) .* mval .- mval
    logps .-= StatsFuns.logsumexp(logps)
    if conv
        ps = exp.(logps)
        ps .= DSP.conv(ps, [0.1, 0.2, 0.4, 0.2, 0.1])[3:end-2]
        logps = log.(ps ./ sum(ps))
    end
    Categorical(logps)
end

function perturb_parameters(pc::ProbCircuit; mval = 0.2, type = :additive)
    mval = Float32(mval)
    foreach(pc) do pn
        @inbounds if issum(pn)
            n_inputs = num_inputs(pn)
            if n_inputs == 1
                pn.params .= zero(Float32)
            else
                if type == :multiplicative
                    logps = pn.params .+ (rand(Float32, n_inputs) .* mval .- mval)
                    logps .-= StatsFuns.logsumexp(logps)
                elseif type == :additive
                    val = mval / n_inputs
                    ps = exp.(pn.params) .+ (rand(Float32, n_inputs) .* val)
                    logps = log.(ps ./ sum(ps))
                else
                    error("Unknown type")
                end
                pn.params .= logps
            end
        elseif isinput(pn)
            perturb_parameters(pn; mval, type)
        end
    end
end

perturb_parameters(n::PCs.PlainInputNode; mval, type) = begin
    d = perturb_parameters(dist(n); mval, type)
    n.dist = d
end

perturb_parameters(d::Categorical; mval, type) = begin
    n_inputs = length(d.logps)
    if type == :multiplicative
        logps = d.logps .+ (rand(Float32, n_inputs) .* mval .- mval)
        logps .-= StatsFuns.logsumexp(logps)
    elseif type == :additive
        val = mval / n_inputs
        ps = exp.(d.logps) .+ (rand(Float32, n_inputs) .* val)
        logps = log.(ps ./ sum(ps))
    else
        error("Unknown type")
    end
    Categorical(logps)
end

function mini_batch_em_beta(bpc::CuBitsProbCircuit, data::CuArray, num_epochs; 
                            batch_size, pseudocount, soft_reg, soft_reg_width,
                            param_inertia, param_inertia_end = param_inertia, 
                            flow_memory = 0, flow_memory_end = flow_memory, 
                            shuffle=:each_epoch, update_interval = 1, fscale = 1.0,
                            mars_mem = nothing, flows_mem = nothing, flows_buf_mem = nothing, 
                            node_aggr_mem = nothing, edge_aggr_mem = nothing,
                            mine = 2, maxe = 32, debug = false, verbose = true,
                            callbacks = [], eval_dataset = nothing, eval_interval = 0)

    @assert pseudocount >= 0
    @assert 0 <= param_inertia <= 1
    @assert param_inertia <= param_inertia_end <= 1
    @assert 0 <= flow_memory  
    @assert flow_memory <= flow_memory_end  
    @assert shuffle ∈ [:once, :each_epoch, :each_batch]

    num_examples = size(data)[1]
    num_nodes = length(bpc.nodes)
    num_edges = length(bpc.edge_layers_down.vectors)
    num_batches = num_examples ÷ batch_size # drop last incomplete batch

    @assert batch_size <= num_examples

    marginals = prep_memory(mars_mem, (batch_size, num_nodes), (false, true))
    flows = prep_memory(flows_mem, (batch_size, num_nodes), (false, true))
    flows_buf = prep_memory(flows_buf_mem, (batch_size, num_nodes), (false, true))
    node_aggr = prep_memory(node_aggr_mem, (num_nodes,))
    edge_aggr = prep_memory(edge_aggr_mem, (num_edges,))

    edge_aggr .= zero(Float32)
    PCs.clear_input_node_mem(bpc; rate = 0, debug)

    shuffled_indices_cpu = Vector{Int32}(undef, num_examples)
    shuffled_indices = CuVector{Int32}(undef, num_examples)
    batches = [@view shuffled_indices[1+(b-1)*batch_size : b*batch_size]
                for b in 1:num_batches]

    do_shuffle() = begin
        randperm!(shuffled_indices_cpu)
        copyto!(shuffled_indices, shuffled_indices_cpu)
    end

    (shuffle == :once) && do_shuffle()

    Δparam_inertia = (param_inertia_end-param_inertia)/num_epochs
    Δflow_memory = (flow_memory_end-flow_memory)/num_epochs

    log_likelihoods = Vector{Float32}()
    log_likelihoods_epoch = CUDA.zeros(Float32, num_batches, 1)
    last_test_ll = nothing

    for epoch in 1:num_epochs

        log_likelihoods_epoch .= zero(Float32)

        (shuffle == :each_epoch) && do_shuffle()

        edge_aggr .= zero(Float32)
        PCs.clear_input_node_mem(bpc; rate = 0, debug)

        for (batch_id, batch) in enumerate(batches)

            (shuffle == :each_batch) && do_shuffle()
            
            if batch_id % update_interval == 1
                if iszero(flow_memory)
                    edge_aggr .= zero(Float32)
                    PCs.clear_input_node_mem(bpc; rate = 0, debug)
                else
                    # slowly forget old edge aggregates
                    rate = max(zero(Float32), one(Float32) - (batch_size + pseudocount) / flow_memory)
                    edge_aggr .*= rate
                    PCs.clear_input_node_mem(bpc; rate)
                end
            end

            scaled_probs_flows_circuit(flows, flows_buf, marginals, edge_aggr, bpc, data, batch; 
                                       mine, maxe, soft_reg, soft_reg_width, 
                                       fscale, debug, weights = nothing)
            
            @views sum!(log_likelihoods_epoch[batch_id:batch_id, 1:1],
                        marginals[1:batch_size,end:end])

            if batch_id % update_interval == 0
                PCs.add_pseudocount(edge_aggr, node_aggr, bpc, pseudocount; debug)
                PCs.aggr_node_flows(node_aggr, bpc, edge_aggr; debug)
                PCs.update_params(bpc, node_aggr, edge_aggr; inertia = param_inertia, debug)
                
                PCs.update_input_node_params(bpc; pseudocount, inertia = param_inertia, debug)
            end
            
        end

        log_likelihood = sum(log_likelihoods_epoch) / batch_size / num_batches
        push!(log_likelihoods, log_likelihood)

        param_inertia += Δparam_inertia
        flow_memory += Δflow_memory

        if verbose
            if eval_dataset !== nothing && eval_interval > 0 && epoch % eval_interval == 0
                test_ll = loglikelihood_probcat(bpc, eval_dataset; batch_size)

                overprint("Mini-batch EM epoch $epoch/$num_epochs: train LL $log_likelihood - test LL $test_ll")

                last_test_ll = test_ll
            else
                if last_test_ll !== nothing
                    overprint("Mini-batch EM epoch $epoch/$num_epochs; train LL $log_likelihood - test LL $last_test_ll")
                else
                    overprint("Mini-batch EM epoch $epoch/$num_epochs; train LL $log_likelihood")
                end
            end
        end
    end

    PCs.cleanup_memory((flows, flows_mem), 
        (node_aggr, node_aggr_mem), (edge_aggr, edge_aggr_mem))
    CUDA.unsafe_free!(shuffled_indices)

    log_likelihoods
end