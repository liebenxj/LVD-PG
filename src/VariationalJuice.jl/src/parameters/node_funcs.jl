

function grad_em_update_params_mul(d::BitsCategorical, heap, lr, batch_size)
    heap_start = d.heap_start
    num_cats = d.num_cats

    @inbounds begin
        cum_val = typemin(Float32)
        missing_flow = heap[heap_start+Int32(2)*num_cats]
        for i = 0 : num_cats-1
            heap[heap_start+num_cats+i] = heap[heap_start+i] + lr * (missing_flow + heap[heap_start+num_cats+i]) / batch_size
            cum_val = logsumexp(cum_val, heap[heap_start+num_cats+i])
        end

        # update parameters
        for i = 0 : num_cats - 1
            heap[heap_start+i] = heap[heap_start+num_cats+i] - cum_val
        end
    end
    nothing
end

function grad_em_update_params_sc_mul(d::BitsCategorical, heap, pseudocount, lr, batch_size)
    heap_start = d.heap_start
    num_cats = d.num_cats

    @inbounds begin
        node_flow = zero(Float32)
        for i = 0 : num_cats-1
            node_flow += heap[heap_start+num_cats+i]
        end
        missing_flow = heap[heap_start+Int32(2)*num_cats]
        node_flow += missing_flow + pseudocount

        cum_val = typemin(Float32)
        for i = 0 : num_cats-1
            # heap[heap_start+num_cats+i] = heap[heap_start+i] + lr * (missing_flow + heap[heap_start+num_cats+i] + pseudocount / num_cats) / node_flow / batch_size
            heap[heap_start+num_cats+i] = heap[heap_start+i] + lr * (missing_flow + heap[heap_start+num_cats+i]) / node_flow
            cum_val = logsumexp(cum_val, heap[heap_start+num_cats+i])
        end

        # update parameters
        for i = 0 : num_cats - 1
            heap[heap_start+i] = heap[heap_start+num_cats+i] - cum_val
        end
    end
    nothing
end

function grad_em_update_params_add(d::BitsCategorical, heap, lr, batch_size)
    heap_start = d.heap_start
    num_cats = d.num_cats

    @inbounds begin
        cum_val = zero(Float32)
        for i = 0 : num_cats-1
            target = heap[heap_start+num_cats+i] / (exp(heap[heap_start+i]) + Float32(1e-8))
            heap[heap_start+num_cats+i] = exp(heap[heap_start+i]) + lr * target / batch_size
            cum_val += heap[heap_start+num_cats+i]
        end

        # update parameters
        for i = 0 : num_cats - 1
            heap[heap_start+i] = log(heap[heap_start+num_cats+i] / cum_val)
        end
    end
    nothing
end

function grad_em_update_params_sc_add(d::BitsCategorical, heap, pseudocount, lr, batch_size)
    heap_start = d.heap_start
    num_cats = d.num_cats

    @inbounds begin
        node_flow = zero(Float32)
        for i = 0 : num_cats-1
            node_flow += heap[heap_start+num_cats+i]
        end
        missing_flow = heap[heap_start+Int32(2)*num_cats]
        node_flow += missing_flow + pseudocount

        # update parameters
        inertia = one(Float32) - lr * (missing_flow + node_flow) / batch_size
        for i = 0 : num_cats - 1
            old = inertia * exp(heap[heap_start+i])
            new = (one(Float32) - inertia) * (heap[heap_start+num_cats+i] + missing_flow + pseudocount / num_cats) / node_flow
            new_logp = log(old + new)
            heap[heap_start+i] = new_logp
        end
    end
    nothing
end