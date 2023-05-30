

function pairwise_marginal_kernel(pair_margs, data1, data2, num_ex_threads::Int32, vpair_work::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    vpair_batch, ex_id = fldmod1(threadid, num_ex_threads)

    num_vars1 = size(pair_margs, 1)
    num_vars2 = size(pair_margs, 2)
    vpair_start = one(Int32) + (vpair_batch - one(Int32)) * vpair_work
    vpair_end = min(vpair_start + vpair_work - one(Int32), num_vars1 * num_vars2)
    
    @inbounds if ex_id <= size(data1, 1)
        for vpair_id = vpair_start : vpair_end
            v1, v2 = fldmod1(vpair_id, num_vars2)
            
            d1 = data1[ex_id, v1]
            d2 = data2[ex_id, v2]
            CUDA.@atomic pair_margs[v1,v2,d1+1,d2+1] += one(Float32)
        end
    end
    nothing
end

function single_marginal_kernel(pair_margs, data, num_ex_threads::Int32, vpair_work::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    vpair_batch, ex_id = fldmod1(threadid, num_ex_threads)

    num_vars = size(pair_margs, 1)
    v_start = one(Int32) + (vpair_batch - one(Int32)) * vpair_work
    v_end = min(v_start + vpair_work - one(Int32), num_vars)
    
    @inbounds if ex_id <= size(data, 1)
        for v = v_start : v_end
            d = data[ex_id, v]
            CUDA.@atomic pair_margs[v,d+1] += one(Float32)
        end
    end
    nothing
end

function single_and_pairwise_marginal(data1::CuMatrix, data2::CuMatrix; num_cats = maximum(data1) - minimum(data1) + 1, pseudocount = zero(Float32))

    @assert eltype(data1) != Bool

    num_examples = size(data1, 1)
    num_vars1 = size(data1, 2)
    num_vars2 = size(data2, 2)
    num_var_pairs = num_vars1 * num_vars2

    ## Pairwise marginal ##

    pair_margs = zeros(Float32, num_vars1, num_vars2, num_cats, num_cats)
    Z = num_examples + pseudocount
    single_smooth = Float32(pseudocount / num_cats)
    pair_smooth = Float32(single_smooth / num_cats)
    
    # init pair_margs
    @inbounds pair_margs[:,:,:,:] .= pair_smooth
    pair_margs = cu(pair_margs)
    
    dummy_args = (pair_margs, data1, data2, Int32(1), Int32(1))
    kernel = @cuda name="pairwise_marginal" launch=false pairwise_marginal_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, vpair_work = 
        balance_threads(num_var_pairs, num_examples, config; mine=2, maxe=32)

    args = (pair_margs, data1, data2, Int32(num_example_threads), Int32(vpair_work))
    kernel(args...; threads, blocks)

    pair_margs ./= Z
    pair_margs = Array(pair_margs)

    ## Single marginal ##

    single_margs1 = zeros(Float32, num_vars1, num_cats)
    single_margs2 = zeros(Float32, num_vars2, num_cats)

    @inbounds single_margs1[:,:] .= single_smooth
    @inbounds single_margs2[:,:] .= single_smooth
    single_margs1 = cu(single_margs1)
    single_margs2 = cu(single_margs2)

    dummy_args = (single_margs1, data1, Int32(1), Int32(1))
    kernel = @cuda name="single_marginal" launch=false single_marginal_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, vpair_work = 
        balance_threads(num_vars1, num_examples, config; mine=2, maxe=32)

    args = (single_margs1, data1, Int32(num_example_threads), Int32(vpair_work))
    kernel(args...; threads, blocks)

    threads, blocks, num_example_threads, vpair_work = 
        balance_threads(num_vars2, num_examples, config; mine=2, maxe=32)

    args = (single_margs2, data2, Int32(num_example_threads), Int32(vpair_work))
    kernel(args...; threads, blocks)

    single_margs1 ./= Z
    single_margs2 ./= Z
    single_margs1 = Array(single_margs1)
    single_margs2 = Array(single_margs2)
    
    pair_margs, single_margs1, single_margs2
end


function pairwise_MI_chunked(data::CuMatrix; chunk_size, num_cats = maximum(data) + 1, pseudocount = 0.1, get_ent = false)
    num_samples = size(data, 1)
    num_vars = size(data, 2)

    data .-= minimum(data)

    xlogx(x) = iszero(x) ? zero(x) : x * log(x)
    xlogy(x, y) = iszero(x) && !isnan(y) ? zero(x) : x * log(y)

    MI = zeros(Float32, num_vars, num_vars)
    ENT = zeros(Float32, num_vars)
    for chunk_x_start = 1 : chunk_size : num_vars
        chunk_x_end = min(num_vars, chunk_x_start + chunk_size - 1)
        for chunk_y_start = chunk_x_start : chunk_size : num_vars
            chunk_y_end = min(num_vars, chunk_y_start + chunk_size - 1)

            joint_cont, unary_cont1, unary_cont2 = single_and_pairwise_marginal(
                data[:,chunk_x_start:chunk_x_end], 
                data[:,chunk_y_start:chunk_y_end];
                num_cats, pseudocount
            )

            for var1_idx = chunk_x_start : chunk_x_end
                i = var1_idx - chunk_x_start + 1
                for var2_idx = chunk_y_start : chunk_y_end
                    j = var2_idx - chunk_y_start + 1

                    @inbounds MI[var1_idx, var2_idx] = sum(
                        xlogx.(joint_cont[i,j,:,:]) .- xlogy.(joint_cont[i,j,:,:], unary_cont1[i,:] .* unary_cont2[j,:]')
                    )
                end
                @inbounds ENT[var1_idx] = sum(xlogx.(unary_cont1[i,:]))
            end
        end
    end

    for var1_idx = 1 : num_vars
        for var2_idx = 1 : var1_idx - 1
            MI[var1_idx, var2_idx] = MI[var2_idx, var1_idx]
        end
    end

    if get_ent
        MI, ENT
    else
        MI
    end
end

function get_pairwise_cont(data::CuMatrix, var1, var2; num_cats = maximum(data) + 1, pseudocount = 0.1)
    joint_cont, unary_cont1, unary_cont2 = single_and_pairwise_marginal(
        data[:,var1:var1], 
        data[:,var2:var2];
        num_cats, pseudocount
    )
    reshape(joint_cont, (num_cats, num_cats))
end

function get_marg_cont(data::CuMatrix, var; num_cats = maximum(data) + 1, pseudocount = 0.1)
    joint_cont, unary_cont1, unary_cont2 = single_and_pairwise_marginal(
        data[:,var:var], 
        data[:,var:var];
        num_cats, pseudocount
    )
    reshape(unary_cont1, (num_cats,))
end

function triwise_marginal_kernel(tri_margs, data1, data2, data3, num_ex_threads::Int32, vtri_work::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    vtri_batch, ex_id = fldmod1(threadid, num_ex_threads)

    num_vars1 = size(tri_margs, 1)
    num_vars2 = size(tri_margs, 2)
    num_vars3 = size(tri_margs, 3)
    vtri_start = one(Int32) + (vtri_batch - one(Int32)) * vtri_work
    vtri_end = min(vtri_start + vtri_work - one(Int32), num_vars1 * num_vars2 * num_vars3)
    
    @inbounds if ex_id <= size(data1, 1)
        for vtri_id = vtri_start : vtri_end
            v1, v2 = fldmod1(vtri_id, num_vars2 * num_vars3)
            v2, v3 = fldmod1(v2, num_vars3)
            
            d1 = data1[ex_id, v1]
            d2 = data2[ex_id, v2]
            d3 = data3[ex_id, v3]
            CUDA.@atomic tri_margs[v1,v2,v3,d1+1,d2+1,d3+1] += one(Float32)
        end
    end
    nothing
end

function triwise_marginal(data1::CuMatrix, data2::CuMatrix, data3::CuMatrix; 
                          num_cats = maximum(data1) - minimum(data1) + 1, pseudocount = zero(Float32))

    @assert eltype(data1) != Bool

    num_examples = size(data1, 1)
    num_vars1 = size(data1, 2)
    num_vars2 = size(data2, 2)
    num_vars3 = size(data3, 2)
    num_var_tris = num_vars1 * num_vars2

    ## Triwise marginal ##

    tri_margs = zeros(Float32, num_vars1, num_vars2, num_vars3, num_cats, num_cats, num_cats)
    Z = num_examples + pseudocount
    tri_smooth = Float32(pseudocount / num_cats^3)
    
    # init pair_margs
    @inbounds tri_margs[:,:,:,:,:,:] .= tri_smooth
    tri_margs = cu(tri_margs)
    
    dummy_args = (tri_margs, data1, data2, data3, Int32(1), Int32(1))
    kernel = @cuda name="triwise_marginal" launch=false triwise_marginal_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, vpair_work = 
        balance_threads(num_var_tris, num_examples, config; mine=2, maxe=32)

    args = (tri_margs, data1, data2, data3, Int32(num_example_threads), Int32(vpair_work))
    kernel(args...; threads, blocks)

    tri_margs ./= Z
    tri_margs = Array(tri_margs)

    tri_margs
end

function single_marginal(data::CuMatrix; num_cats = maximum(data) - minimum(data) + 1, pseudocount = zero(Float32))

    @assert eltype(data) != Bool

    num_examples = size(data, 1)
    num_vars = size(data, 2)

    ## Single marginal ##

    single_margs = zeros(Float32, num_vars, num_cats)

    Z = num_examples + pseudocount
    single_smooth = Float32(pseudocount / num_cats)

    @inbounds single_margs[:,:] .= single_smooth
    single_margs = cu(single_margs)

    dummy_args = (single_margs, data, Int32(1), Int32(1))
    kernel = @cuda name="single_marginal" launch=false single_marginal_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, vpair_work = 
        balance_threads(num_vars, num_examples, config; mine=2, maxe=32)

    args = (single_margs, data, Int32(num_example_threads), Int32(vpair_work))
    kernel(args...; threads, blocks)

    single_margs ./= Z
    single_margs = Array(single_margs)
    
    single_margs
end

function tri_MI_chunked(data::CuMatrix; chunk_size, num_cats = maximum(data) + 1, pseudocount = 0.1)
    num_samples = size(data, 1)
    num_vars = size(data, 2)

    data .-= minimum(data)

    xlogx(x) = iszero(x) ? zero(x) : x * log(x)
    xlogy(x, y) = iszero(x) && !isnan(y) ? zero(x) : x * log(y)

    MI = zeros(Float32, num_vars, num_vars, num_vars)
    for chunk_x_start = 1 : chunk_size : num_vars
        chunk_x_end = min(num_vars, chunk_x_start + chunk_size - 1)
        for chunk_y_start = 1 : chunk_size : num_vars
            chunk_y_end = min(num_vars, chunk_y_start + chunk_size - 1)
            for chunk_z_start = 1 : chunk_size : num_vars
                chunk_z_end = min(num_vars, chunk_z_start + chunk_size - 1)

                tri_cont = triwise_marginal(
                    data[:,chunk_x_start:chunk_x_end], 
                    data[:,chunk_y_start:chunk_y_end],
                    data[:,chunk_z_start:chunk_z_end];
                    num_cats, pseudocount
                )
                unary_cont1 = single_marginal(
                    data[:,chunk_x_start:chunk_x_end];
                    num_cats, pseudocount
                )
                unary_cont2 = single_marginal(
                    data[:,chunk_y_start:chunk_y_end];
                    num_cats, pseudocount
                )
                unary_cont3 = single_marginal(
                    data[:,chunk_z_start:chunk_z_end];
                    num_cats, pseudocount
                )

                num_x_vars = chunk_x_end - chunk_x_start + 1
                num_y_vars = chunk_y_end - chunk_y_start + 1
                num_z_vars = chunk_z_end - chunk_z_start + 1

                for var1_idx = chunk_x_start : chunk_x_end
                    i = var1_idx - chunk_x_start + 1
                    for var2_idx = chunk_y_start : chunk_y_end
                        j = var2_idx - chunk_y_start + 1
                        for var3_idx = chunk_z_start : chunk_z_end
                            k = var3_idx - chunk_z_start + 1

                            s = reshape(reshape(unary_cont1[i,:] .* unary_cont2[j,:]', (:,)) .* unary_cont3[k,:]', (num_cats, num_cats, num_cats))
                            @inbounds MI[var1_idx, var2_idx, var3_idx] = sum(
                                xlogx.(tri_cont[i,j,k,:,:,:]) .- xlogy.(tri_cont[i,j,k,:,:,:], s)
                            )
                        end
                    end
                end

            end
        end
    end

    MI
end