
############################################
## Preallocation for kld and kld_backward ##
############################################

function kld_preallocation(mbpc::CuMetaBitsProbCircuit; num_examples)
    n_nodes = num_nodes(mbpc)
    n_edges = num_edges(mbpc)
    n_pars = num_parameters(mbpc)
    num_input_params = mbpc.num_input_params

    global prealloc_klds = prep_memory(nothing, (num_examples, n_nodes), (false, true))
    global prealloc_klds_last = prep_memory(nothing, (num_examples,), (false,))
    global prealloc_edge_klds = prep_memory(nothing, (num_examples, n_edges), (false, true))
    global prealloc_kld_params1 = prep_memory(nothing, (num_examples, n_pars), (false, true))
    global prealloc_kld_params2 = prep_memory(nothing, (num_examples, n_pars), (false, true))

    global prealloc_kld_grads = prep_memory(nothing, (num_examples, n_nodes), (false, true))
    global prealloc_kld_edge_grads1 = prep_memory(nothing, (num_examples, n_edges), (false, true))
    global prealloc_kld_edge_grads2 = prep_memory(nothing, (num_examples, n_edges), (false, true))
    global prealloc_kld_input_edge_grads1 = prep_memory(nothing, (num_examples, num_input_params), (false, true))
    global prealloc_kld_input_edge_grads2 = prep_memory(nothing, (num_examples, num_input_params), (false, true))

    global prealloc_kld_param_grads1 = prep_memory(nothing, (num_examples, n_pars), (false, true))
    global prealloc_kld_param_grads2 = prep_memory(nothing, (num_examples, n_pars), (false, true))
    
    nothing
end

function kld_free_mem()
    CUDA.unsafe_free!(prealloc_klds)
    CUDA.unsafe_free!(prealloc_klds_last)
    CUDA.unsafe_free!(prealloc_edge_klds)
    CUDA.unsafe_free!(prealloc_kld_params1)
    CUDA.unsafe_free!(prealloc_kld_params2)

    CUDA.unsafe_free!(prealloc_kld_grads)
    CUDA.unsafe_free!(prealloc_kld_edge_grads1)
    CUDA.unsafe_free!(prealloc_kld_edge_grads2)
    CUDA.unsafe_free!(prealloc_kld_input_edge_grads1)
    CUDA.unsafe_free!(prealloc_kld_input_edge_grads2)

    CUDA.unsafe_free!(prealloc_kld_param_grads1)
    CUDA.unsafe_free!(prealloc_kld_param_grads2)
end

function kld_with_prealloc(mbpc::CuMetaBitsProbCircuit, params1, params2; norm_params = false)
    example_ids = 1 : size(params1, 1)

    @assert size(params1, 1) == size(prealloc_kld_params1, 1)
    @assert size(params1, 2) == size(prealloc_kld_params1, 2)

    copyto!(prealloc_kld_params1, params1)
    copyto!(prealloc_kld_params2, params2)

    kld(prealloc_klds, prealloc_edge_klds, mbpc, prealloc_kld_params1, prealloc_kld_params2, example_ids; mine = 2, maxe = 32, norm_params)

    prealloc_klds_last .= prealloc_klds[:,end]

    Array(prealloc_klds_last)
end

function kld_backward_with_prealloc(mbpc::CuMetaBitsProbCircuit; norm_params = false, grad_wrt_logparams = true)
    example_ids = 1 : size(prealloc_kld_params1, 1)

    kld_backward_inner(prealloc_kld_grads, prealloc_kld_edge_grads1, prealloc_kld_edge_grads2, nothing, mbpc, prealloc_kld_params1, prealloc_kld_params2,
                       prealloc_klds, prealloc_edge_klds, example_ids; mine = 2, maxe = 32, debug = false, grad_wrt_logparams)
    input_kld_down(prealloc_kld_grads, prealloc_kld_input_edge_grads1, prealloc_kld_input_edge_grads2, mbpc, prealloc_kld_params1, prealloc_kld_params2, 
                   example_ids; mine = 2, maxe = 32, debug = false, norm_params)

    merge_params(mbpc, prealloc_kld_edge_grads1, prealloc_kld_input_edge_grads1, prealloc_kld_param_grads1)
    merge_params(mbpc, prealloc_kld_edge_grads2, prealloc_kld_input_edge_grads2, prealloc_kld_param_grads2)

    kld_param_grads1_cpu = Array(prealloc_kld_param_grads1)
    kld_param_grads2_cpu = Array(prealloc_kld_param_grads2)

    kld_param_grads1_cpu, kld_param_grads2_cpu
end

##############################################
## Preallocation for kld_with_update_target ##
##############################################

function kld_target_preallocation(mbpc::CuMetaBitsProbCircuit; num_examples)
    n_nodes = num_nodes(mbpc)
    n_edges = num_edges(mbpc)
    n_pars = num_parameters(mbpc)
    n_pargroups = mbpc.num_param_groups

    global prealloc_klds = prep_memory(nothing, (num_examples, n_nodes), (false, true))
    global prealloc_edge_klds = prep_memory(nothing, (num_examples, n_edges), (false, true))
    global prealloc_kld_params1_target = prep_memory(nothing, (num_examples, n_pars), (false, true))
    global prealloc_kld_par_buffer = prep_memory(nothing, (num_examples, n_pars), (false, true))
    global prealloc_kld_cum_par_buffer = prep_memory(nothing, (num_examples, n_pargroups), (false, true))
    global prealloc_kld_cum_logp_buffer = prep_memory(nothing, (num_examples, n_pargroups), (false, true))
    global prealloc_kld_new_klds = prep_memory(nothing, (num_examples, n_nodes), (false, true))
    global prealloc_kld_new_edge_klds = prep_memory(nothing, (num_examples, n_edges), (false, true))

    global prealloc_kld_params1 = prep_memory(nothing, (num_examples, n_pars), (false, true))
    global prealloc_kld_params2 = prep_memory(nothing, (num_examples, n_pars), (false, true))

    global prealloc_klds_last = prep_memory(nothing, (num_examples,), (false,))
    
    nothing
end

function kld_with_update_target_prealloc(mbpc::CuMetaBitsProbCircuit, params1, params2; newton_step_size = 0.02, newton_nsteps = 3, norm_params = false)
    example_ids = 1 : size(params1, 1)

    @assert size(params1, 1) == size(prealloc_kld_params1, 1)
    @assert size(params1, 2) == size(prealloc_kld_params1, 2)

    copyto!(prealloc_kld_params1, params1)
    copyto!(prealloc_kld_params2, params2)

    kld_with_target(prealloc_klds, prealloc_edge_klds, prealloc_kld_new_klds, prealloc_kld_new_edge_klds, mbpc, 
                    prealloc_kld_params1, prealloc_kld_params2, prealloc_kld_params1_target, 
                    prealloc_kld_cum_par_buffer, prealloc_kld_cum_par_buffer,
                    prealloc_kld_cum_logp_buffer, example_ids, newton_step_size, newton_nsteps; 
                    mine = 2, maxe = 32, norm_params)

    prealloc_klds_last .= prealloc_klds[:,end]

    Array(prealloc_klds_last), Array(prealloc_kld_params1_target)
end

################################################################
## Preallocation for gumbel_sample and gumbel_sample_backward ##
################################################################

function gumbel_sample_preallocation(mbpc::CuMetaBitsProbCircuit; num_examples, num_vars, par_buffer_size, allocate_for_grad = false, allocate_for_target = false)
    n_nodes = num_nodes(mbpc)
    n_edges = num_edges(mbpc)
    n_pars = num_parameters(mbpc)

    global prealloc_td_probs = prep_memory(nothing, (num_examples, n_nodes), (false, true))
    global prealloc_edge_td_probs = prep_memory(nothing, (num_examples, n_edges), (false, true))
    global prealloc_sample_edge_probs = prep_memory(nothing, (num_examples, length(mbpc.param2edge)), (false, true))
    global prealloc_sample_cum_probs = prep_memory(nothing, (num_examples, mbpc.num_param_groups), (false, true))

    global prealloc_sample_params = prep_memory(nothing, (num_examples, n_pars), (false, true))
    global prealloc_sample_input_aggr_params = prep_memory(nothing, (num_examples, num_vars, par_buffer_size), (false, true, true))

    if allocate_for_grad
        global prealloc_sample_grads = prep_memory(nothing, (num_examples, n_nodes), (false, true))
        global prealloc_sample_param_grads = prep_memory(nothing, (num_examples, n_pars), (false, true))
        global prealloc_sample_cum_grads = prep_memory(nothing, (num_examples, mbpc.num_param_groups), (false, true))
        global prealloc_sample_input_aggr_grads = prep_memory(nothing, (num_examples, num_vars, par_buffer_size), (false, true, true))
    end

    if allocate_for_target
        global prealloc_sample_input_aggr_targets = prep_memory(nothing, (num_examples, num_vars, par_buffer_size), (false, true, true))
        
        global prealloc_sample_params_target = prep_memory(nothing, (num_examples, n_pars), (false, true))
        global prealloc_sample_target_td_probs = prep_memory(nothing, (num_examples, n_nodes), (false, true))
        global prealloc_sample_cum_par_buffer = prep_memory(nothing, (num_examples, mbpc.num_param_groups), (false, true))
    end

    nothing
end

function gumbel_sample_free_mem(; allocate_for_grad = false, allocate_for_target = false)
    CUDA.unsafe_free!(prealloc_td_probs)
    CUDA.unsafe_free!(prealloc_edge_td_probs)
    CUDA.unsafe_free!(prealloc_sample_edge_probs)
    CUDA.unsafe_free!(prealloc_sample_cum_probs)

    CUDA.unsafe_free!(prealloc_sample_params)
    CUDA.unsafe_free!(prealloc_sample_input_aggr_params)

    if allocate_for_grad
        CUDA.unsafe_free!(prealloc_sample_grads)
        CUDA.unsafe_free!(prealloc_sample_param_grads)
        CUDA.unsafe_free!(prealloc_sample_cum_grads)
        CUDA.unsafe_free!(prealloc_sample_input_aggr_grads)
    end

    if allocate_for_target
        CUDA.unsafe_free!(prealloc_sample_input_aggr_targets)

        CUDA.unsafe_free!(prealloc_sample_params_target)
        CUDA.unsafe_free!(prealloc_sample_target_td_probs)
        CUDA.unsafe_free!(prealloc_sample_cum_par_buffer)
    end

    nothing
end

function gumbel_sample_with_prealloc(mbpc::CuMetaBitsProbCircuit, params; temperature, no_gumbel = false, norm_params = false)
    example_ids = 1 : size(params, 1)

    @assert size(params, 1) == size(prealloc_sample_params, 1)
    @assert size(params, 2) == size(prealloc_sample_params, 2)

    copyto!(prealloc_sample_params, params)

    gumbel_sample(prealloc_td_probs, prealloc_edge_td_probs, prealloc_sample_edge_probs, prealloc_sample_cum_probs, prealloc_sample_params, 
                  prealloc_sample_input_aggr_params, mbpc, temperature, example_ids; mine = 2, maxe = 32, norm_params, no_gumbel)

    input_aggr_params_cpu = Array(prealloc_sample_input_aggr_params)

    input_aggr_params_cpu
end

function gumbel_sample_backward_with_prealloc(mbpc::CuMetaBitsProbCircuit, input_aggr_grads; temperature, norm_params = false, grad_wrt_logparams = true)
    example_ids = 1 : size(prealloc_sample_params, 1)

    copyto!(prealloc_sample_input_aggr_grads, input_aggr_grads)

    gumbel_sample_backward(prealloc_td_probs, prealloc_sample_grads, prealloc_sample_param_grads, prealloc_edge_td_probs, 
                           prealloc_sample_edge_probs, prealloc_sample_cum_probs, prealloc_sample_cum_grads, prealloc_sample_params, 
                           prealloc_sample_input_aggr_params, prealloc_sample_input_aggr_grads, mbpc, temperature, example_ids; mine = 2, maxe = 32, 
                           norm_params, grad_wrt_logparams)

    param_grads_cpu = Array(prealloc_sample_param_grads)

    param_grads_cpu
end

function gumbel_sample_target_with_prealloc(mbpc::CuMetaBitsProbCircuit, input_aggr_targets; step_size = 0.1, norm_params = false)
    example_ids = 1 : size(prealloc_sample_params, 1)

    copyto!(prealloc_sample_input_aggr_targets, input_aggr_targets)

    gumbel_sample_param_target(prealloc_td_probs, prealloc_edge_td_probs, prealloc_sample_edge_probs, prealloc_sample_cum_probs,
                               prealloc_sample_params_target, prealloc_sample_target_td_probs, prealloc_sample_cum_par_buffer, 
                               prealloc_sample_input_aggr_targets, mbpc, step_size, example_ids; mine = 2, maxe = 32, norm_params)

    Array(prealloc_sample_params_target)
end

##################################################
## Preallocation for computing normalized flows ##
##################################################

function normalized_flows_preallocation(mbpc_xz::CuMetaBitsProbCircuit, mbpc_z::CuMetaBitsProbCircuit; num_examples, num_vars, 
                                        mbpc_mapping)
    n_nodes_xz = num_nodes(mbpc_xz)
    n_edges_xz = num_edges(mbpc_xz)
    n_pars_xz = num_parameters(mbpc_xz)
    n_pargroups_xz = mbpc_xz.num_param_groups

    n_pars_z = num_parameters(mbpc_z)
    n_pargroups_z = mbpc_z.num_param_groups

    global prealloc_xz_mars = prep_memory(nothing, (num_examples, n_nodes_xz), (false, true))
    global prealloc_xz_edge_mars = prep_memory(nothing, (num_examples, n_edges_xz), (false, true))
    global prealloc_xz_flows = prep_memory(nothing, (num_examples, n_nodes_xz), (false, true))
    global prealloc_xz_edge_flows = prep_memory(nothing, (num_examples, n_edges_xz), (false, true))
    global prealloc_xz_input_edge_flows = prep_memory(nothing, (num_examples, mbpc_xz.num_input_params), (false, true))
    global prealloc_xz_par_groups = prep_memory(nothing, (num_examples, n_pargroups_xz), (false, true))

    global prealloc_xz_data = prep_memory(nothing, (num_examples, num_vars), (false, true))

    global prealloc_xz_params = prep_memory(nothing, (num_examples, n_pars_xz), (false, true))
    global prealloc_z_params = prep_memory(nothing, (num_examples, n_pars_z), (false, true))

    global prealloc_z_cum_params = prep_memory(nothing, (num_examples, n_pargroups_z), (false, true))

    global prealloc_mbpc_z_mapping1 = cu(mbpc_mapping[1])
    global prealloc_mbpc_z_mapping2 = cu(mbpc_mapping[2])

    nothing
end

function normalized_flows_free_mem()
    CUDA.unsafe_free!(prealloc_xz_mars)
    CUDA.unsafe_free!(prealloc_xz_edge_mars)
    CUDA.unsafe_free!(prealloc_xz_flows)
    CUDA.unsafe_free!(prealloc_xz_edge_flows)
    CUDA.unsafe_free!(prealloc_xz_input_edge_flows)
    CUDA.unsafe_free!(prealloc_xz_par_groups)

    CUDA.unsafe_free!(prealloc_xz_data)

    CUDA.unsafe_free!(prealloc_xz_params)
    CUDA.unsafe_free!(prealloc_z_params)

    CUDA.unsafe_free!(prealloc_z_cum_params)

    CUDA.unsafe_free!(prealloc_mbpc_z_mapping1)
    CUDA.unsafe_free!(prealloc_mbpc_z_mapping2)

    nothing
end

function normalized_flows_with_prealloc(mbpc_xz::CuMetaBitsProbCircuit, mbpc_z::CuMetaBitsProbCircuit, data::Matrix)
    
    copyto!(prealloc_xz_data, data)
    
    per_sample_normalized_flows(mbpc_xz, prealloc_xz_data; batch_size = size(data, 1), 
                                mars_mem = prealloc_xz_mars, edge_mars_mem = prealloc_xz_edge_mars, flows_mem = prealloc_xz_flows,
                                edge_flows_mem = prealloc_xz_edge_flows, input_edge_flows_mem = prealloc_xz_input_edge_flows, 
                                par_groups_mem = prealloc_xz_par_groups, normalized_flows_mem = prealloc_xz_params)

    map_pc_parameters(mbpc_z, mbpc_xz, prealloc_z_params, prealloc_xz_params; 
                      param_mapping = (prealloc_mbpc_z_mapping1, prealloc_mbpc_z_mapping2), to_cpu = false, pc2topc1 = nothing)

    CUDA.clamp!(prealloc_z_params, -20.0, 0.0)

    normalize_params(mbpc_z, prealloc_z_params; undo = false, cum_params_mem = prealloc_z_cum_params)

    Array(prealloc_z_params)
end

##################################################
## Preallocation for conditioned TD probability ##
##################################################

function cond_td_prob_preallocation(mbpc::CuMetaBitsProbCircuit, marked_pc_idx1, marked_pc_idx2, z_shape; num_examples)
    n_nodes = length(mbpc.bpc.nodes)
    num_vars = num_variables(mbpc)
    num_cats = num_categories(mbpc)

    global prealloc_cond_td_probs = prep_memory(nothing, (num_examples, n_nodes), (false, true))

    global prealloc_marked_pc_idx1 = cu(marked_pc_idx1)
    global prealloc_marked_pc_idx2 = cu(marked_pc_idx2)

    global prealloc_cond_td_probs_node_prob = prep_memory(nothing, (num_examples, z_shape[1], z_shape[2]), (false, true, true))

    global prealloc_cond_td_input_aggr_params = prep_memory(nothing, (num_examples, num_vars, num_cats), (false, true, true))

    nothing
end

function cond_td_prob_free_mem()
    CUDA.unsafe_free!(prealloc_cond_td_probs)
    CUDA.unsafe_free!(prealloc_marked_pc_idx1)
    CUDA.unsafe_free!(prealloc_marked_pc_idx2)

    CUDA.unsafe_free!(prealloc_cond_td_probs_node_prob)
    CUDA.unsafe_free!(prealloc_cond_td_input_aggr_params)

    nothing
end

function cond_td_prob_with_prealloc(mbpc, node_probs)

    copyto!(prealloc_cond_td_probs_node_prob, node_probs)
    
    conditioned_td_prob(mbpc, prealloc_marked_pc_idx1, prealloc_marked_pc_idx2, 
                        prealloc_cond_td_probs_node_prob,
                        prealloc_cond_td_input_aggr_params; td_probs_mem = prealloc_cond_td_probs)

    Array(prealloc_cond_td_input_aggr_params)
end