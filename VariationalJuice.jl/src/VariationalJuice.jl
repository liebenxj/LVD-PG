using ProbabilisticCircuits
import ProbabilisticCircuits as PCs
using CUDA


include("nodes/fixable_categorical_dist.jl")
include("nodes/gaussian_dist.jl")
include("nodes/discrete_logistic_dist.jl")
include("nodes/vtree.jl")

include("bits_circuit.jl")

include("misc.jl")
include("transformations.jl")

include("queries/likelihood.jl")
include("queries/flow.jl")
include("queries/gradient.jl")
include("queries/kld.jl")
include("queries/sample.jl")
include("queries/td_prob.jl")

include("learning/prob_flow_circuit.jl")
include("learning/em.jl")

include("regularization/reg_likelihood.jl")
include("regularization/reg_flow.jl")
include("regularization/reg_em.jl")

include("sgd/adam.jl")

include("structures/hclts.jl")
include("structures/customizable_hclt.jl")
include("structures/categorical_clt.jl")
include("structures/hclt_new.jl")

include("utils/preprocess_pcs.jl")
include("utils/map_params.jl")
include("utils/softmax.jl")
include("utils/aggregate.jl")
include("utils/cuda_preallocation.jl")
include("utils/utils.jl")
include("utils/information.jl")
include("utils/deepcopy.jl")

include("prune_grow/prune.jl")
include("prune_grow/grow.jl")

include("pretrain/layered_bit_circuit.jl")
include("pretrain/layered_value.jl")
include("pretrain/layered_flow.jl")
include("pretrain/mini_batch_warmup.jl")

include("multi_head_pc/multi_head_bit_circuit.jl")
include("multi_head_pc/multi_head_pc_likelihood.jl")
include("multi_head_pc/multi_head_pc_flow.jl")
include("multi_head_pc/multi_head_pc_em.jl")

include("conditional_pc/bit_circuit.jl")
include("conditional_pc/utils.jl")
include("conditional_pc/likelihood.jl")
include("conditional_pc/flow.jl")

include("parameters/node_funcs.jl")
include("parameters/gradient_em.jl")