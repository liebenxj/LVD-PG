#do global clustering and log firstly
CUDA_VISIBLE_DEVICES=2 julia parallel_PG.jl 1 1 400 "${@}"
#run multi-process to do progressive growing
CUDA_VISIBLE_DEVICES=2 julia parallel_PG.jl 1 100 400 "${@}" &
CUDA_VISIBLE_DEVICES=3 julia parallel_PG.jl 101 200 400 "${@}" &
CUDA_VISIBLE_DEVICES=4 julia parallel_PG.jl 201 300 400 "${@}" &
CUDA_VISIBLE_DEVICES=5 julia parallel_PG.jl 301 400 400 "${@}" &
wait
