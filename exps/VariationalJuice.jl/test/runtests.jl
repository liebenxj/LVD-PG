using Jive

ENV["JIVE_PROCS"]="16"

runtests(@__DIR__, skip=["runtests.jl", "helper"])