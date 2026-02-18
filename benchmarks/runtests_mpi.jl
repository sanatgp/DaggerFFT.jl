#!/usr/bin/env julia

# This is based on the runtests.jl file of MPI.jl.

using MPI: MPI, mpiexec
using InteractiveUtils: versioninfo
#using Dagger
#using DaggerGPU
#using CUDA
#JULIA_NVTX_CALLBACKS="gc"
#GC.gc(true)
test_files = [
    "fftmpigpu.jl"
]

Nproc = 4
println()
versioninfo()
println("\n", MPI.MPI_LIBRARY_VERSION_STRING, "\n")
#scope = Dagger.scope(cuda_gpus=1)
#Dagger.with_options(;scope) do
for fname in test_files
    @info "Running $fname with $Nproc processes..."
    @time mpiexec() do cmd
       run(`$cmd -np $Nproc --oversubscribe $(first(Base.julia_cmd())) --project $fname`)
     end
    println()
end
#end

