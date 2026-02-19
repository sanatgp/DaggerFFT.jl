#!/usr/bin/env julia
using MPI: MPI, mpiexec
using InteractiveUtils: versioninfo

# Usage: julia --project=@. benchmarks/runtests_mpi.jl <backend> <nproc> [N] [chunk] [decomp]
#   backend: "cpu" or "gpu"
#   nproc:   number of MPI ranks
#   N:       grid size (default: 512)
#   chunk:   chunk size (default: 64 for cpu, 256 for gpu)
#   decomp:  "pencil" or "slab" (default: pencil)
#
# Examples:
#   julia --project=@. benchmarks/runtests_mpi.jl cpu 4 512 64 pencil
#   julia --project=@. benchmarks/runtests_mpi.jl cpu 16 1024 128 slab
#   julia --project=@. benchmarks/runtests_mpi.jl gpu 4 512 256 pencil
#   julia --project=@. benchmarks/runtests_mpi.jl gpu 8 720 360 slab

backend = lowercase(get(ARGS, 1, "cpu"))
nproc = parse(Int, get(ARGS, 2, "4"))
N = get(ARGS, 3, "512")
chunk = get(ARGS, 4, backend == "gpu" ? "256" : "64")
decomp = get(ARGS, 5, "pencil")

if backend == "gpu"
    test_file = joinpath(@__DIR__, "fftmpigpu.jl")
else
    test_file = joinpath(@__DIR__, "fftmpi.jl")
end
extra_args = [N, chunk, decomp]

println()
versioninfo()
println("\n", MPI.MPI_LIBRARY_VERSION_STRING, "\n")

@info "Running $(basename(test_file)) with $nproc processes ($backend, N=$N, chunk=$chunk, decomp=$decomp)..."
@time mpiexec() do cmd
    run(`$cmd -np $nproc --oversubscribe $(first(Base.julia_cmd())) --project=$(joinpath(@__DIR__, "..")) $test_file $extra_args`)
end