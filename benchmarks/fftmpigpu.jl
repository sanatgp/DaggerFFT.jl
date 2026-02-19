using MPI
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
sz = MPI.Comm_size(comm)

using CUDA
CUDA.reclaim()
GC.gc()
CUDA.reclaim()

using Dagger
accel = Dagger.MPIAcceleration(comm)
Dagger.accelerate!(accel)
using LinearAlgebra
using AbstractFFTs
using KernelAbstractions
using GPUArraysCore
using FFTW
using DaggerFFT

# Parse command-line arguments
N = parse(Int, get(ARGS, 1, "512"))
chunk = parse(Int, get(ARGS, 2, "256"))
decomp = lowercase(get(ARGS, 3, "pencil"))  # "pencil" or "slab"

indexes(a::ArrayDomain) = a.indexes
Base.getindex(arr::GPUArraysCore.AbstractGPUArray, d::ArrayDomain) = arr[indexes(d)...]

A = CUDA.rand(ComplexF32, N, N, N)
backend = KernelAbstractions.get_backend(A)
B = Array(A)

USE_SLAB = (decomp == "slab")

all_procs = collect(Dagger.get_processors(Dagger.MPIClusterProc()))
all_scopes = map(Dagger.ExactScope, all_procs)
scope = Dagger.UnionScope(all_scopes...)

if USE_SLAB
    if rank == 0
        DA = distribute(A, Blocks(N, N, chunk); root=0, comm=comm)
        DB = distribute(A, Blocks(chunk, N, N); root=0, comm=comm)
    else
        DA = distribute(nothing, Blocks(N, N, chunk); root=0, comm=comm)
        DB = distribute(nothing, Blocks(chunk, N, N); root=0, comm=comm)
    end

    workspace_AB = create_gpu_workspace(DA, DB)
    workspace_BA = create_gpu_workspace(DB, DA)

    if rank == 0
        println("Using SLAB decomposition (1 transpose)")
    end
else
    if rank == 0
        DA = distribute(A, Blocks(N, chunk, chunk); root=0, comm=comm)
        DB = distribute(A, Blocks(chunk, N, chunk); root=0, comm=comm)
        DC = distribute(A, Blocks(chunk, chunk, N); root=0, comm=comm)
    else
        DA = distribute(nothing, Blocks(N, chunk, chunk); root=0, comm=comm)
        DB = distribute(nothing, Blocks(chunk, N, chunk); root=0, comm=comm)
        DC = distribute(nothing, Blocks(chunk, chunk, N); root=0, comm=comm)
    end

    workspace_AB = create_gpu_workspace(DA, DB)
    workspace_BC = create_gpu_workspace(DB, DC)
    workspace_CB = create_gpu_workspace(DC, DB)
    workspace_BA = create_gpu_workspace(DB, DA)

    if rank == 0
        println("Using PENCIL decomposition (2 transposes)")
    end
end

# Warm-up
if USE_SLAB
    gpu_fft!(DB, DA, workspace_AB, scope, (FFT!(), FFT!(), FFT!()), (1, 2, 3), Slab())
    gpu_ifft!(DA, DB, workspace_BA, scope, (IFFT!(), IFFT!(), IFFT!()), (1, 2, 3), Slab())
else
    gpu_fft!(DC, DA, DB, workspace_AB, workspace_BC, scope, (FFT!(), FFT!(), FFT!()), (1, 2, 3), Pencil())
    gpu_ifft!(DA, DC, DB, workspace_CB, workspace_BA, scope, (IFFT!(), IFFT!(), IFFT!()), (1, 2, 3), Pencil())
end

for iter in 1:10
    start_time = MPI.Wtime()

    if USE_SLAB
        @time gpu_fft!(DB, DA, workspace_AB, scope,
                       (FFT!(), FFT!(), FFT!()), (1, 2, 3), Slab())
        @time gpu_ifft!(DA, DB, workspace_BA, scope,
                        (IFFT!(), IFFT!(), IFFT!()), (1, 2, 3), Slab())
    else
        @time gpu_fft!(DC, DA, DB, workspace_AB, workspace_BC, scope,
                       (FFT!(), FFT!(), FFT!()), (1, 2, 3), Pencil())
        @time gpu_ifft!(DA, DC, DB, workspace_CB, workspace_BA, scope,
                        (IFFT!(), IFFT!(), IFFT!()), (1, 2, 3), Pencil())
    end

    elapsed_time = MPI.Wtime() - start_time
    MPI.Barrier(comm)

    time_per_transform = elapsed_time / 2.0
    fftsize = Float64(N * N * N)
    floprate = 5.0 * fftsize * log2(fftsize) * 1e-9 / time_per_transform
#=
    if iter == 1 && rank == 0
        DA_cpu = Array(fetch(DC.chunks[1]))

        println("DA first chunk (GPU->CPU) after round-trip: ", DA_cpu[1:5, 1:5, 1])
        println("Original A first slice: ", A[1:5, 1:5, 1])

        reference_fft = FFTW.fft(B)
        reference_ifft = FFTW.ifft(reference_fft)
        println("Reference round-trip first slice: ", reference_fft[1:5, 1:5, 1])

        max_error = maximum(abs.(DA_cpu .- reference_ifft[1:size(DA_cpu,1), 1:size(DA_cpu,2), 1:size(DA_cpu,3)]))
        println("Max error after round-trip: ", max_error)
    end
=#
    if iter == 10 && rank == 0
        println("DaggerFFT GPU performance test ($(USE_SLAB ? "SLAB" : "PENCIL") decomposition)")
        println("Size:      $(N)x$(N)x$(N)")
        println("Chunk:     $(chunk)")
        println("MPI ranks: $(sz)")
        println("Time per transform: $(time_per_transform) (s)")
        println("Performance:  $(floprate) GFlops/s")
    end

    MPI.Barrier(comm)
end

if USE_SLAB
    gpu_cleanup_workspace!(workspace_AB)
    gpu_cleanup_workspace!(workspace_BA)
else
    gpu_cleanup_workspace!(workspace_AB)
    gpu_cleanup_workspace!(workspace_BC)
    gpu_cleanup_workspace!(workspace_CB)
    gpu_cleanup_workspace!(workspace_BA)
end

MPI.Finalize()