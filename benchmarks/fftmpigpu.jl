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
#using DaggerGPU
using FFTW
include("src/fftgpu.jl")

import .DaggerGPUFFTs: FFT!, FFT, fft, fft!, ifft!, Pencil, Slab, IFFT!, ifft

N = 512
chunk = 256
indexes(a::ArrayDomain) = a.indexes
Base.getindex(arr::GPUArraysCore.AbstractGPUArray, d::ArrayDomain) = arr[indexes(d)...]

A = CUDA.rand(ComplexF32, N, N, N);
backend = KernelAbstractions.get_backend(A) 
B = Array(A);

if rank == 0 
    DA = distribute(A, Blocks(N, chunk, chunk); root=0, comm=comm);
    DB = distribute(A, Blocks(chunk, N, chunk); root=0, comm=comm);
    DC = distribute(A, Blocks(chunk, chunk, N); root=0, comm=comm);
else
    DA = distribute(nothing, Blocks(N, chunk, chunk); root=0, comm=comm);
    DB = distribute(nothing, Blocks(chunk, N, chunk); root=0, comm=comm);
    DC = distribute(nothing, Blocks(chunk, chunk, N); root=0, comm=comm);
end

all_procs = collect(Dagger.get_processors(Dagger.MPIClusterProc()))
all_scopes = map(Dagger.ExactScope, all_procs)
scope = Dagger.UnionScope(all_scopes...)

workspace_AB = DaggerGPUFFTs.create_gpu_workspace(DA, DB)
workspace_BC = DaggerGPUFFTs.create_gpu_workspace(DB, DC)
workspace_CB = DaggerGPUFFTs.create_gpu_workspace(DC, DB)
workspace_BA = DaggerGPUFFTs.create_gpu_workspace(DB, DA)

# Warm-up
DaggerGPUFFTs.fft!(DC, DA, DB, workspace_AB, workspace_BC, scope, (FFT!(), FFT!(), FFT!()), (1, 2, 3), Pencil())
DaggerGPUFFTs.ifft!(DA, DC, DB, workspace_CB, workspace_BA, scope, (IFFT!(), IFFT!(), IFFT!()), (1, 2, 3), Pencil())

for iter in 1:10
    start_time = MPI.Wtime()
    
    # Forward FFT with Pencil decomposition
    @time DaggerGPUFFTs.fft!(DC, DA, DB, workspace_AB, workspace_BC, scope, 
                            (FFT!(), FFT!(), FFT!()), (1, 2, 3), Pencil())
    
    # Inverse FFT with Pencil decomposition
    @time DaggerGPUFFTs.ifft!(DA, DC, DB, workspace_CB, workspace_BA, scope, 
                             (IFFT!(), IFFT!(), IFFT!()), (1, 2, 3), Pencil())
    
    elapsed_time = MPI.Wtime() - start_time
    MPI.Barrier(comm)
    
    time_per_transform = elapsed_time / 2.0
    fftsize = Float64(N * N * N)
    floprate = 5.0 * fftsize * log2(fftsize) * 1e-9 / time_per_transform
#=    
    if iter == 1 && rank == 0
        # Transfer GPU results to CPU for comparison
        DA_cpu = Array(fetch(DC.chunks[1]))  # Get first chunk as CPU array
        
        println("DA first chunk (GPU->CPU) after round-trip: ", DA_cpu[1:5, 1:5, 1])
        println("Original A first slice: ", A[1:5, 1:5, 1])
        
        # Compare with reference FFT
        reference_fft = FFTW.fft(B)
        reference_ifft = FFTW.ifft(reference_fft)
        println("Reference round-trip first slice: ", reference_fft[1:5, 1:5, 1])
        
        # Check error
        max_error = maximum(abs.(DA_cpu .- reference_ifft[1:size(DA_cpu,1), 1:size(DA_cpu,2), 1:size(DA_cpu,3)]))
        println("Max error after round-trip: ", max_error)
    end
=#
    if iter == 10 && rank == 0
        println("----------------------------------------------------------------------------- ")
        println("DaggerFFT performance test (Pencil Decomposition)")
        println("----------------------------------------------------------------------------- ")
        println("Size:      $(N)x$(N)x$(N)")
        println("MPI ranks: $(sz)")
        println("Time per transform: $(time_per_transform) (s)")
        println("Performance:  $(floprate) GFlops/s")
    end
    
    MPI.Barrier(comm)
end


#=
#    if rank == 0
#        println("\n\nTesting Slab Decomposition...")
#    end
    
    # For slab, we only need two arrays and one workspace
#    workspace_AB_slab = DaggerGPUFFTs.create_gpu_workspace(DA, DB)
#    workspace_BA_slab = DaggerGPUFFTs.create_gpu_workspace(DB, DA)
    
    # Warm-up
#    DaggerGPUFFTs.fft!(DB, DA, workspace_AB_slab, scope, (FFT!(), FFT!(), FFT!()), (1, 2, 3), Slab())
 #   DaggerGPUFFTs.ifft!(DA, DB, workspace_BA_slab, scope, (IFFT!(), IFFT!(), IFFT!()), (1, 2, 3), Slab())
    
 #   for iter in 1:10
  #      start_time = MPI.Wtime()
        
        # Forward FFT with Slab decomposition
 #       @time DaggerGPUFFTs.fft!(DB, DA, workspace_AB_slab, scope, 
  #                              (FFT!(), FFT!(), FFT!()), (1, 2, 3), Slab())
        
        # Inverse FFT with Slab decomposition
   #     @time DaggerGPUFFTs.ifft!(DA, DB, workspace_BA_slab, scope, 
    #                             (IFFT!(), IFFT!(), IFFT!()), (1, 2, 3), Slab())
        
     #   elapsed_time = MPI.Wtime() - start_time
      #  MPI.Barrier(comm)
        
#        time_per_transform = elapsed_time / 2.0
 #       fftsize = Float64(N * N * N)
  #      floprate = 5.0 * fftsize * log2(fftsize) * 1e-9 / time_per_transform
        
        if iter == 10 && rank == 0
            println("----------------------------------------------------------------------------- ")
            println("DaggerFFT performance test (Slab Decomposition)")
            println("----------------------------------------------------------------------------- ")
            println("Size:      $(N)x$(N)x$(N)")
            println("MPI ranks: $(sz)")
            println("Time per transform: $(time_per_transform) (s)")
            println("Performance:  $(floprate) GFlops/s")
        end
        
        MPI.Barrier(comm)
    end
=#

DaggerGPUFFTs.cleanup_workspace!(workspace_AB)
DaggerGPUFFTs.cleanup_workspace!(workspace_BC)
DaggerGPUFFTs.cleanup_workspace!(workspace_CB)
DaggerGPUFFTs.cleanup_workspace!(workspace_BA)

MPI.Finalize()