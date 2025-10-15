using Dagger
using MPI
using LinearAlgebra
using AbstractFFTs
using KernelAbstractions
using FFTW
include("src/fft.jl")
using .DaggerFFTs
import .DaggerFFTs: FFT!, FFT, fft, fft!, ifft!, Pencil, Slab, IFFT!, ifft

Dagger.accelerate!(:mpi)
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
sz = MPI.Comm_size(comm)
#FFTW.set_num_threads(4)
N = 512
chunk = 64
A = rand(ComplexF32, N, N, N);

USE_SLAB = false

if USE_SLAB
    if rank == 0
        DA = distribute(A, Blocks(N, N, chunk); root=0, comm=comm);  
        DB = distribute(A, Blocks(chunk, N, N); root=0, comm=comm);  
    else
        DA = distribute(nothing, Blocks(N, N, chunk); root=0, comm=comm);
        DB = distribute(nothing, Blocks(chunk, N, N); root=0, comm=comm);
    end
    
    workspace_AB = DaggerFFTs.create_workspace(DA, DB)
    workspace_BA = DaggerFFTs.create_workspace(DB, DA) 
    
    if rank == 0
        println("Using SLAB decomposition (1 transpose)")
    end
else

    if rank == 0
        DA = distribute(A, Blocks(N, chunk, chunk); root=0, comm=comm);
        DB = distribute(A, Blocks(chunk, N, chunk); root=0, comm=comm);
        DC = distribute(A, Blocks(chunk, chunk, N); root=0, comm=comm);
    else
        DA = distribute(nothing, Blocks(N, chunk, chunk); root=0, comm=comm);
        DB = distribute(nothing, Blocks(chunk, N, chunk); root=0, comm=comm);
        DC = distribute(nothing, Blocks(chunk, chunk, N); root=0, comm=comm);
    end
    
    workspace_AB = DaggerFFTs.create_workspace(DA, DB)
    workspace_BC = DaggerFFTs.create_workspace(DB, DC)
    workspace_CB = DaggerFFTs.create_workspace(DC, DB)
    workspace_BA = DaggerFFTs.create_workspace(DB, DA)
    
    if rank == 0
        println("Using PENCIL decomposition (2 transposes)")
    end
end

if USE_SLAB
    DaggerFFTs.fft!(DB, DA, workspace_AB, (FFT!(), FFT!(), FFT!()), (1, 2, 3), Slab())
    DaggerFFTs.ifft!(DA, DB, workspace_BA, (IFFT!(), IFFT!(), IFFT!()), (1, 2, 3), Slab())
else
    DaggerFFTs.fft!(DC, DA, DB, workspace_AB, workspace_BC, (FFT!(), FFT!(), FFT!()), (1, 2, 3))
    DaggerFFTs.ifft!(DA, DC, DB, workspace_CB, workspace_BA, (IFFT!(), IFFT!(), IFFT!()), (1, 2, 3))
end

for iter in 1:10 
    MPI.Barrier(comm) 
    start_time = MPI.Wtime()
    
    if USE_SLAB
        @time DaggerFFTs.fft!(DB, DA, workspace_AB, (FFT!(), FFT!(), FFT!()), (1, 2, 3), Slab())
     #   @time DaggerFFTs.ifft!(DA, DB, workspace_BA, (IFFT!(), IFFT!(), IFFT!()), (1, 2, 3), Slab())
    else
        @time DaggerFFTs.fft!(DC, DA, DB, workspace_AB, workspace_BC, (FFT!(), FFT!(), FFT!()), (1, 2, 3))
     #   @time DaggerFFTs.ifft!(DA, DC, DB, workspace_CB, workspace_BA, (IFFT!(), IFFT!(), IFFT!()), (1, 2, 3))
    end
    
    elapsed_time = MPI.Wtime() - start_time
    MPI.Barrier(comm)
    time_per_transform = elapsed_time / 2.0
    fftsize = Float64(N * N * N)
    floprate = 5.0 * fftsize * log2(fftsize) * 1e-9 / time_per_transform
#=    
    if iter == 1
        if USE_SLAB
            result_array = DB
        else
            result_array = DC
        end
        
        CC = collect(result_array)
        MPI.Barrier(comm)
        if rank == 0
            reference = FFTW.fft(A)
            is_correct = CC ≈ reference
            println("Correctness check: ", is_correct)
            if !is_correct
                max_error = maximum(abs.(CC .- reference))
                println("Max error: ", max_error)
            end
        end
    end
   =# 
    if iter == 10 && rank == 0
        println("----------------------------------------------------------------------------- ")
        println("DaggerFFT performance test ($(USE_SLAB ? "SLAB" : "PENCIL") decomposition)")
        println("----------------------------------------------------------------------------- ")
        println("Size:      $(N)x$(N)x$(N)")
        println("MPI ranks: $(sz)")
        println("Time per transform: $(time_per_transform) (s)")
        println("Performance:  $(floprate) GFlops/s")
    end
    
    MPI.Barrier(comm)
end

if USE_SLAB
    DaggerFFTs.cleanup_workspace!(workspace_AB)
    DaggerFFTs.cleanup_workspace!(workspace_BA)
else
    DaggerFFTs.cleanup_workspace!(workspace_AB)
    DaggerFFTs.cleanup_workspace!(workspace_BC)
    DaggerFFTs.cleanup_workspace!(workspace_CB)
    DaggerFFTs.cleanup_workspace!(workspace_BA)
end