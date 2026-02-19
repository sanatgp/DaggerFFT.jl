using Dagger
using MPI
using DaggerFFT

# Parse command-line arguments
N = parse(Int, get(ARGS, 1, "512"))
chunk = parse(Int, get(ARGS, 2, "64"))
decomp = lowercase(get(ARGS, 3, "pencil"))  # "pencil" or "slab"

Dagger.accelerate!(:mpi)
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
sz = MPI.Comm_size(comm)

A = rand(ComplexF32, N, N, N)
USE_SLAB = (decomp == "slab")

if USE_SLAB
    if rank == 0
        DA = distribute(A, Blocks(N, N, chunk); root=0, comm=comm)
        DB = distribute(A, Blocks(chunk, N, N); root=0, comm=comm)
    else
        DA = distribute(nothing, Blocks(N, N, chunk); root=0, comm=comm)
        DB = distribute(nothing, Blocks(chunk, N, N); root=0, comm=comm)
    end

    workspace_AB = create_workspace(DA, DB)
    workspace_BA = create_workspace(DB, DA)

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

    workspace_AB = create_workspace(DA, DB)
    workspace_BC = create_workspace(DB, DC)
    workspace_CB = create_workspace(DC, DB)
    workspace_BA = create_workspace(DB, DA)

    if rank == 0
        println("Using PENCIL decomposition (2 transposes)")
    end
end

# Warm-up
if USE_SLAB
    fft!(DB, DA, workspace_AB, (FFT!(), FFT!(), FFT!()), (1, 2, 3), Slab())
    ifft!(DA, DB, workspace_BA, (IFFT!(), IFFT!(), IFFT!()), (1, 2, 3), Slab())
else
    fft!(DC, DA, DB, workspace_AB, workspace_BC, (FFT!(), FFT!(), FFT!()), (1, 2, 3))
    ifft!(DA, DC, DB, workspace_CB, workspace_BA, (IFFT!(), IFFT!(), IFFT!()), (1, 2, 3))
end

for iter in 1:10
    MPI.Barrier(comm)
    start_time = MPI.Wtime()

    if USE_SLAB
        @time fft!(DB, DA, workspace_AB, (FFT!(), FFT!(), FFT!()), (1, 2, 3), Slab())
        @time ifft!(DA, DB, workspace_BA, (IFFT!(), IFFT!(), IFFT!()), (1, 2, 3), Slab())
    else
        @time fft!(DC, DA, DB, workspace_AB, workspace_BC, (FFT!(), FFT!(), FFT!()), (1, 2, 3))
        @time ifft!(DA, DC, DB, workspace_CB, workspace_BA, (IFFT!(), IFFT!(), IFFT!()), (1, 2, 3))
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
        println("DaggerFFT performance test ($(USE_SLAB ? "SLAB" : "PENCIL") decomposition)")
        println("Size:      $(N)x$(N)x$(N)")
        println("Chunk:     $(chunk)")
        println("MPI ranks: $(sz)")
        println("Time per transform: $(time_per_transform) (s)")
        println("Performance:  $(floprate) GFlops/s")
    end

    MPI.Barrier(comm)
end

if USE_SLAB
    cleanup_workspace!(workspace_AB)
    cleanup_workspace!(workspace_BA)
else
    cleanup_workspace!(workspace_AB)
    cleanup_workspace!(workspace_BC)
    cleanup_workspace!(workspace_CB)
    cleanup_workspace!(workspace_BA)
end