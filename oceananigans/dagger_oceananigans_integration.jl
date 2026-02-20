import FFTW
using Dagger
using MPI
using LinearAlgebra
using AbstractFFTs
using KernelAbstractions
using CUDA

#include("../src/fftgpu.jl")

using DaggerFFT
using DaggerFFT: FFT!, IFFT!, Pencil, Slab, create_gpu_workspace, gpu_fft!, gpu_ifft!, gpu_cleanup_workspace!
using GPUArraysCore
using Oceananigans.Grids: XYZRegularRG, XYRegularRG, XZRegularRG, YZRegularRG, RectilinearGrid
import Oceananigans.Solvers: poisson_eigenvalues, solve!
import Oceananigans.Architectures: architecture, child_architecture, GPU, CPU
import Oceananigans.Fields: interior
using Oceananigans.Grids: topology, size, stretched_dimensions
using Oceananigans.Fields: CenterField
using Oceananigans.DistributedComputations: TransposableField, partition_coordinate
using Oceananigans.Utils: launch!
import Oceananigans.Models.NonhydrostaticModels: compute_source_term!
using GPUArraysCore

# Global timing arrays
const global_dagger_fft_times = Float64[]
const global_dagger_ifft_times = Float64[]

struct DistributedDaggerGPUFFTBasedPoissonSolver{F, L, λ, D, W, S, DT}
    global_grid :: F
    local_grid :: L
    eigenvalues :: λ
    dagger_arrays :: D
    workspaces :: W
    storage :: S
    decomp_type :: DT
end

architecture(solver::DistributedDaggerGPUFFTBasedPoissonSolver) =
    architecture(solver.global_grid)

indexes(a::ArrayDomain) = a.indexes
Base.getindex(arr::GPUArraysCore.AbstractGPUArray, d::ArrayDomain) = arr[indexes(d)...]

    
function DistributedDaggerGPUFFTBasedPoissonSolver(global_grid, local_grid, planner_flag=FFTW.PATIENT)
    
    validate_poisson_solver_distributed_grid(global_grid)
    validate_poisson_solver_configuration(global_grid, local_grid)
    
    FT = ComplexF64
    
    if !MPI.Initialized()
        MPI.Init()
    end
    Dagger.accelerate!(:mpi)
    
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    sz = MPI.Comm_size(comm)
    
    # Set GPU device
    CUDA.device!(rank % CUDA.ndevices())
    
    Nx, Ny, Nz = size(global_grid)
    arch = architecture(local_grid)
    Rx, Ry, Rz = arch.ranks
    
    is_slab = (Rx == 1) || (Ry == 1)
    decomp_type = is_slab ? Slab() : Pencil()
    
    # Create DArrays with CuArrays for GPU
    if is_slab
        if Rx > 1
            if rank == 0
                dummy_data = CUDA.zeros(FT, Nx, Ny, Nz)
                DA = distribute(dummy_data, Blocks(Nx, div(Ny, Rx), Nz); root=0, comm=comm)
                DB = distribute(dummy_data, Blocks(div(Nx, Rx), Ny, Nz); root=0, comm=comm)
            else
                DA = distribute(nothing, Blocks(Nx, div(Ny, Rx), Nz); root=0, comm=comm)
                DB = distribute(nothing, Blocks(div(Nx, Rx), Ny, Nz); root=0, comm=comm)
            end
        else
            if rank == 0
                dummy_data = CUDA.zeros(FT, Nx, Ny, Nz)
                DA = distribute(dummy_data, Blocks(Nx, Ny, div(Nz, Ry)); root=0, comm=comm)
                DB = distribute(dummy_data, Blocks(div(Nx, Ry), Ny, Nz); root=0, comm=comm)
            else
                DA = distribute(nothing, Blocks(Nx, Ny, div(Nz, Ry)); root=0, comm=comm)
                DB = distribute(nothing, Blocks(div(Nx, Ry), Ny, Nz); root=0, comm=comm)
            end
        end
        
        workspace_AB = create_gpu_workspace(DA, DB)
        workspace_BA = create_gpu_workspace(DB, DA)
        
        dagger_arrays = (DA=DA, DB=DB)
        workspaces = (AB=workspace_AB, BA=workspace_BA)
        
    else  # Pencil
        chunk_x = div(Nx, Rx)
        chunk_y = div(Ny, Ry)
        chunk_z = div(Nz, max(Rx, Ry))
        
        if rank == 0
            dummy_data = CUDA.zeros(FT, Nx, Ny, Nz)
            DA = distribute(dummy_data, Blocks(Nx, chunk_y, chunk_z); root=0, comm=comm)
            DB = distribute(dummy_data, Blocks(chunk_x, Ny, chunk_z); root=0, comm=comm)
            DC = distribute(dummy_data, Blocks(chunk_x, chunk_y, Nz); root=0, comm=comm)
        else
            DA = distribute(nothing, Blocks(Nx, chunk_y, chunk_z); root=0, comm=comm)
            DB = distribute(nothing, Blocks(chunk_x, Ny, chunk_z); root=0, comm=comm)
            DC = distribute(nothing, Blocks(chunk_x, chunk_y, Nz); root=0, comm=comm)
        end
        
        workspace_AB = create_gpu_workspace(DA, DB)
        workspace_BC = create_gpu_workspace(DB, DC)
        workspace_CB = create_gpu_workspace(DC, DB)
        workspace_BA = create_gpu_workspace(DB, DA)
        
        dagger_arrays = (DA=DA, DB=DB, DC=DC)
        workspaces = (AB=workspace_AB, BC=workspace_BC, CB=workspace_CB, BA=workspace_BA)
    end
    
    storage = TransposableField(CenterField(local_grid), FT)
    
    topo = (TX, TY, TZ) = topology(global_grid)
    λx = dropdims(poisson_eigenvalues(global_grid.Nx, global_grid.Lx, 1, TX()), dims=(2, 3))
    λy = dropdims(poisson_eigenvalues(global_grid.Ny, global_grid.Ly, 2, TY()), dims=(1, 3))
    λz = dropdims(poisson_eigenvalues(global_grid.Nz, global_grid.Lz, 3, TZ()), dims=(1, 2))
    
    eigenvalues = (λx, λy, λz)
    
    return DistributedDaggerGPUFFTBasedPoissonSolver(global_grid, local_grid, eigenvalues, 
                                                     dagger_arrays, workspaces, storage, decomp_type)
end

function compute_source_term!(solver::DistributedDaggerGPUFFTBasedPoissonSolver, rhs)
    divergence_free_rhs = solver.storage.zfield
    arch = architecture(solver.local_grid) 
    grid = solver.local_grid
    u, v, w = rhs.u, rhs.v, rhs.w
    
    launch!(arch, grid, :xyz, _compute_divergence!, 
            parent(divergence_free_rhs), 
            parent(u), parent(v), parent(w))
    
    return nothing
end

@kernel function _compute_divergence!(div, u, v, w)
    i, j, k = @index(Global, NTuple)
    @inbounds div[i, j, k] = (u[i+1, j, k] - u[i, j, k]) +
                              (v[i, j+1, k] - v[i, j, k]) +  
                              (w[i, j, k+1] - w[i, j, k])
end

@kernel function _copy_real_from_dagger!(output, input)
    i, j, k = @index(Global, NTuple)
    @inbounds output[i, j, k] = real(input[i, j, k])
end


function solve!(x, solver::DistributedDaggerGPUFFTBasedPoissonSolver)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    
    Nx, Ny, Nz = size(solver.global_grid)
    λx, λy, λz = solver.eigenvalues
    
    b_local = parent(solver.storage.zfield)
    local_Nx, local_Ny, local_Nz = size(b_local)
    
    # Convert to CuArray and copy to DArray
    b_gpu = CuArray(b_local)
    DA = solver.dagger_arrays.DA

    for (idx, chunk) in enumerate(DA.chunks)
        if chunk.handle.rank == rank
            chunk_data = fetch(chunk)
            copyto!(chunk_data, b_gpu)
        end
    end
    
    MPI.Barrier(comm)
    
    # Create scope for Dagger operations
    all_procs = collect(Dagger.get_processors(Dagger.MPIClusterProc()))
    all_scopes = map(Dagger.ExactScope, all_procs)
    scope = Dagger.UnionScope(all_scopes...)
    
    # TIME FORWARD FFT
    fft_start = MPI.Wtime()
    
    if solver.decomp_type isa Slab
        DaggerFFT.gpu_fft!(solver.dagger_arrays.DB, solver.dagger_arrays.DA, solver.workspaces.AB,
             scope, (FFT!(), FFT!(), FFT!()), (1, 2, 3), solver.decomp_type)
        result_array = solver.dagger_arrays.DB
    else
        DaggerFFT.gpu_fft!(solver.dagger_arrays.DC, solver.dagger_arrays.DA, solver.dagger_arrays.DB,
             solver.workspaces.AB, solver.workspaces.BC, scope,
             (FFT!(), FFT!(), FFT!()), (1, 2, 3), solver.decomp_type)
        result_array = solver.dagger_arrays.DC
    end
    
    MPI.Barrier(comm)
    fft_time = MPI.Wtime() - fft_start
    
    # Spectral solve on GPU
    for (idx, chunk) in enumerate(result_array.chunks)
        if chunk.handle.rank == rank
            chunk_data = fetch(chunk)
            subdomain = result_array.subdomains[idx]
            launch_spectral_solve_gpu!(chunk_data, subdomain, λx, λy, λz, rank)
        end
    end
    
    MPI.Barrier(comm)
    
    # TIME INVERSE FFT
    ifft_start = MPI.Wtime()
    
    if solver.decomp_type isa Slab
        DaggerFFT.gpu_ifft!(solver.dagger_arrays.DA, solver.dagger_arrays.DB, solver.workspaces.BA,
              scope, (IFFT!(), IFFT!(), IFFT!()), (1, 2, 3), solver.decomp_type)
    else
        DaggerFFT.gpu_ifft!(solver.dagger_arrays.DA, solver.dagger_arrays.DC, solver.dagger_arrays.DB,
              solver.workspaces.CB, solver.workspaces.BA, scope,
              (IFFT!(), IFFT!(), IFFT!()), (1, 2, 3), solver.decomp_type)
    end
    
    MPI.Barrier(comm)
    ifft_time = MPI.Wtime() - ifft_start
    
    # Store timings
    push!(global_dagger_fft_times, fft_time)
    push!(global_dagger_ifft_times, ifft_time)
    
    # Extract solution - EXACTLY like baseline
    arch = architecture(solver.local_grid)
    
    for (idx, chunk) in enumerate(DA.chunks)
        if chunk.handle.rank == rank
            chunk_data = fetch(chunk)
            # Copy real part directly on GPU using kernel
            launch!(arch, solver.local_grid, :xyz,
                    _copy_real_from_dagger!, parent(x), chunk_data)
        end
    end
    
    return x
end

function launch_spectral_solve_gpu!(chunk_data::CuArray, subdomain, λx, λy, λz, rank)
    λx_gpu = CuArray(λx)
    λy_gpu = CuArray(λy)
    λz_gpu = CuArray(λz)
    
    # Get global index ranges
    i_global_range = subdomain.indexes[1]
    j_global_range = subdomain.indexes[2]
    k_global_range = subdomain.indexes[3]
    
    # Launch CUDA kernel
    kernel = @cuda launch=false spectral_solve_kernel!(
        chunk_data, λx_gpu, λy_gpu, λz_gpu,
        first(i_global_range), first(j_global_range), first(k_global_range), rank
    )
    
    config = launch_configuration(kernel.fun)
    threads = min(config.threads, 256)
    blocks = cld(length(chunk_data), threads)
    
    kernel(chunk_data, λx_gpu, λy_gpu, λz_gpu,
           first(i_global_range), first(j_global_range), first(k_global_range), rank;
           threads, blocks)
    
    CUDA.synchronize()
end

function spectral_solve_kernel!(chunk_data, λx, λy, λz, i_offset, j_offset, k_offset, rank)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    nx, ny, nz = size(chunk_data)
    total = nx * ny * nz
    
    if idx <= total
        # Convert linear index to 3D indices
        k_local = div(idx - 1, nx * ny) + 1
        rem = (idx - 1) % (nx * ny)
        j_local = div(rem, nx) + 1
        i_local = rem % nx + 1
        
        # Convert to global indices
        i_global = i_local + i_offset - 1
        j_global = j_local + j_offset - 1
        k_global = k_local + k_offset - 1
        
        if i_global == 1 && j_global == 1 && k_global == 1 && rank == 0
            chunk_data[i_local, j_local, k_local] = 0
        else
            denominator = λx[i_global] + λy[j_global] + λz[k_global]
            chunk_data[i_local, j_local, k_local] = -chunk_data[i_local, j_local, k_local] / denominator
        end
    end
    
    return nothing
end

validate_poisson_solver_distributed_grid(global_grid) =
    throw("Grids other than the RectilinearGrid are not supported")

function validate_poisson_solver_distributed_grid(global_grid::RectilinearGrid)
    TX, TY, TZ = topology(global_grid)
    
    if (TY == Bounded && TZ == Periodic) || (TX == Bounded && TY == Periodic) || (TX == Bounded && TZ == Periodic)
        throw("Distributed Poisson solvers do not support grids with topology ($TX, $TY, $TZ)")
    end
    
    if !(global_grid isa YZRegularRG) && !(global_grid isa XYRegularRG) && !(global_grid isa XZRegularRG)
        throw("The provided grid is stretched in directions $(stretched_dimensions(global_grid))")
    end
    
    return nothing
end

function validate_poisson_solver_configuration(global_grid, local_grid)
    Rx, Ry, Rz = architecture(local_grid).ranks
    Rz == 1 || throw("Non-singleton ranks in the vertical are not supported")
    
    if Ry > 1 && global_grid.Nz % Ry != 0
        throw("The number of ranks in the y-direction are $(Ry) with Nz = $(global_grid.Nz)")
    end
    
    if Rx > 1 && global_grid.Ny % Rx != 0
        throw("The number of ranks in the x-direction are $(Rx) with Ny = $(global_grid.Ny)")
    end
    
    return nothing
end