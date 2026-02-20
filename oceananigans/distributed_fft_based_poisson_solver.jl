# Timing-only override for Oceananigans' DistributedFFTBasedPoissonSolver.
# This file redefines ONLY solve!() to add FFT/IFFT timing instrumentation.
# The struct, constructor, compute_source_term!, and validation all come
# from Oceananigans unchanged.
#
# IMPORTANT: include() this file AFTER `using Oceananigans` and AFTER
# defining the timing arrays in Main:
#
#   const global_baseline_fft_times = Float64[]
#   const global_baseline_ifft_times = Float64[]
#   include("distributed_fft_based_poisson_solver.jl")

import MPI
import Oceananigans.Solvers: solve!
import Oceananigans.Architectures: architecture
using Oceananigans.DistributedComputations: DistributedFFTBasedPoissonSolver,
    transpose_z_to_y!, transpose_y_to_x!, transpose_x_to_y!, transpose_y_to_z!
using Oceananigans.Utils: launch!
using KernelAbstractions
using CUDA

function solve!(x, solver::DistributedFFTBasedPoissonSolver)
    storage = solver.storage
    buffer  = solver.buffer
    arch    = architecture(storage.xfield.grid)

    comm = MPI.COMM_WORLD

    MPI.Barrier(comm)
    fft_start = MPI.Wtime()

    solver.plan.forward.z!(parent(storage.zfield), buffer.z)
    transpose_z_to_y!(storage)
    solver.plan.forward.y!(parent(storage.yfield), buffer.y)
    transpose_y_to_x!(storage)
    solver.plan.forward.x!(parent(storage.xfield), buffer.x)

    MPI.Barrier(comm)
    fft_time = MPI.Wtime() - fft_start

    λ = solver.eigenvalues
    x̂ = b̂ = parent(storage.xfield)

    launch!(arch, storage.xfield.grid, :xyz,
            _baseline_solve_poisson_in_spectral_space!, x̂, b̂, λ[1], λ[2], λ[3])

    if arch.local_rank == 0
        @allowscalar x̂[1, 1, 1] = 0
    end

    # TIME ONLY INVERSE FFT
    MPI.Barrier(comm)
    ifft_start = MPI.Wtime()

    solver.plan.backward.x!(parent(storage.xfield), buffer.x)
    transpose_x_to_y!(storage)
    solver.plan.backward.y!(parent(storage.yfield), buffer.y)
    transpose_y_to_z!(storage)
    solver.plan.backward.z!(parent(storage.zfield), buffer.z)

    MPI.Barrier(comm)
    ifft_time = MPI.Wtime() - ifft_start

    push!(Main.global_baseline_fft_times, fft_time)
    push!(Main.global_baseline_ifft_times, ifft_time)

    launch!(arch, solver.local_grid, :xyz,
            _baseline_copy_real_component!, x, parent(storage.zfield))

    return x
end

@kernel function _baseline_solve_poisson_in_spectral_space!(x̂, b̂, λx, λy, λz)
    i, j, k = @index(Global, NTuple)
    @inbounds x̂[i, j, k] = - b̂[i, j, k] / (λx[i] + λy[j] + λz[k])
end

@kernel function _baseline_copy_real_component!(ϕ, ϕc)
    i, j, k = @index(Global, NTuple)
    @inbounds ϕ[i, j, k] = real(ϕc[i, j, k])
end