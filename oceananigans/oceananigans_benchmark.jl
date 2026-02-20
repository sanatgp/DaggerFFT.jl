#!/usr/bin/env julia
# Usage: mpiexec -np <N> julia --project=oceananigans/ oceananigans/oceananigans_benchmark.jl <mode> <Nx> <Ny> <Nz> <Rx> <Ry> <topo>
#
# Arguments:
#   mode:  "baseline" or "dagger"
#   Nx, Ny, Nz: grid dimensions
#   Rx, Ry: MPI rank layout (Rz is always 1)
#   topo: "PPP" (Periodic,Periodic,Periodic) or "PPB" (Periodic,Periodic,Bounded)
#
# Examples:
#   mpiexec -np 4  julia --project=oceananigans/ oceananigans/oceananigans_benchmark.jl baseline 896 448 224 2 2 PPP
#   mpiexec -np 28 julia --project=oceananigans/ oceananigans/oceananigans_benchmark.jl dagger  896 448 224 7 4 PPP
using MPI: MPI, mpiexec
using InteractiveUtils: versioninfo

#using MPI
MPI.Init()

using Random
Random.seed!(1234)
using KernelAbstractions

# Parse arguments
mode   = lowercase(get(ARGS, 1, "dagger"))
Nx     = parse(Int, get(ARGS, 2, "64"))
Ny     = parse(Int, get(ARGS, 3, "64"))
Nz     = parse(Int, get(ARGS, 4, "16"))
Rx     = parse(Int, get(ARGS, 5, "4"))
Ry     = parse(Int, get(ARGS, 6, "1"))
topo_s = uppercase(get(ARGS, 7, "PPP"))

include("dependencies_for_runtests.jl")
include("dependencies_for_poisson_solvers.jl")

using Oceananigans.DistributedComputations: reconstruct_global_grid, DistributedGrid, Partition
using Oceananigans.Models.NonhydrostaticModels: solve_for_pressure!

const global_baseline_fft_times = Float64[]
const global_baseline_ifft_times = Float64[]

if mode == "baseline"
    include(joinpath(@__DIR__, "distributed_fft_based_poisson_solver.jl"))
else
    using Dagger
    Dagger.accelerate!(:mpi)
    using DaggerFFT
    include(joinpath(@__DIR__, "dagger_oceananigans_integration.jl"))
end

topo_map = Dict(
    "PPP" => (Periodic, Periodic, Periodic),
    "PPB" => (Periodic, Periodic, Bounded),
    "PBB" => (Periodic, Bounded, Bounded),
)
grid_topo = get(topo_map, topo_s) do
    error("Unknown topology: $topo_s. Use PPP, PPB, or PBB.")
end

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

function random_divergent_source_term(grid::DistributedGrid)
    arch = architecture(grid)
    default_bcs = FieldBoundaryConditions()

    u_bcs = regularize_field_boundary_conditions(default_bcs, grid, :u)
    v_bcs = regularize_field_boundary_conditions(default_bcs, grid, :v)
    w_bcs = regularize_field_boundary_conditions(default_bcs, grid, :w)

    u_bcs = inject_halo_communication_boundary_conditions(u_bcs, arch.local_rank, arch.connectivity, topology(grid))
    v_bcs = inject_halo_communication_boundary_conditions(v_bcs, arch.local_rank, arch.connectivity, topology(grid))
    w_bcs = inject_halo_communication_boundary_conditions(w_bcs, arch.local_rank, arch.connectivity, topology(grid))

    Ru = XFaceField(grid, boundary_conditions=u_bcs)
    Rv = YFaceField(grid, boundary_conditions=v_bcs)
    Rw = ZFaceField(grid, boundary_conditions=w_bcs)
    U = (u=Ru, v=Rv, w=Rw)

    Nx, Ny, Nz = size(grid)
    set!(Ru, rand(size(Ru)...))
    set!(Rv, rand(size(Rv)...))
    set!(Rw, rand(size(Rw)...))

    fill_halo_regions!(Ru)
    fill_halo_regions!(Rv)
    fill_halo_regions!(Rw)

    ArrayType = array_type(arch)
    R = zeros(Nx, Ny, Nz) |> ArrayType
    launch!(arch, grid, :xyz, divergence!, grid, U.u.data, U.v.data, U.w.data, R)

    return R, U
end

arch = Distributed(child_arch, partition=Partition(Rx, Ry, 1))
local_grid = RectilinearGrid(arch, topology=grid_topo, size=(Nx, Ny, Nz), extent=(2π, 2π, 2π))
global_grid = reconstruct_global_grid(local_grid)

ϕ = CenterField(local_grid)
R, U = random_divergent_source_term(local_grid)

if mode == "dagger"
    solver = DistributedDaggerGPUFFTBasedPoissonSolver(global_grid, local_grid)
else
    solver = DistributedFFTBasedPoissonSolver(global_grid, local_grid)
end

# Warmup
solve_for_pressure!(ϕ, solver, 1, U)

if mode == "dagger"
    empty!(global_dagger_fft_times)
    empty!(global_dagger_ifft_times)
else
    empty!(global_baseline_fft_times)
    empty!(global_baseline_ifft_times)
end

num_runs = 5
for i in 1:num_runs
    solve_for_pressure!(ϕ, solver, 1, U)
end

MPI.Barrier(comm)

if rank == 0
    if mode == "dagger"
        fft_times  = global_dagger_fft_times
        ifft_times = global_dagger_ifft_times
        label = "DAGGERFFT"
    else
        fft_times  = global_baseline_fft_times
        ifft_times = global_baseline_ifft_times
        label = "BASELINE"
    end

    if length(fft_times) > 0
        avg_fft  = sum(fft_times) / length(fft_times)
        avg_ifft = sum(ifft_times) / length(ifft_times)
        avg_total = avg_fft + avg_ifft

        min_fft  = minimum(fft_times)
        max_fft  = maximum(fft_times)
        min_ifft = minimum(ifft_times)
        max_ifft = maximum(ifft_times)

        fftsize = Float64(Nx * Ny * Nz)
        floprate = 2.0 * 5.0 * fftsize * log2(fftsize) * 1e-9 / avg_total

        println("$(label): Grid ($(Nx), $(Ny), $(Nz)), Ranks ($(Rx), $(Ry), 1), Topology $(topo_s)")
        println("Forward FFT  - Avg: $(round(avg_fft, digits=6)) s, Min: $(round(min_fft, digits=6)) s, Max: $(round(max_fft, digits=6)) s")
        println("Inverse FFT  - Avg: $(round(avg_ifft, digits=6)) s, Min: $(round(min_ifft, digits=6)) s, Max: $(round(max_ifft, digits=6)) s")
        println("Total FFT Time: $(round(avg_total, digits=6)) seconds")
        println("FFT Performance: $(round(floprate, digits=2)) GFlops/s")
        flush(stdout)
    end
end
