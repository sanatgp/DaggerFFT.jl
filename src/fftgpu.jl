# DaggerGPUFFTs.jl 
module DaggerGPUFFTs

using AbstractFFTs
using LinearAlgebra
using Dagger: DArray, @spawn, InOut, In
import Dagger: Chunk, DArray, @spawn, InOut, In, memory_space, move!, move
using Dagger
using KernelAbstractions
using MPI
#using DaggerGPU
using CUDA
using GPUArraysCore
using NVTX
import MPI: Comm_rank, Comm_size

abstract type Decomposition end
struct Pencil <: Decomposition end
struct Slab <: Decomposition end

struct FFT end
struct RFFT end
struct IRFFT end
struct IFFT end
struct FFT! end
struct RFFT! end
struct IRFFT! end
struct IFFT! end

export FFT, RFFT, IRFFT, IFFT, FFT!, RFFT!, IRFFT!, IFFT!, fft, ifft
export GPUFFTWorkspace, create_gpu_workspace, fft!, ifft!
export Pencil, Slab

struct HierarchicalInfo
    node_rank::Int
    local_rank::Int
    gpus_per_node::Int
    total_nodes::Int
    is_aggregator::Bool
    
    node_comm::MPI.Comm
    inter_comm::MPI.Comm
end

struct CoalescedPattern
    src_idx::Int
    dst_idx::Int
    overlap::NTuple{3, UnitRange{Int}}
    src_indices::NTuple{3, UnitRange{Int}}
    dst_indices::NTuple{3, UnitRange{Int}}
    buffer_offset::Int
    buffer_size::Int
    target_node::Int
end

struct PeerCommInfo{T}
    peer_rank::Int
    total_size::Int
    patterns::Vector{CoalescedPattern}
    send_buffer::Union{CuArray{T,1}, Nothing}
    recv_buffer::Union{CuArray{T,1}, Nothing}
    send_stream::CUDA.CuStream
    recv_stream::CUDA.CuStream
end

mutable struct CoalescedWorkspace{T}
    send_peers::Dict{Int, PeerCommInfo{T}}
    recv_peers::Dict{Int, PeerCommInfo{T}}
    local_patterns::Vector{CoalescedPattern}
    
    hierarchical::Union{HierarchicalInfo, Nothing}
    node_send_buffers::Dict{Int, CuArray{T,1}}
    node_recv_buffers::Dict{Int, CuArray{T,1}}
    intra_send_patterns::Dict{Int, Vector{CoalescedPattern}}
    intra_recv_patterns::Dict{Int, Vector{CoalescedPattern}}
    
    send_reqs::Vector{MPI.Request}
    recv_reqs::Vector{MPI.Request}
    node_send_reqs::Vector{MPI.Request}
    node_recv_reqs::Vector{MPI.Request}
    
    send_ranks::Vector{Int}
    recv_ranks::Vector{Int}
    target_nodes::Vector{Int}
    source_nodes::Vector{Int}
    
    src_shape::NTuple{3, Int}
    dst_shape::NTuple{3, Int}
    use_cuda_aware::Bool
    chunk_cache::Dict{Int, Any}
    use_hierarchical::Bool
    
    function CoalescedWorkspace{T}(args...) where T
        new(args...)
    end
end

function detect_hierarchical_topology(; gpus_per_node::Int=0)::HierarchicalInfo
    comm = MPI.COMM_WORLD
    rank = Comm_rank(comm)
    nranks = Comm_size(comm)
    
    if gpus_per_node == 0
        gpus_per_node = length(CUDA.devices())
        gpus_per_node = MPI.Allreduce(gpus_per_node, MPI.MIN, comm)
    end
    
    node_rank = rank ÷ gpus_per_node
    local_rank = rank % gpus_per_node
    total_nodes = cld(nranks, gpus_per_node)
    is_aggregator = (local_rank == 0)
    
    node_comm = MPI.Comm_split(comm, node_rank, local_rank)
    
    inter_comm = if is_aggregator
        MPI.Comm_split(comm, 1, node_rank)
    else
        MPI.Comm_split(comm, nothing, 0)
    end
    
    return HierarchicalInfo(node_rank, local_rank, gpus_per_node, total_nodes,
                           is_aggregator, node_comm, inter_comm)
end

@inline function launch_1d!(f, n, stream, threads, args...)
    blocks = cld(n, threads)
    @cuda threads=threads blocks=blocks stream=stream f(args...)
end

const GPU_PLAN_CACHE = Dict{Tuple{DataType, NTuple, Type, Tuple}, Any}()

function plan_transform(transform, A::CuArray, dims; kwargs...)
    if transform isa RFFT
        return CUDA.CUFFT.plan_rfft(A, dims; kwargs...)
    elseif transform isa FFT
        return CUDA.CUFFT.plan_fft(A, dims; kwargs...)
    elseif transform isa IRFFT
        return CUDA.CUFFT.plan_irfft(A, dims; kwargs...)
    elseif transform isa IFFT
        return CUDA.CUFFT.plan_ifft(A, dims; kwargs...)
    elseif transform isa RFFT!
        return CUDA.CUFFT.plan_rfft!(A, dims; kwargs...)
    elseif transform isa FFT!
        return CUDA.CUFFT.plan_fft!(A, dims; kwargs...)
    elseif transform isa IRFFT!
        return CUDA.CUFFT.plan_irfft!(A, dims; kwargs...)
    elseif transform isa IFFT!
        return CUDA.CUFFT.plan_ifft!(A, dims; kwargs...)
    else
        throw(ArgumentError("Unknown transform type"))
    end
end

function get_or_create_gpu_plan(transform, A::CuArray, dims; kwargs...)
    dims_tuple = dims isa Tuple ? dims : (dims,)
    key = (eltype(A), size(A), typeof(transform), dims_tuple)
    
    if !haskey(GPU_PLAN_CACHE, key)
        GPU_PLAN_CACHE[key] = plan_transform(transform, A, dims; kwargs...)
    end
    
    return GPU_PLAN_CACHE[key]
end

function apply_gpu_fft!(out_part, in_part, transform, dim)
    plan = get_or_create_gpu_plan(transform, in_part, dim)
    mul!(out_part, plan, in_part)
    return nothing
end

apply_gpu_fft!(inout_part, transform, dim) = apply_gpu_fft!(inout_part, inout_part, transform, dim)

function create_gpu_workspace(src::DArray{T,3}, dst::DArray{T,3}; 
                             use_hierarchical::Bool=true, 
                             gpus_per_node::Int=0) where T
    comm = MPI.COMM_WORLD
    rank = Comm_rank(comm)
    nranks = Comm_size(comm)
    
    CUDA.device!(rank % length(CUDA.devices()))
    
    cuda_aware = check_cuda_aware_mpi()
    if !cuda_aware
        @warn "CUDA-aware MPI not detected. Performance may be suboptimal."
    end
    
    current_device = CUDA.device()
    @assert current_device isa CUDA.CuDevice "No CUDA device available"
    
    hierarchical = if use_hierarchical && nranks > 1
        detect_hierarchical_topology(gpus_per_node=gpus_per_node)
    else
        nothing
    end
    
    send_peer_patterns = Dict{Int, Vector{Tuple}}()
    recv_peer_patterns = Dict{Int, Vector{Tuple}}()
    local_patterns = CoalescedPattern[]
    
    node_send_patterns = Dict{Int, Vector{Tuple}}()
    node_recv_patterns = Dict{Int, Vector{Tuple}}()
    
    for (dst_idx, dst_chunk) in enumerate(dst.chunks)
        dst_rank = dst_chunk.handle.rank
        dst_domain = dst.subdomains[dst_idx]
        dst_node = hierarchical !== nothing ? dst_rank ÷ hierarchical.gpus_per_node : -1
        
        for (src_idx, src_chunk) in enumerate(src.chunks)
            src_rank = src_chunk.handle.rank
            src_domain = src.subdomains[src_idx]
            src_node = hierarchical !== nothing ? src_rank ÷ hierarchical.gpus_per_node : -1
            
            overlap = compute_overlap(src_domain, dst_domain)
            
            if !isempty(overlap[1]) && !isempty(overlap[2]) && !isempty(overlap[3])
                buffer_size = prod(length.(overlap))
                
                src_indices = ntuple(i -> overlap[i] .- first(src_domain.indexes[i]) .+ 1, 3)
                dst_indices = ntuple(i -> overlap[i] .- first(dst_domain.indexes[i]) .+ 1, 3)
                
                if src_rank == rank && dst_rank != rank
                    if use_hierarchical && hierarchical !== nothing && dst_node != hierarchical.node_rank
                        if !haskey(node_send_patterns, dst_node)
                            node_send_patterns[dst_node] = []
                        end
                        push!(node_send_patterns[dst_node], 
                             (src_idx, dst_idx, overlap, src_indices, dst_indices, buffer_size, dst_node))
                    else
                        if !haskey(send_peer_patterns, dst_rank)
                            send_peer_patterns[dst_rank] = []
                        end
                        push!(send_peer_patterns[dst_rank], 
                             (src_idx, dst_idx, overlap, src_indices, dst_indices, buffer_size, dst_node))
                    end
                    
                elseif dst_rank == rank && src_rank != rank
                    if use_hierarchical && hierarchical !== nothing && src_node != hierarchical.node_rank
                        if !haskey(node_recv_patterns, src_node)
                            node_recv_patterns[src_node] = []
                        end
                        push!(node_recv_patterns[src_node], 
                             (src_idx, dst_idx, overlap, src_indices, dst_indices, buffer_size, src_node))
                    else
                        if !haskey(recv_peer_patterns, src_rank)
                            recv_peer_patterns[src_rank] = []
                        end
                        push!(recv_peer_patterns[src_rank], 
                             (src_idx, dst_idx, overlap, src_indices, dst_indices, buffer_size, src_node))
                    end
                    
                elseif src_rank == rank && dst_rank == rank
                    pattern = CoalescedPattern(src_idx, dst_idx, overlap, src_indices, dst_indices, 
                                             0, buffer_size, dst_node)
                    push!(local_patterns, pattern)
                end
            end
        end
    end
    
    send_peers = Dict{Int, PeerCommInfo{T}}()
    recv_peers = Dict{Int, PeerCommInfo{T}}()
    
    for (peer_rank, pattern_list) in send_peer_patterns
        total_size = 0
        patterns = CoalescedPattern[]
        
        for (src_idx, dst_idx, overlap, src_indices, dst_indices, buffer_size, target_node) in pattern_list
            pattern = CoalescedPattern(src_idx, dst_idx, overlap, src_indices, dst_indices, 
                                     total_size, buffer_size, target_node)
            push!(patterns, pattern)
            total_size += buffer_size
        end
        
        send_buffer = CUDA.zeros(T, total_size)
        send_stream = CUDA.CuStream()
        recv_stream = CUDA.CuStream()
        
        peer_info = PeerCommInfo{T}(peer_rank, total_size, patterns, send_buffer, nothing, 
                                   send_stream, recv_stream)
        send_peers[peer_rank] = peer_info
    end
    
    for (peer_rank, pattern_list) in recv_peer_patterns
        total_size = 0
        patterns = CoalescedPattern[]
        
        for (src_idx, dst_idx, overlap, src_indices, dst_indices, buffer_size, source_node) in pattern_list
            pattern = CoalescedPattern(src_idx, dst_idx, overlap, src_indices, dst_indices, 
                                     total_size, buffer_size, source_node)
            push!(patterns, pattern)
            total_size += buffer_size
        end
        
        recv_buffer = CUDA.zeros(T, total_size)
        send_stream = CUDA.CuStream()
        recv_stream = CUDA.CuStream()
        
        peer_info = PeerCommInfo{T}(peer_rank, total_size, patterns, nothing, recv_buffer,
                                   send_stream, recv_stream)
        recv_peers[peer_rank] = peer_info
    end
    
    node_send_buffers = Dict{Int, CuArray{T,1}}()
    node_recv_buffers = Dict{Int, CuArray{T,1}}()
    intra_send_patterns = Dict{Int, Vector{CoalescedPattern}}()
    intra_recv_patterns = Dict{Int, Vector{CoalescedPattern}}()
    
    if use_hierarchical && hierarchical !== nothing
        for (target_node, pattern_list) in node_send_patterns
            if hierarchical.is_aggregator
                total_size = sum(p[6] for p in pattern_list)
                node_send_buffers[target_node] = CUDA.zeros(T, total_size)
            end
            
            patterns = CoalescedPattern[]
            offset = 0
            for (src_idx, dst_idx, overlap, src_indices, dst_indices, buffer_size, _) in pattern_list
                pattern = CoalescedPattern(src_idx, dst_idx, overlap, src_indices, dst_indices, 
                                         offset, buffer_size, target_node)
                push!(patterns, pattern)
                offset += buffer_size
            end
            intra_send_patterns[target_node] = patterns
        end
        
        for (source_node, pattern_list) in node_recv_patterns
            if hierarchical.is_aggregator
                total_size = sum(p[6] for p in pattern_list)
                node_recv_buffers[source_node] = CUDA.zeros(T, total_size)
            end
            
            patterns = CoalescedPattern[]
            offset = 0
            for (src_idx, dst_idx, overlap, src_indices, dst_indices, buffer_size, _) in pattern_list
                pattern = CoalescedPattern(src_idx, dst_idx, overlap, src_indices, dst_indices, 
                                         offset, buffer_size, source_node)
                push!(patterns, pattern)
                offset += buffer_size
            end
            intra_recv_patterns[source_node] = patterns
        end
    end
    
    send_ranks = sort(collect(keys(send_peers)))
    recv_ranks = sort(collect(keys(recv_peers)))
    target_nodes = sort(collect(keys(node_send_buffers)))
    source_nodes = sort(collect(keys(node_recv_buffers)))
    
    send_reqs = Vector{MPI.Request}(undef, length(send_peers))
    recv_reqs = Vector{MPI.Request}(undef, length(recv_peers))
    node_send_reqs = Vector{MPI.Request}(undef, length(target_nodes))
    node_recv_reqs = Vector{MPI.Request}(undef, length(source_nodes))
    
    chunk_cache = Dict{Int, Any}()
    
    workspace = CoalescedWorkspace{T}(
        send_peers, recv_peers, local_patterns, hierarchical,
        node_send_buffers, node_recv_buffers, intra_send_patterns, intra_recv_patterns,
        send_reqs, recv_reqs, node_send_reqs, node_recv_reqs,
        send_ranks, recv_ranks, target_nodes, source_nodes,
        size(src), size(dst), cuda_aware, chunk_cache, use_hierarchical
    )
    
    return workspace
end

function coalesced_redistribute!(dst::DArray{T,3}, src::DArray{T,3}, workspace::CoalescedWorkspace{T}; phase_id::Int=1) where T
#    if workspace.use_hierarchical && workspace.hierarchical !== nothing
 #       hierarchical_redistribute!(dst, src, workspace, phase_id)
#    else
        flat_redistribute!(dst, src, workspace, phase_id)
  #  end
   # return nothing
end

function flat_redistribute!(dst::DArray{T,3}, src::DArray{T,3}, workspace::CoalescedWorkspace{T}, phase_id::Int) where T
    comm = MPI.COMM_WORLD
    rank = Comm_rank(comm)
    
    NVTX.@range "FLAT_REDISTRIBUTE" begin
        
        NVTX.@range "CACHE_CHUNKS" begin
            workspace.chunk_cache = Dict{Int, Any}()
            
            for (idx, chunk) in enumerate(src.chunks)
                if chunk.handle.rank == rank
                    workspace.chunk_cache[idx] = fetch(chunk)
                end
            end
            
            for (idx, chunk) in enumerate(dst.chunks)
                if chunk.handle.rank == rank
                    workspace.chunk_cache[idx + 1000] = fetch(chunk)
                end
            end
        end
        
        NVTX.@range "POST_RECEIVES" begin
            for (i, src_rank) in enumerate(workspace.recv_ranks)
                peer_info = workspace.recv_peers[src_rank]
                tag = generate_tag(src_rank, phase_id)
                workspace.recv_reqs[i] = MPI.Irecv!(peer_info.recv_buffer, src_rank, tag, comm)
            end
        end
        
        NVTX.@range "PACK_SEND" begin
            for dst_rank in workspace.send_ranks
                peer_info = workspace.send_peers[dst_rank]
                pack_peer!(peer_info, workspace.chunk_cache)
            end
            
            for (i, dst_rank) in enumerate(workspace.send_ranks)
                tag = generate_tag(rank, phase_id)
                peer_info = workspace.send_peers[dst_rank]
                workspace.send_reqs[i] = MPI.Isend(peer_info.send_buffer, dst_rank, tag, comm)
            end
        end
        
        NVTX.@range "LOCAL_COPIES" begin
            for pattern in workspace.local_patterns
                src_key = pattern.src_idx
                dst_key = pattern.dst_idx + 1000
                
                if haskey(workspace.chunk_cache, src_key) && haskey(workspace.chunk_cache, dst_key)
                    src_data = workspace.chunk_cache[src_key]
                    dst_data = workspace.chunk_cache[dst_key]
                    
                    nx, ny, nz = length.(pattern.dst_indices)
                    dst_xs, dst_ys, dst_zs = first.(pattern.dst_indices)
                    src_xs, src_ys, src_zs = first.(pattern.src_indices)
                    
                    launch_1d!(local_copy_kernel!, pattern.buffer_size, CUDA.default_stream(), 256,
                              dst_data, src_data, dst_xs, dst_ys, dst_zs, 
                              src_xs, src_ys, src_zs, nx, ny, nz)
                end
            end
        end
        
        NVTX.@range "RECV_UNPACK" begin
            remaining = Set(1:length(workspace.recv_ranks))
            
            while !isempty(remaining)
                for i in copy(remaining)
                    if i <= length(workspace.recv_reqs)
                        flag, _ = MPI.Test!(workspace.recv_reqs[i])
                        
                        if flag
                            src_rank = workspace.recv_ranks[i]
                            peer_info = workspace.recv_peers[src_rank]
                            
                            unpack_peer!(peer_info, workspace.chunk_cache)
                            
                            delete!(remaining, i)
                        end
                    else
                        delete!(remaining, i)
                    end
                end
                
                if !isempty(remaining)
                    yield()
                end
            end
            
        end
        
    end
end


function pack_peer!(info::PeerCommInfo{T}, cache::Dict{Int,Any}) where T
    s = info.send_stream
    
    for pat in info.patterns
        if haskey(cache, pat.src_idx)
            src = cache[pat.src_idx]::CuArray{T,3}
            nx, ny, nz = length.(pat.src_indices)
            xs, ys, zs = first.(pat.src_indices)
            
            launch_1d!(pack_kernel!, pat.buffer_size, s, 256,
                      info.send_buffer, src, xs, ys, zs, nx, ny, nz, pat.buffer_offset)
        end
    end
    
    return nothing
end


function unpack_peer!(info::PeerCommInfo{T}, cache::Dict{Int,Any}) where T
    s = info.recv_stream
    
    # Launch kernels asynchronously - NO synchronization
    for pat in info.patterns
        dst_key = pat.dst_idx + 1000
        if haskey(cache, dst_key)
            dst = cache[dst_key]::CuArray{T,3}
            nx, ny, nz = length.(pat.dst_indices)
            xs, ys, zs = first.(pat.dst_indices)
            
            launch_1d!(unpack_kernel!, pat.buffer_size, s, 256,
                      dst, info.recv_buffer, xs, ys, zs, nx, ny, nz, pat.buffer_offset)
        end
    end
    
    return nothing
end


function pack_kernel!(send_buffer, src, xs, ys, zs, nx, ny, nz, offset)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx <= nx * ny * nz
        k = div(idx - 1, nx * ny)
        rem = (idx - 1) % (nx * ny)
        j = div(rem, nx)
        i = rem % nx
        
        send_buffer[offset + idx] = src[xs + i, ys + j, zs + k]
    end
    
    return nothing
end


function unpack_kernel!(dst, recv_buffer, xs, ys, zs, nx, ny, nz, offset)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx <= nx * ny * nz
        k = div(idx - 1, nx * ny)
        rem = (idx - 1) % (nx * ny)
        j = div(rem, nx)
        i = rem % nx
        
        dst[xs + i, ys + j, zs + k] = recv_buffer[offset + idx]
    end
    
    return nothing
end


function local_copy_kernel!(dst, src, dst_xs, dst_ys, dst_zs, src_xs, src_ys, src_zs, nx, ny, nz)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx <= nx * ny * nz
        k = div(idx - 1, nx * ny)
        rem = (idx - 1) % (nx * ny)
        j = div(rem, nx)
        i = rem % nx
        
        dst[dst_xs + i, dst_ys + j, dst_zs + k] = src[src_xs + i, src_ys + j, src_zs + k]
    end
    
    return nothing
end


function fft!(C::DArray{T,3}, A::DArray{T,3}, B::DArray{T,3},
              workspace_AB::CoalescedWorkspace{T}, workspace_BC::CoalescedWorkspace{T},
              scope, transforms, dims, ::Pencil) where T
    
    spawn_datadeps(scheduler=:locality_aware) do
        Dagger.with_options(;scope) do
            for idx in eachindex(A.chunks)
                @spawn apply_gpu_fft!(A.chunks[idx], In(transforms[1]), In(dims[1]))
            end
        end
    end
    
    coalesced_redistribute!(B, A, workspace_AB; phase_id=1)
    
    spawn_datadeps(scheduler=:locality_aware) do
        Dagger.with_options(;scope) do
            for idx in eachindex(B.chunks)
                @spawn apply_gpu_fft!(B.chunks[idx], In(transforms[2]), In(dims[2]))
            end
        end
    end
    
    coalesced_redistribute!(C, B, workspace_BC; phase_id=2)
    
    spawn_datadeps(scheduler=:locality_aware) do
        Dagger.with_options(;scope) do
            for idx in eachindex(C.chunks)
                @spawn apply_gpu_fft!(C.chunks[idx], In(transforms[3]), In(dims[3]))
            end
        end
    end
    
    return C
end

function fft!(B::DArray{T,3}, A::DArray{T,3},
              workspace_AB::CoalescedWorkspace{T},
              scope, transforms, dims, ::Slab) where T
    
    spawn_datadeps(scheduler=:locality_aware) do
        Dagger.with_options(;scope) do
            for idx in eachindex(A.chunks)
                @spawn apply_gpu_fft!(A.chunks[idx], In(transforms[1]), In((dims[1], dims[2])))
            end
        end
    end
    
    coalesced_redistribute!(B, A, workspace_AB; phase_id=1)
    
    spawn_datadeps(scheduler=:locality_aware) do
        Dagger.with_options(;scope) do
            for idx in eachindex(B.chunks)
                @spawn apply_gpu_fft!(B.chunks[idx], In(transforms[3]), In(dims[3]))
            end
        end
    end
    
    return B
end

function ifft!(C::DArray{T,3}, A::DArray{T,3}, B::DArray{T,3},
               workspace_AB::CoalescedWorkspace{T}, workspace_BC::CoalescedWorkspace{T},
               scope, transforms, dims, ::Pencil) where T
    
    spawn_datadeps(scheduler=:locality_aware) do
        Dagger.with_options(;scope) do
            for idx in eachindex(A.chunks)
                @spawn apply_gpu_fft!(A.chunks[idx], In(transforms[3]), In(dims[3]))
            end
        end
    end
    
    coalesced_redistribute!(B, A, workspace_AB; phase_id=3)
    
    spawn_datadeps(scheduler=:locality_aware) do
        Dagger.with_options(;scope) do
            for idx in eachindex(B.chunks)
                @spawn apply_gpu_fft!(B.chunks[idx], In(transforms[2]), In(dims[2]))
            end
        end
    end
    
    coalesced_redistribute!(C, B, workspace_BC; phase_id=4)
    
    spawn_datadeps(scheduler=:locality_aware) do
        Dagger.with_options(;scope) do
            for idx in eachindex(C.chunks)
                @spawn apply_gpu_fft!(C.chunks[idx], In(transforms[1]), In(dims[1]))
            end
        end
    end
    
    return C
end

function ifft!(A::DArray{T,3}, B::DArray{T,3},
               workspace_BA::CoalescedWorkspace{T},
               scope, transforms, dims, ::Slab) where T
    
    spawn_datadeps(scheduler=:locality_aware) do
        Dagger.with_options(;scope) do
            for idx in eachindex(B.chunks)
                @spawn apply_gpu_fft!(B.chunks[idx], In(transforms[3]), In(dims[3]))
            end
        end
    end
    
    coalesced_redistribute!(A, B, workspace_BA; phase_id=3)
    
    spawn_datadeps(scheduler=:locality_aware) do
        Dagger.with_options(;scope) do
            for idx in eachindex(A.chunks)
                @spawn apply_gpu_fft!(A.chunks[idx], In(transforms[1]), In((dims[1], dims[2])))
            end
        end
    end
    
    return A
end

# High-level interface for Pencil
function fft(A::DArray{T,3}, B::DArray{T,3}, C::DArray{T,3}, scope, transforms, dims,
             ::Pencil=Pencil()) where T
    
    workspace_AB = create_gpu_workspace(A, B)
    workspace_BC = create_gpu_workspace(B, C)
    
    return fft!(C, A, B, workspace_AB, workspace_BC, scope, transforms, dims, Pencil())
end

function ifft(A::DArray{T,3}, B::DArray{T,3}, C::DArray{T,3}, scope, transforms, dims,
              ::Pencil=Pencil()) where T
    
    workspace_AB = create_gpu_workspace(A, B)
    workspace_BC = create_gpu_workspace(B, C)
    
    return ifft!(C, A, B, workspace_AB, workspace_BC, scope, transforms, dims, Pencil())
end

# High-level interface for Slab
function fft(A::DArray{T,3}, B::DArray{T,3}, scope, transforms, dims, ::Slab) where T
    workspace_AB = create_gpu_workspace(A, B)
    return fft!(B, A, workspace_AB, scope, transforms, dims, Slab())
end

function ifft(A::DArray{T,3}, B::DArray{T,3}, scope, transforms, dims, ::Slab) where T
    workspace_BA = create_gpu_workspace(B, A)
    return ifft!(A, B, workspace_BA, scope, transforms, dims, Slab())
end

function compute_overlap(src_domain, dst_domain)
    return ntuple(i -> intersect(src_domain.indexes[i], dst_domain.indexes[i]), 3)
end

function generate_tag(peer_rank::Int, phase_id::Int)
    return 1000 + phase_id * 97 + (peer_rank & 0x3FFF)
end

function check_cuda_aware_mpi()
    try
        test_array = CUDA.zeros(Float32, 10)
        return true
    catch
        return false
    end
end

function AbstractCollect(d::DArray{T,N}, backend=KernelAbstractions.GPU()) where {T,N}
    total_size = size(d)
    result = KernelAbstractions.zeros(backend, T, total_size...)

    for (idx, chunk) in enumerate(d.chunks)
        chunk_domain = d.subdomains[idx]
        fetched_chunk = fetch(chunk)

        if fetched_chunk === nothing
            @warn "Chunk $idx was not computed. Filling with zeros."
            fetched_chunk = KernelAbstractions.zeros(backend, T, map(length, chunk_domain.indexes)...)
        end

        indices = map(r -> r.start:r.stop, chunk_domain.indexes)
        result[indices...] .= fetched_chunk
    end

    return result
end

function Base.similar(x::DArray{T,N}) where {T,N}
    alloc(idx, sz) = CUDA.zeros(T, sz)
    thunks = [Dagger.@spawn alloc(i, size(x)) for (i, x) in enumerate(x.subdomains)]
    return DArray(T, x.domain, x.subdomains, thunks, x.partitioning, x.concat)
end

function cleanup_workspace!(workspace::CoalescedWorkspace)
    # Clear the chunk cache
    empty!(workspace.chunk_cache)
    
    CUDA.reclaim()
    
end

const GPUFFTWorkspace = CoalescedWorkspace

end