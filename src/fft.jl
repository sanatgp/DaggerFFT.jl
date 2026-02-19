const R2R_SUPPORTED_KINDS = (
    FFTW.DHT,
    FFTW.REDFT00,
    FFTW.REDFT01,
    FFTW.REDFT10,
    FFTW.REDFT11,
    FFTW.RODFT00,
    FFTW.RODFT01,
    FFTW.RODFT10,
    FFTW.RODFT11,
)

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

struct HierarchicalInfo
    node_rank::Int
    local_rank::Int
    ranks_per_node::Int
    total_nodes::Int
    is_aggregator::Bool
    
    node_comm::MPI.Comm
    inter_comm::MPI.Comm
end

function detect_hierarchical_topology(; ranks_per_node::Int=0)::HierarchicalInfo
    comm = MPI.COMM_WORLD
    rank = Comm_rank(comm)
    nranks = Comm_size(comm)
    
    if ranks_per_node == 0
        node_comm_temp = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)
        ranks_per_node = Comm_size(node_comm_temp)
        MPI.free(node_comm_temp)
        ranks_per_node = MPI.Allreduce(ranks_per_node, MPI.MIN, comm)
    end
    
    node_rank = rank ÷ ranks_per_node
    local_rank = rank % ranks_per_node
    total_nodes = cld(nranks, ranks_per_node)
    is_aggregator = (local_rank == 0)
    
    node_comm = MPI.Comm_split(comm, node_rank, local_rank)
    
    inter_comm = if is_aggregator
        MPI.Comm_split(comm, 1, node_rank)
    else
        MPI.Comm_split(comm, nothing, 0)
    end
    
    return HierarchicalInfo(node_rank, local_rank, ranks_per_node, total_nodes,
                           is_aggregator, node_comm, inter_comm)
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
    send_buffer::Union{Vector{T}, Nothing}
    recv_buffer::Union{Vector{T}, Nothing}
end

mutable struct FFTWorkspace{T}
    send_peers::Dict{Int, PeerCommInfo{T}}
    recv_peers::Dict{Int, PeerCommInfo{T}}
    local_patterns::Vector{CoalescedPattern}
    
    hierarchical::Union{HierarchicalInfo, Nothing}
    node_send_buffers::Dict{Int, Vector{T}}
    node_recv_buffers::Dict{Int, Vector{T}}
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
    use_hierarchical::Bool
    
    chunk_cache::Dict{Int, Any}
end

const PLAN_CACHE = Dict{Tuple{DataType, NTuple, Type, Tuple}, Any}()
const WISDOM_FILE = "fftw_wisdom.dat"

function __init__()
    if isfile(WISDOM_FILE)
        FFTW.import_wisdom(WISDOM_FILE)
    end
end

function save_wisdom()
    FFTW.export_wisdom(WISDOM_FILE)
end

function plan_transform(transform, A, dims; kwargs...)
    flags = get(kwargs, :flags, FFTW.MEASURE)
    
    if transform isa FFT
        return FFTW.plan_fft(A, dims; flags=flags, kwargs...)
    elseif transform isa IFFT
        return FFTW.plan_ifft(A, dims; flags=flags, kwargs...)
    elseif transform isa RFFT!
        return FFTW.plan_rfft!(A, dims; flags=flags, kwargs...)
    elseif transform isa FFT!
        return FFTW.plan_fft!(A, dims; flags=flags, kwargs...)
    elseif transform isa IRFFT!
        return FFTW.plan_irfft!(A, size(A, first(dims)), dims; flags=flags, kwargs...)
    elseif transform isa IFFT!
        return FFTW.plan_ifft!(A, dims; flags=flags, kwargs...)
    else
        throw(ArgumentError("Unknown transform type"))
    end
end

function plan_transform(transform, A, dims, n; kwargs...)
    if transform isa RFFT
        return plan_rfft(A, dims; kwargs...)
    elseif transform isa IRFFT
        return plan_irfft(A, n, dims; kwargs...)
    else
        throw(ArgumentError("Unknown transform type"))
    end
end

function get_or_create_plan(transform, A, dims; kwargs...)
    dims_tuple = dims isa Tuple ? dims : (dims,)
    key = (eltype(A), size(A), typeof(transform), dims_tuple)
    
    if !haskey(PLAN_CACHE, key)
        PLAN_CACHE[key] = plan_transform(transform, A, dims; kwargs...)
        save_wisdom()
    end
    
    return PLAN_CACHE[key]
end

function apply_fft!(out_part, in_part, transform, dim)
    plan = get_or_create_plan(transform, in_part, dim)
    mul!(out_part, plan, in_part)
    return nothing
end

apply_fft!(inout_part, transform, dim) = apply_fft!(inout_part, inout_part, transform, dim)

function compute_overlap(src_domain, dst_domain)
    return ntuple(i -> intersect(src_domain.indexes[i], dst_domain.indexes[i]), 3)
end

function pack_data!(buffer::Vector{T}, src::Array{T,3}, indices::NTuple{3, UnitRange{Int}}, offset::Int=0) where T
    nx, ny, nz = length.(indices)
    xs, ys, zs = first.(indices)
    
    @inbounds for k in 1:nz
        kk = zs + k - 1
        for j in 1:ny
            jj = ys + j - 1
            base_idx = offset + (k-1)*nx*ny + (j-1)*nx
            for i in 1:nx
                ii = xs + i - 1
                buffer[base_idx + i] = src[ii, jj, kk]
            end
        end
    end
    return nothing
end

function unpack_data!(dst::Array{T,3}, buffer::Vector{T}, indices::NTuple{3, UnitRange{Int}}, offset::Int=0) where T
    nx, ny, nz = length.(indices)
    xs, ys, zs = first.(indices)
    
    @inbounds for k in 1:nz
        kk = zs + k - 1
        for j in 1:ny
            jj = ys + j - 1
            base_idx = offset + (k-1)*nx*ny + (j-1)*nx
            for i in 1:nx
                ii = xs + i - 1
                dst[ii, jj, kk] = buffer[base_idx + i]
            end
        end
    end
    return nothing
end

function local_copy!(dst::Array{T,3}, src::Array{T,3}, 
                    dst_indices::NTuple{3, UnitRange{Int}}, 
                    src_indices::NTuple{3, UnitRange{Int}}) where T
    nx, ny, nz = length.(dst_indices)
    dst_xs, dst_ys, dst_zs = first.(dst_indices)
    src_xs, src_ys, src_zs = first.(src_indices)
    
    @inbounds for k in 1:nz
        for j in 1:ny
            for i in 1:nx
                dst[dst_xs+i-1, dst_ys+j-1, dst_zs+k-1] = src[src_xs+i-1, src_ys+j-1, src_zs+k-1]
            end
        end
    end
    return nothing
end

function create_workspace(src::DArray{T,3}, dst::DArray{T,3}; 
                         use_hierarchical::Bool=true,
                         ranks_per_node::Int=0) where T
    comm = MPI.COMM_WORLD
    rank = Comm_rank(comm)
    nranks = Comm_size(comm)
    
    hierarchical = if use_hierarchical && nranks > 1
        detect_hierarchical_topology(ranks_per_node=ranks_per_node)
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
        dst_node = hierarchical !== nothing ? dst_rank ÷ hierarchical.ranks_per_node : -1
        
        for (src_idx, src_chunk) in enumerate(src.chunks)
            src_rank = src_chunk.handle.rank
            src_domain = src.subdomains[src_idx]
            src_node = hierarchical !== nothing ? src_rank ÷ hierarchical.ranks_per_node : -1
            
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
        
        send_buffer = Vector{T}(undef, total_size)
        peer_info = PeerCommInfo{T}(peer_rank, total_size, patterns, send_buffer, nothing)
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
        
        recv_buffer = Vector{T}(undef, total_size)
        peer_info = PeerCommInfo{T}(peer_rank, total_size, patterns, nothing, recv_buffer)
        recv_peers[peer_rank] = peer_info
    end
    
    node_send_buffers = Dict{Int, Vector{T}}()
    node_recv_buffers = Dict{Int, Vector{T}}()
    intra_send_patterns = Dict{Int, Vector{CoalescedPattern}}()
    intra_recv_patterns = Dict{Int, Vector{CoalescedPattern}}()
    
    if use_hierarchical && hierarchical !== nothing
        for (target_node, pattern_list) in node_send_patterns
            if hierarchical.is_aggregator
                total_size = sum(p[6] for p in pattern_list)
                node_send_buffers[target_node] = Vector{T}(undef, total_size)
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
                node_recv_buffers[source_node] = Vector{T}(undef, total_size)
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
    
    workspace = FFTWorkspace{T}(
        send_peers, recv_peers, local_patterns, hierarchical,
        node_send_buffers, node_recv_buffers, intra_send_patterns, intra_recv_patterns,
        send_reqs, recv_reqs, node_send_reqs, node_recv_reqs,
        send_ranks, recv_ranks, target_nodes, source_nodes,
        size(src), size(dst), use_hierarchical, chunk_cache
    )
    
    return workspace
end

function coalesced_redistribute!(dst::DArray{T,3}, src::DArray{T,3}, workspace::FFTWorkspace{T}; phase_id::Int=1) where T
    flat_redistribute!(dst, src, workspace, phase_id)
end

function flat_redistribute!(dst::DArray{T,3}, src::DArray{T,3}, workspace::FFTWorkspace{T}, phase_id::Int) where T
    comm = MPI.COMM_WORLD
    rank = Comm_rank(comm)
    
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
    
    for (i, src_rank) in enumerate(workspace.recv_ranks)
        peer_info = workspace.recv_peers[src_rank]
        tag = generate_tag(src_rank, phase_id)
        workspace.recv_reqs[i] = MPI.Irecv!(peer_info.recv_buffer, src_rank, tag, comm)
    end
    
    for dst_rank in workspace.send_ranks
        peer_info = workspace.send_peers[dst_rank]
        for pattern in peer_info.patterns
            if haskey(workspace.chunk_cache, pattern.src_idx)
                src_data = workspace.chunk_cache[pattern.src_idx]
                pack_data!(peer_info.send_buffer, src_data, pattern.src_indices, pattern.buffer_offset)
            end
        end
    end
    
    for (i, dst_rank) in enumerate(workspace.send_ranks)
        tag = generate_tag(rank, phase_id)
        peer_info = workspace.send_peers[dst_rank]
        workspace.send_reqs[i] = MPI.Isend(peer_info.send_buffer, dst_rank, tag, comm)
    end
    
    for pattern in workspace.local_patterns
        src_key = pattern.src_idx
        dst_key = pattern.dst_idx + 1000
        
        if haskey(workspace.chunk_cache, src_key) && haskey(workspace.chunk_cache, dst_key)
            src_data = workspace.chunk_cache[src_key]
            dst_data = workspace.chunk_cache[dst_key]
            local_copy!(dst_data, src_data, pattern.dst_indices, pattern.src_indices)
        end
    end
    
    remaining = Set(1:length(workspace.recv_ranks))
    
    while !isempty(remaining)
        for i in copy(remaining)
            if i <= length(workspace.recv_reqs)
                flag, _ = MPI.Test!(workspace.recv_reqs[i])
                
                if flag
                    src_rank = workspace.recv_ranks[i]
                    peer_info = workspace.recv_peers[src_rank]
                    
                    for pattern in peer_info.patterns
                        dst_key = pattern.dst_idx + 1000
                        if haskey(workspace.chunk_cache, dst_key)
                            dst_data = workspace.chunk_cache[dst_key]
                            unpack_data!(dst_data, peer_info.recv_buffer, 
                                       pattern.dst_indices, pattern.buffer_offset)
                        end
                    end
                    
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

function hierarchical_redistribute!(dst::DArray{T,3}, src::DArray{T,3}, workspace::FFTWorkspace{T}, phase_id::Int) where T
    hierarchical = workspace.hierarchical
    comm = MPI.COMM_WORLD
    rank = Comm_rank(comm)
    
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
    
    for (target_node, patterns) in workspace.intra_send_patterns
        if haskey(workspace.node_send_buffers, target_node)
            node_buffer = workspace.node_send_buffers[target_node]
            
            for pattern in patterns
                if haskey(workspace.chunk_cache, pattern.src_idx)
                    src_data = workspace.chunk_cache[pattern.src_idx]
                    pack_data!(node_buffer, src_data, pattern.src_indices, pattern.buffer_offset)
                end
            end
        end
    end
    
    MPI.Barrier(hierarchical.node_comm)
    
    if hierarchical.is_aggregator
        for (i, source_node) in enumerate(workspace.source_nodes)
            if haskey(workspace.node_recv_buffers, source_node)
                node_buffer = workspace.node_recv_buffers[source_node]
                source_aggregator = source_node * hierarchical.ranks_per_node
                tag = generate_tag(source_aggregator, phase_id + 1000)
                workspace.node_recv_reqs[i] = MPI.Irecv!(node_buffer, source_aggregator, tag, comm)
            end
        end
        
        for (i, target_node) in enumerate(workspace.target_nodes)
            if haskey(workspace.node_send_buffers, target_node)
                node_buffer = workspace.node_send_buffers[target_node]
                target_aggregator = target_node * hierarchical.ranks_per_node
                tag = generate_tag(rank, phase_id + 1000)
                workspace.node_send_reqs[i] = MPI.Isend(node_buffer, target_aggregator, tag, comm)
            end
        end
        
        remaining_node = Set(1:length(workspace.source_nodes))
        while !isempty(remaining_node)
            for i in copy(remaining_node)
                flag, _ = MPI.Test!(workspace.node_recv_reqs[i])
                if flag
                    source_node = workspace.source_nodes[i]
                    if haskey(workspace.node_recv_buffers, source_node) && 
                       haskey(workspace.intra_recv_patterns, source_node)
                        node_buffer = workspace.node_recv_buffers[source_node]
                        
                        for pattern in workspace.intra_recv_patterns[source_node]
                            dst_key = pattern.dst_idx + 1000
                            if haskey(workspace.chunk_cache, dst_key)
                                dst_data = workspace.chunk_cache[dst_key]
                                unpack_data!(dst_data, node_buffer, pattern.dst_indices, pattern.buffer_offset)
                            end
                        end
                    end
                    delete!(remaining_node, i)
                end
            end
            if !isempty(remaining_node)
                yield()
            end
        end
        
        for i in 1:length(workspace.target_nodes)
            MPI.Wait!(workspace.node_send_reqs[i])
        end
    end
    
    MPI.Barrier(hierarchical.node_comm)
    
    for (i, src_rank) in enumerate(workspace.recv_ranks)
        peer_info = workspace.recv_peers[src_rank]
        tag = generate_tag(src_rank, phase_id)
        workspace.recv_reqs[i] = MPI.Irecv!(peer_info.recv_buffer, src_rank, tag, comm)
    end
    
    for dst_rank in workspace.send_ranks
        peer_info = workspace.send_peers[dst_rank]
        for pattern in peer_info.patterns
            if haskey(workspace.chunk_cache, pattern.src_idx)
                src_data = workspace.chunk_cache[pattern.src_idx]
                pack_data!(peer_info.send_buffer, src_data, pattern.src_indices, pattern.buffer_offset)
            end
        end
    end
    
    for (i, dst_rank) in enumerate(workspace.send_ranks)
        tag = generate_tag(rank, phase_id)
        peer_info = workspace.send_peers[dst_rank]
        workspace.send_reqs[i] = MPI.Isend(peer_info.send_buffer, dst_rank, tag, comm)
    end
    
    remaining = Set(1:length(workspace.recv_ranks))
    while !isempty(remaining)
        for i in copy(remaining)
            flag, _ = MPI.Test!(workspace.recv_reqs[i])
            if flag
                src_rank = workspace.recv_ranks[i]
                peer_info = workspace.recv_peers[src_rank]
                
                for pattern in peer_info.patterns
                    dst_key = pattern.dst_idx + 1000
                    if haskey(workspace.chunk_cache, dst_key)
                        dst_data = workspace.chunk_cache[dst_key]
                        unpack_data!(dst_data, peer_info.recv_buffer, 
                                   pattern.dst_indices, pattern.buffer_offset)
                    end
                end
                
                delete!(remaining, i)
            end
        end
        if !isempty(remaining)
            yield()
        end
    end
    
    for i in 1:length(workspace.send_ranks)
        MPI.Wait!(workspace.send_reqs[i])
    end
    
    for pattern in workspace.local_patterns
        src_key = pattern.src_idx
        dst_key = pattern.dst_idx + 1000
        
        if haskey(workspace.chunk_cache, src_key) && haskey(workspace.chunk_cache, dst_key)
            src_data = workspace.chunk_cache[src_key]
            dst_data = workspace.chunk_cache[dst_key]
            local_copy!(dst_data, src_data, pattern.dst_indices, pattern.src_indices)
        end
    end
end

function generate_tag(peer_rank::Int, phase_id::Int)
    return 1000 + phase_id * 97 + (peer_rank & 0x3FFF)
end

# Pencil decomposition FFT (3 stages, 2 redistributions)
function fft!(C::DArray{T,3}, A::DArray{T,3}, B::DArray{T,3},
              workspace_AB::FFTWorkspace{T}, workspace_BC::FFTWorkspace{T},
              transforms, dims) where T
    NVTX.@range "DIM 1" begin
        spawn_datadeps(scheduler=:dynamic,
                      enable_continuous_stealing=true,
                      steal_threshold_ms=3.0) do
            for idx in eachindex(A.chunks)
                Dagger.@spawn apply_fft!(A.chunks[idx], In(transforms[1]), In(dims[1]))
            end
        end
    end
    NVTX.@range "Redistribute 1" begin
        coalesced_redistribute!(B, A, workspace_AB; phase_id=1)
    end
    NVTX.@range "DIM 2" begin
        spawn_datadeps(scheduler=:dynamic,
                      enable_continuous_stealing=true,
                      steal_threshold_ms=3.0) do
            for idx in eachindex(B.chunks)
                Dagger.@spawn apply_fft!(B.chunks[idx], In(transforms[2]), In(dims[2]))
            end
        end
    end
    NVTX.@range "Redistribute 2" begin
        coalesced_redistribute!(C, B, workspace_BC; phase_id=2)
    end
    NVTX.@range "DIM 3" begin
        spawn_datadeps(scheduler=:dynamic,
                      enable_continuous_stealing=true,
                      steal_threshold_ms=3.0) do
            for idx in eachindex(C.chunks)
                Dagger.@spawn apply_fft!(C.chunks[idx], In(transforms[3]), In(dims[3]))
            end
        end
    end
    return C
end

# Pencil decomposition IFFT
function ifft!(C::DArray{T,3}, A::DArray{T,3}, B::DArray{T,3},
               workspace_AB::FFTWorkspace{T}, workspace_BC::FFTWorkspace{T},
               transforms, dims) where T
    
    spawn_datadeps(scheduler=:dynamic,
                  enable_continuous_stealing=true,
                  steal_threshold_ms=3.0) do
        for idx in eachindex(A.chunks)
            Dagger.@spawn apply_fft!(A.chunks[idx], In(transforms[3]), In(dims[3]))
        end
    end
    
    coalesced_redistribute!(B, A, workspace_AB; phase_id=3)
    
    spawn_datadeps(scheduler=:dynamic,
                  enable_continuous_stealing=true,
                  steal_threshold_ms=3.0) do
        for idx in eachindex(B.chunks)
            Dagger.@spawn apply_fft!(B.chunks[idx], In(transforms[2]), In(dims[2]))
        end
    end
    
    coalesced_redistribute!(C, B, workspace_BC; phase_id=4)
    
    spawn_datadeps(scheduler=:dynamic,
                  enable_continuous_stealing=true,
                  steal_threshold_ms=3.0) do
        for idx in eachindex(C.chunks)
            Dagger.@spawn apply_fft!(C.chunks[idx], In(transforms[1]), In(dims[1]))
        end
    end
    
    return C
end

# Slab decomposition FFT (2 stages, 1 redistribution)
function fft!(B::DArray{T,3}, A::DArray{T,3},
              workspace_AB::FFTWorkspace{T},
              transforms, dims, ::Slab) where T
    NVTX.@range "DIM 1" begin
        spawn_datadeps(scheduler=:dynamic,
                      enable_continuous_stealing=true,
                      steal_threshold_ms=3.0) do
            for idx in eachindex(A.chunks)
                Dagger.@spawn apply_fft!(A.chunks[idx], In(transforms[1]), In((dims[1], dims[2])))
            end
        end
    end
    NVTX.@range "Redistribute 1" begin
        coalesced_redistribute!(B, A, workspace_AB; phase_id=1)
    end
    spawn_datadeps(scheduler=:dynamic,
                  enable_continuous_stealing=true,
                  steal_threshold_ms=3.0) do
        for idx in eachindex(B.chunks)
            Dagger.@spawn apply_fft!(B.chunks[idx], In(transforms[3]), In(dims[3]))
        end
    end
    return B
end

# Slab decomposition IFFT
function ifft!(A::DArray{T,3}, B::DArray{T,3},
               workspace_BA::FFTWorkspace{T},
               transforms, dims, ::Slab) where T
    
    spawn_datadeps(scheduler=:dynamic,
                  enable_continuous_stealing=true,
                  steal_threshold_ms=3.0) do
        for idx in eachindex(B.chunks)
            Dagger.@spawn apply_fft!(B.chunks[idx], In(transforms[3]), In(dims[3]))
        end
    end
    
    coalesced_redistribute!(A, B, workspace_BA; phase_id=3)
    
    spawn_datadeps(scheduler=:dynamic,
                  enable_continuous_stealing=true,
                  steal_threshold_ms=3.0) do
        for idx in eachindex(A.chunks)
            Dagger.@spawn apply_fft!(A.chunks[idx], In(transforms[1]), In((dims[1], dims[2])))
        end
    end
    
    return A
end

function cleanup_workspace!(workspace::FFTWorkspace)
    empty!(workspace.chunk_cache)
    empty!(PLAN_CACHE)
end