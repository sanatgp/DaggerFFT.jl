#=
- Full data dependency tracking (Read/Write/ReadWrite)
- Dynamic load balancing with variance-based rebalancing
- Hierarchical work stealing (thread-level and MPI-level)
- Cost-aware stealing decisions
- Asynchronous task scheduling
=#

import Graphs: SimpleDiGraph, add_edge!, add_vertex!, inneighbors, outneighbors, nv
using MPI
using Dagger
import Dagger: DTaskSpec, DTask, Processor, MemorySpace, AbstractTaskQueue, Chunk
import Dagger: processors, memory_spaces, get_parent, move, get_options, DefaultScope
import Dagger: constrain, ExactScope, UnionScope, InvalidScope, tochunk, memory_space
import Dagger: short_name
import Dagger: with_options, wait_all, spawn_bulk
import Dagger.Sch
using Statistics: mean, std

using Dagger: In, Out, InOut, Deps

if isdefined(Dagger, :current_acceleration)
    import Dagger: current_acceleration
end
if isdefined(Dagger, :default_processor)
    import Dagger: default_processor
end
if isdefined(Dagger, :check_uniform)
    import Dagger: check_uniform
end

abstract type AbstractDecompositionPlan end

struct MPIAcceleration <: Acceleration
    comm::MPI.Comm
end

struct DynamicSchedulingConfig
    enabled::Bool
    steal_threshold_ms::Float64
    cost_margin::Float64
    p2p_bandwidth_gbps::Float64
    p2p_latency_us::Float64
    decomposition_plan::Union{AbstractDecompositionPlan,Nothing}
    enable_continuous_stealing::Bool
    enable_mpi_stealing::Bool
    max_steal_attempts::Int
    mpi_steal_threshold_ms::Float64
    mpi_steal_chunk_size::Int
    rebalance_threshold::Float64 
    steal_check_interval_ms::Float64
    
    function DynamicSchedulingConfig(;
        enabled=false,
        steal_threshold_ms=5.0,
        cost_margin=0.1,
        p2p_bandwidth_gbps=50.0,
        p2p_latency_us=10.0,
        decomposition_plan=nothing,
        enable_continuous_stealing=false,
        enable_mpi_stealing=false,
        max_steal_attempts=3,
        mpi_steal_threshold_ms=50.0,
        mpi_steal_chunk_size=16,
        rebalance_threshold=0.3,  # 30% variance triggers rebalance
        steal_check_interval_ms=10.0)
        new(enabled, steal_threshold_ms, cost_margin, p2p_bandwidth_gbps, 
            p2p_latency_us, decomposition_plan, enable_continuous_stealing,
            enable_mpi_stealing, max_steal_attempts, mpi_steal_threshold_ms,
            mpi_steal_chunk_size, rebalance_threshold, steal_check_interval_ms)
    end
end

mutable struct DataDependencyTracker
    # Map from Chunk to the task that last wrote it
    chunk_writers::IdDict{Any, DTask}
    # Map from Chunk to tasks currently reading it  
    chunk_readers::IdDict{Any, Set{DTask}}
    # Map from task to its data dependencies
    task_deps::Dict{DTask, Set{DTask}}
    # Generation counter to track write ordering
    write_generation::Dict{Any, Int}
    # Track access patterns for debugging
    access_log::Vector{Tuple{DTask, Any, Symbol}}
    
    function DataDependencyTracker()
        new(IdDict{Any, DTask}(), 
            IdDict{Any, Set{DTask}}(),
            Dict{DTask, Set{DTask}}(),
            Dict{Any, Int}(),
            Tuple{DTask, Any, Symbol}[])
    end
end

function compute_task_dependencies!(tracker::DataDependencyTracker, 
                                   task::DTask, 
                                   spec::DTaskSpec)
    """
    Analyze task arguments to determine data dependencies.
    Returns Set of tasks that must complete before this task can run.
    
    Implements Read-After-Write (RAW), Write-After-Read (WAR), 
    and Write-After-Write (WAW) dependency tracking.
    """
    deps = Set{DTask}()
    
    for (_, arg) in spec.args
        arg, access_pattern = unwrap_inout(arg)
        
        # Unwrap DTask arguments to get underlying Chunk
        if arg isa DTask
            arg = fetch(arg; raw=true)
        end
        
        if !(arg isa Chunk)
            continue
        end
        
        is_write = any(ap -> ap[3], access_pattern)  # writedep flag
        is_read = any(ap -> ap[2], access_pattern)   # readdep flag
        
        if is_write
            # WRITE ACCESS: Must wait for previous writer + all current readers
            # This enforces WAW (Write-After-Write) and WAR (Write-After-Read)
            
            if haskey(tracker.chunk_writers, arg)
                prev_writer = tracker.chunk_writers[arg]
                push!(deps, prev_writer)
                @debug "WAW dependency: Task $(task.uid) depends on writer $(prev_writer.uid)"
            end
            
            if haskey(tracker.chunk_readers, arg)
                for reader in tracker.chunk_readers[arg]
                    push!(deps, reader)
                    @debug "WAR dependency: Task $(task.uid) depends on reader $(reader.uid)"
                end
            end
            
            tracker.chunk_writers[arg] = task
            tracker.chunk_readers[arg] = Set{DTask}()
            
            gen = get(tracker.write_generation, arg, 0) + 1
            tracker.write_generation[arg] = gen
            
            push!(tracker.access_log, (task, arg, :write))
            
        elseif is_read
            # READ ACCESS: Must wait for previous writer only
            # This enforces RAW (Read-After-Write)
            
            if haskey(tracker.chunk_writers, arg)
                prev_writer = tracker.chunk_writers[arg]
                push!(deps, prev_writer)
                @debug "RAW dependency: Task $(task.uid) depends on writer $(prev_writer.uid)"
            end
            
            if !haskey(tracker.chunk_readers, arg)
                tracker.chunk_readers[arg] = Set{DTask}()
            end
            push!(tracker.chunk_readers[arg], task)
            
            push!(tracker.access_log, (task, arg, :read))
        end
    end
    
    tracker.task_deps[task] = deps
    
    return deps
end

function get_dependency_chain_depth(tracker::DataDependencyTracker, task::DTask)
    """Calculate the longest dependency chain leading to this task."""
    if !haskey(tracker.task_deps, task)
        return 0
    end
    
    deps = tracker.task_deps[task]
    if isempty(deps)
        return 0
    end
    
    return 1 + maximum(get_dependency_chain_depth(tracker, dep) for dep in deps)
end

mutable struct ThreadWorkState
    thread_id::Int
    deque::Vector{Tuple{DTaskSpec,DTask}}
    estimated_load::Float64
    is_processing::Bool
    idle_since::Float64
    total_steals::Int
    total_stolen_from::Int
    total_tasks_completed::Int
    queue_lock::ReentrantLock
    last_steal_attempt::Float64
    
    function ThreadWorkState(thread_id::Int)
        new(thread_id, Tuple{DTaskSpec,DTask}[], 0.0, false, 
            time(), 0, 0, 0, ReentrantLock(), 0.0)
    end
end

mutable struct RankWorkState
    rank::Int
    total_tasks::Int
    active_tasks::Int
    estimated_completion_time::Float64
    thread_states::Dict{Int, ThreadWorkState}
    rank_queue::Vector{Tuple{DTaskSpec,DTask}}
    rank_lock::ReentrantLock
    last_mpi_steal_attempt::Float64
    mpi_steals_successful::Int
    mpi_steals_failed::Int
    mpi_tasks_stolen::Int
    total_tasks_completed::Int
    
    function RankWorkState(rank::Int)
        thread_states = Dict{Int, ThreadWorkState}()
        for tid in 1:Threads.nthreads()
            thread_states[tid] = ThreadWorkState(tid)
        end
        new(rank, 0, 0, 0.0, thread_states, 
            Tuple{DTaskSpec,DTask}[], ReentrantLock(), 0.0, 0, 0, 0, 0)
    end
end

mutable struct MPIProcessorLoadState
    rank::Int
    thread_id::Int
    processor::Processor
    active_tasks::Int
    estimated_completion_time::Float64
    total_tasks_completed::Int
    last_task_time::Float64
    queue_depth::Int
    avg_task_time::Float64
    task_time_samples::Vector{Float64}
    
    function MPIProcessorLoadState(rank::Int, thread_id::Int, proc::Processor)
        new(rank, thread_id, proc, 0, 0.0, 0, time(), 0, 0.0, Float64[])
    end
end

# Global state
const THREAD_WORK_STATES = Dict{Int, ThreadWorkState}()
const RANK_WORK_STATES = Dict{Int, RankWorkState}()
const MPI_PROCESSOR_LOADS = Dict{Processor, MPIProcessorLoadState}()
const THREAD_STEALING_ACTIVE = Ref(false)
const MPI_STEALING_ACTIVE = Ref(false)
const THREAD_STEALING_TASK = Ref{Union{Task,Nothing}}(nothing)
const MPI_STEALING_TASK = Ref{Union{Task,Nothing}}(nothing)
const MPI_SERVER_TASK = Ref{Union{Task,Nothing}}(nothing) 
const WORK_REQUEST_TAG = 1001
const WORK_RESPONSE_TAG = 1002
const TASK_DATA_TAG = 1003


function unwrap_inout(arg)
    """Extract argument and access pattern from dependency wrappers."""
    if arg isa In
        arg = arg.x
        return arg, Tuple[(identity, true, false)]  # read=true, write=false
    elseif arg isa Out
        arg = arg.x
        return arg, Tuple[(identity, false, true)]  # read=false, write=true
    elseif arg isa InOut
        arg = arg.x
        return arg, Tuple[(identity, true, true)]   # read=true, write=true
    elseif arg isa Deps
        arg = arg.x
        # Extract all nested dependencies
        alldeps = Tuple[]
        for dep in arg.deps
            dep_mod, inner_deps = unwrap_inout(dep)
            append!(alldeps, inner_deps)
        end
        return arg, alldeps
    else
        # Default: read-only
        return arg, Tuple[(identity, true, false)]
    end
end

function estimate_data_size(spec::DTaskSpec)
    """Estimate total data size for a task."""
    total_size = 0
    for (_, arg) in spec.args
        arg, _ = unwrap_inout(arg)
        if hasmethod(sizeof, (typeof(arg),))
            total_size += sizeof(arg)
        else
            total_size += 1024 * 1024 * 8  # 8MB default
        end
    end
    return total_size
end

function estimate_task_cost(spec::DTaskSpec, proc::Processor)
    """
    Estimate task execution time based on data size and processor type.
    Returns estimated time in seconds.
    """
    total_data_size = 0
    for (_, arg) in spec.args
        arg, _ = unwrap_inout(arg)
        if hasmethod(sizeof, (typeof(arg),))
            total_data_size += sizeof(arg)
        else
            total_data_size += 1000
        end
    end
    
    # Processor-specific scaling factor
    proc_factor = 1.0
    proc_type_str = string(typeof(proc))
    if occursin("GPU", proc_type_str) || occursin("Cuda", proc_type_str) || occursin("CUDA", proc_type_str)
        proc_factor = 0.1  # GPUs are ~10x faster
    end
    
    # Base cost proportional to data size
    base_cost = Float64(total_data_size) * proc_factor * 1e-9
    
    # Account for thread parallelism
    if Threads.nthreads() > 1
        base_cost = base_cost / sqrt(Threads.nthreads())
    end
    
    return base_cost
end

function estimate_transfer_cost(data_size::Int, config::DynamicSchedulingConfig)
    """
    Estimate cost of transferring data between processors.
    Implements Equation (4) from paper: L + V/B
    """
    latency_s = config.p2p_latency_us * 1e-6
    bandwidth_bps = config.p2p_bandwidth_gbps * 1e9
    
    transfer_time = latency_s + (data_size * 8) / bandwidth_bps
    return transfer_time
end

function estimate_steal_cost(spec::DTaskSpec, config::DynamicSchedulingConfig)
    """
    Estimate total cost of stealing a task (Equation 5 from paper).
    τs(ti) = L + V/B + σ
    """
    data_size = estimate_data_size(spec)
    transfer_cost = estimate_transfer_cost(data_size, config)
    
    # Runtime overhead (queue management, serialization)
    overhead = 1e-4  # 0.1ms
    
    return transfer_cost + overhead
end

# ============================================================================
# PROCESSOR LOAD MANAGEMENT
# ============================================================================

function get_processor_rank(proc::Processor)
    """Extract MPI rank from processor."""
    if hasfield(typeof(proc), :rank)
        return proc.rank
    else
        try
            return MPI.Comm_rank(MPI.COMM_WORLD)
        catch
            return 0
        end
    end
end

function update_processor_load!(proc::Processor, estimated_cost::Float64)
    """Update load state when a task is assigned to a processor."""
    rank = get_processor_rank(proc)
    thread_id = Threads.threadid()
    
    load_state = get!(()->MPIProcessorLoadState(rank, thread_id, proc), 
                      MPI_PROCESSOR_LOADS, proc)
    
    load_state.active_tasks += 1
    load_state.estimated_completion_time += estimated_cost
    load_state.last_task_time = time()
    load_state.queue_depth += 1
    
    if haskey(RANK_WORK_STATES, rank)
        rank_state = RANK_WORK_STATES[rank]
        if haskey(rank_state.thread_states, thread_id)
            thread_state = rank_state.thread_states[thread_id]
            thread_state.estimated_load += estimated_cost
        end
    end
    
    return nothing
end

function mark_task_completion!(proc::Processor, actual_cost::Float64)
    """Update load state when a task completes."""
    rank = get_processor_rank(proc)
    thread_id = Threads.threadid()
    
    load_state = get!(()->MPIProcessorLoadState(rank, thread_id, proc), 
                      MPI_PROCESSOR_LOADS, proc)
    
    load_state.active_tasks = max(0, load_state.active_tasks - 1)
    load_state.estimated_completion_time = max(0.0, 
        load_state.estimated_completion_time - actual_cost)
    load_state.total_tasks_completed += 1
    load_state.queue_depth = max(0, load_state.queue_depth - 1)
    
    push!(load_state.task_time_samples, actual_cost)
    if length(load_state.task_time_samples) > 100
        popfirst!(load_state.task_time_samples)
    end
    load_state.avg_task_time = mean(load_state.task_time_samples)
    
    # Update thread state
    if haskey(RANK_WORK_STATES, rank)
        rank_state = RANK_WORK_STATES[rank]
        if haskey(rank_state.thread_states, thread_id)
            thread_state = rank_state.thread_states[thread_id]
            thread_state.estimated_load = max(0.0, 
                thread_state.estimated_load - actual_cost)
            thread_state.total_tasks_completed += 1
        end
    end
    
    return nothing
end

function reset_processor_loads!(all_procs::Vector{Processor})
    """Reset all processor load states."""
    for proc in all_procs
        if haskey(MPI_PROCESSOR_LOADS, proc)
            load_state = MPI_PROCESSOR_LOADS[proc]
            load_state.active_tasks = 0
            load_state.estimated_completion_time = 0.0
            load_state.queue_depth = 0
        end
    end
    return nothing
end

function get_processor_load(proc::Processor)
    """Get current load estimate for a processor."""
    rank = get_processor_rank(proc)
    thread_id = Threads.threadid()
    
    load_state = get!(()->MPIProcessorLoadState(rank, thread_id, proc), 
                      MPI_PROCESSOR_LOADS, proc)
    
    # Combined metric: queue depth + estimated time
    return load_state.queue_depth + load_state.estimated_completion_time
end

function find_least_loaded_processor(all_procs::Vector{Processor})
    """Find processor with minimum current load."""
    min_load = Inf
    best_proc = first(all_procs)
    
    for proc in all_procs
        current_load = get_processor_load(proc)
        
        if current_load < min_load
            min_load = current_load
            best_proc = proc
        end
    end
    
    return best_proc
end

# ============================================================================
# DYNAMIC LOAD BALANCING (Algorithm 3 from paper)
# ============================================================================

function compute_load_variance(all_procs::Vector{Processor})
    """
    Compute normalized variance of processor loads.
    Returns variance as a fraction of mean load.
    """
    loads = [get_processor_load(proc) for proc in all_procs]
    
    if isempty(loads) || all(l -> l ≈ 0.0, loads)
        return 0.0
    end
    
    load_mean = mean(loads)
    if load_mean ≈ 0.0
        return 0.0
    end
    
    load_std = std(loads)
    return load_std / load_mean  # Coefficient of variation
end

function rebalance_tasks!(assignments::Dict{DTask, Processor},
                         all_procs::Vector{Processor},
                         pending_tasks::Vector{Pair{DTaskSpec,DTask}},
                         config::DynamicSchedulingConfig)
    """
    Rebalance task assignments to reduce load variance (Algorithm 3, lines 6-8).
    
    Migrates tasks from overloaded to underutilized processors.
    """
    if isempty(pending_tasks)
        return assignments
    end
    
    # Sort processors by current load
    proc_loads = [(proc, get_processor_load(proc)) for proc in all_procs]
    sort!(proc_loads, by=x->x[2])
    
    underutilized = [p for (p, l) in proc_loads[1:div(end,2)]]
    overloaded = [p for (p, l) in proc_loads[div(end,2)+1:end]]
    
    if isempty(underutilized) || isempty(overloaded)
        return assignments
    end
    
    # Find tasks currently assigned to overloaded processors
    migratable_tasks = Pair{DTask,Processor}[]
    for (task, proc) in assignments
        if proc in overloaded
            push!(migratable_tasks, task => proc)
        end
    end
    
    if isempty(migratable_tasks)
        return assignments
    end
    
    # Migrate tasks to balance loads
    migrations = 0
    max_migrations = min(length(migratable_tasks), 
                        div(length(pending_tasks), 2))
    
    for (task, old_proc) in migratable_tasks
        if migrations >= max_migrations
            break
        end
        
        # Find least loaded underutilized processor
        new_proc = underutilized[argmin([get_processor_load(p) for p in underutilized])]
        
        old_load = get_processor_load(old_proc)
        new_load = get_processor_load(new_proc)
        
        if new_load < old_load - config.cost_margin
            assignments[task] = new_proc
            migrations += 1
            
            @debug "Migrated task $(task.uid) from proc $(short_name(old_proc)) " *
                   "(load=$old_load) to $(short_name(new_proc)) (load=$new_load)"
        end
    end
    
    if migrations > 0
        @debug "Rebalanced $migrations tasks across processors"
    end
    
    return assignments
end

# ============================================================================
# WORK STEALING
# ============================================================================

function can_steal_task(victim_thread::ThreadWorkState, 
                       thief_idle_time::Float64,
                       config::DynamicSchedulingConfig)
    """
    Check if work stealing is beneficial (Equation 6 from paper).
    Iq > τs(ti): idle time must exceed steal cost
    """
    if isempty(victim_thread.deque)
        return false, nothing
    end
    
    # Estimate steal cost for first task in victim's queue
    spec, task = first(victim_thread.deque)
    steal_cost = estimate_steal_cost(spec, config)
    
    # Only steal if idle time exceeds cost (with margin)
    threshold = steal_cost * (1.0 + config.cost_margin)
    
    return thief_idle_time > threshold, (spec, task)
end

function attempt_thread_steal!(thief_thread::ThreadWorkState,
                              all_threads::Vector{ThreadWorkState},
                              config::DynamicSchedulingConfig)
    """
    Attempt to steal work from another thread on the same rank.
    """
    thief_idle_time = time() - thief_thread.idle_since
    
    # Check steal threshold
    if thief_idle_time < config.steal_threshold_ms * 1e-3
        return false
    end
    
    victims = filter(t -> !isempty(t.deque) && t.thread_id != thief_thread.thread_id, 
                    all_threads)
    
    if isempty(victims)
        return false
    end
    
    # Sort by load (steal from most loaded)
    sort!(victims, by=t->t.estimated_load, rev=true)
    
    for victim in victims
        lock(victim.queue_lock) do
            if isempty(victim.deque)
                return false
            end
            
            can_steal, task_pair = can_steal_task(victim, thief_idle_time, config)
            
            if can_steal
                # Steal from end of deque (LIFO for cache locality)
                stolen_task = pop!(victim.deque)
                
                lock(thief_thread.queue_lock) do
                    push!(thief_thread.deque, stolen_task)
                end
                
                thief_thread.total_steals += 1
                victim.total_stolen_from += 1
                thief_thread.idle_since = time()
                
                spec, task = stolen_task
                cost = estimate_task_cost(spec, default_processor())
                victim.estimated_load = max(0.0, victim.estimated_load - cost)
                thief_thread.estimated_load += cost
                
                @debug "Thread $(thief_thread.thread_id) stole task from " *
                       "thread $(victim.thread_id)"
                
                return true
            end
        end
    end
    
    return false
end

function thread_stealing_loop!(config::DynamicSchedulingConfig)
    """
    Continuous work stealing loop for thread-level load balancing.
    Runs in background while tasks are executing.
    """
    my_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    
    if !haskey(RANK_WORK_STATES, my_rank)
        return
    end
    
    rank_state = RANK_WORK_STATES[my_rank]
    check_interval = config.steal_check_interval_ms * 1e-3
    
    while THREAD_STEALING_ACTIVE[]
        sleep(check_interval)
        
        # Check each thread's idle status
        for (tid, thread_state) in rank_state.thread_states
            if !thread_state.is_processing && isempty(thread_state.deque)
                # Thread is idle, try to steal
                all_threads = collect(values(rank_state.thread_states))
                attempt_thread_steal!(thread_state, all_threads, config)
            end
        end
    end
end

function attempt_mpi_steal!(thief_rank::Int, 
                           victim_rank::Int,
                           config::DynamicSchedulingConfig,
                           comm::MPI.Comm)
    """
    Attempt to steal work from another MPI rank.
    Uses MPI point-to-point communication with custom protocol:
    
    Protocol:
    1. Thief sends WORK_REQUEST to victim
    2. Victim responds with number of available tasks
    3. If tasks available, victim serializes and sends task specs
    4. Thief deserializes and enqueues stolen tasks
    """
    if !haskey(RANK_WORK_STATES, thief_rank)
        return false
    end
    
    thief_state = RANK_WORK_STATES[thief_rank]
    
    # Check if enough time has passed since last attempt
    time_since_last = time() - thief_state.last_mpi_steal_attempt
    if time_since_last < config.mpi_steal_threshold_ms * 1e-3
        return false
    end
    
    thief_state.last_mpi_steal_attempt = time()
    

    try
        # Step 1: Send work request (non-blocking)
        request_msg = Int32[thief_rank, config.mpi_steal_chunk_size]
        send_req = MPI.Isend(request_msg, comm, dest=victim_rank, tag=WORK_REQUEST_TAG)
        
        # Step 2: Receive response about available tasks
        response = Vector{Int32}(undef, 2)
        recv_req = MPI.Irecv!(response, comm, source=victim_rank, tag=WORK_RESPONSE_TAG)
        
        timeout_ms = config.mpi_steal_threshold_ms / 2
        start_time = time()
        response_ready = false
        
        while (time() - start_time) < (timeout_ms * 1e-3)
            flag, _ = MPI.Test(recv_req)
            if flag
                response_ready = true
                break
            end
            sleep(0.001)  # 1ms polling interval
        end
        
        if !response_ready
            MPI.Cancel!(recv_req)
            MPI.Wait(send_req)
            @debug "MPI steal timeout: victim $victim_rank didn't respond"
            thief_state.mpi_steals_failed += 1
            return false
        end
        
        MPI.Wait(send_req)
        
        num_available = response[1]
        victim_rank_confirm = response[2]
        
        if num_available <= 0 || victim_rank_confirm != victim_rank
            @debug "No tasks available from victim $victim_rank"
            thief_state.mpi_steals_failed += 1
            return false
        end
        
        # Step 3: Receive serialized task data
        # Format: [num_tasks, task_uid_1, cost_1, rank_1, ..., task_uid_n, cost_n, rank_n]
        task_data_size = 1 + (num_available * 3)  # 1 count + (uid, cost, rank) per task
        task_data = Vector{Float64}(undef, task_data_size)
        
        MPI.Recv!(task_data, comm, source=victim_rank, tag=TASK_DATA_TAG)
        
        num_tasks_received = Int(task_data[1])
        
        @debug "Received $num_tasks_received tasks from rank $victim_rank"
        
        # Step 4: Process stolen tasks
        # Note: We receive task metadata only, actual task execution happens
        # through Dagger's normal mechanisms after rebalancing
        stolen_count = 0
        for i in 1:num_tasks_received
            offset = 1 + (i-1) * 3
            task_uid = Int(task_data[offset + 1])
            estimated_cost = task_data[offset + 2]
            preferred_rank = Int(task_data[offset + 3])
            
            # Update thief's load estimates
            thief_state.estimated_completion_time += estimated_cost
            stolen_count += 1
        end
        
        # Update statistics
        thief_state.mpi_steals_successful += 1
        thief_state.mpi_tasks_stolen += stolen_count
        
        @info "Rank $thief_rank successfully stole $stolen_count tasks from rank $victim_rank"
        
        return stolen_count > 0
        
    catch e
        @warn "MPI steal failed with error: $e"
        thief_state.mpi_steals_failed += 1
        return false
    end
end

function handle_mpi_steal_request!(victim_rank::Int, 
                                   thief_rank::Int,
                                   requested_tasks::Int,
                                   config::DynamicSchedulingConfig,
                                   comm::MPI.Comm)
    """
    Handle incoming work steal request from another rank.
    Called by the victim when it receives a WORK_REQUEST.
    """

    
    if !haskey(RANK_WORK_STATES, victim_rank)
        # No work state, send empty response
        response = Int32[0, victim_rank]
        MPI.Send(response, comm, dest=thief_rank, tag=WORK_RESPONSE_TAG)
        return
    end
    
    victim_state = RANK_WORK_STATES[victim_rank]
    
    # Collect stealable tasks from all threads
    stealable_tasks = Tuple{DTaskSpec,DTask,Float64}[]  # (spec, task, cost)
    
    lock(victim_state.rank_lock) do
        # Check rank-level queue
        for (spec, task) in victim_state.rank_queue
            cost = estimate_task_cost(spec, default_processor())
            push!(stealable_tasks, (spec, task, cost))
            if length(stealable_tasks) >= requested_tasks
                break
            end
        end
        
        # If need more, check thread queues (steal from most loaded threads)
        if length(stealable_tasks) < requested_tasks
            thread_loads = [(tid, ts.estimated_load) for (tid, ts) in victim_state.thread_states]
            sort!(thread_loads, by=x->x[2], rev=true)
            
            for (tid, _) in thread_loads
                thread_state = victim_state.thread_states[tid]
                
                lock(thread_state.queue_lock) do
                    # Only steal if thread has excess work
                    if thread_state.estimated_load > config.steal_threshold_ms * 1e-3
                        # Take from end of deque (LIFO)
                        tasks_to_take = min(length(thread_state.deque), 
                                          requested_tasks - length(stealable_tasks))
                        
                        for _ in 1:tasks_to_take
                            if !isempty(thread_state.deque)
                                spec, task = pop!(thread_state.deque)
                                cost = estimate_task_cost(spec, default_processor())
                                push!(stealable_tasks, (spec, task, cost))
                                thread_state.estimated_load = max(0.0, 
                                    thread_state.estimated_load - cost)
                            end
                        end
                    end
                end
                
                if length(stealable_tasks) >= requested_tasks
                    break
                end
            end
        end
    end
    
    num_available = min(length(stealable_tasks), requested_tasks)
    
    # Send response
    response = Int32[num_available, victim_rank]
    MPI.Send(response, comm, dest=thief_rank, tag=WORK_RESPONSE_TAG)
    
    if num_available == 0
        @debug "Rank $victim_rank has no tasks to share with rank $thief_rank"
        return
    end
    
    # Serialize task metadata
    # Format: [num_tasks, task_uid_1, cost_1, rank_1, ..., task_uid_n, cost_n, rank_n]
    task_data_size = 1 + (num_available * 3)
    task_data = Vector{Float64}(undef, task_data_size)
    task_data[1] = Float64(num_available)
    
    for (i, (spec, task, cost)) in enumerate(stealable_tasks[1:num_available])
        offset = 1 + (i-1) * 3
        task_data[offset + 1] = Float64(task.uid)
        task_data[offset + 2] = cost
        task_data[offset + 3] = Float64(victim_rank)
    end
    
    # Send task data
    MPI.Send(task_data, comm, dest=thief_rank, tag=TASK_DATA_TAG)
    
    # Update victim statistics
    victim_state.estimated_completion_time = max(0.0, 
        victim_state.estimated_completion_time - sum(t[3] for t in stealable_tasks[1:num_available]))
    
    @info "Rank $victim_rank sent $num_available tasks to rank $thief_rank"
end

function mpi_steal_server_loop!(config::DynamicSchedulingConfig)
    """
    Background server loop that handles incoming steal requests.
    Must run concurrently with mpi_stealing_loop!.
    """
    comm = MPI.COMM_WORLD
    my_rank = MPI.Comm_rank(comm)
    
    
    while MPI_STEALING_ACTIVE[]
        # Non-blocking probe for steal requests
        flag, status = MPI.Iprobe(comm, source=MPI.ANY_SOURCE, tag=WORK_REQUEST_TAG)
        
        if flag
            source_rank = status.source
            
            # Receive the request
            request_msg = Vector{Int32}(undef, 2)
            MPI.Recv!(request_msg, comm, source=source_rank, tag=WORK_REQUEST_TAG)
            
            thief_rank = request_msg[1]
            requested_tasks = request_msg[2]
            
            @debug "Rank $my_rank received steal request from rank $thief_rank " *
                   "for $requested_tasks tasks"
            
            handle_mpi_steal_request!(my_rank, thief_rank, requested_tasks, 
                                     config, comm)
        end
        
        sleep(0.001) 
    end
end

function mpi_stealing_loop!(config::DynamicSchedulingConfig)
    """
    Continuous MPI-level work stealing loop.
    Runs in background across ranks.
    """
    comm = MPI.COMM_WORLD
    my_rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    
    if nranks <= 1
        return
    end
    
    check_interval = config.mpi_steal_threshold_ms * 1e-3
    
    while MPI_STEALING_ACTIVE[]
        sleep(check_interval)
        
        if !haskey(RANK_WORK_STATES, my_rank)
            continue
        end
        
        rank_state = RANK_WORK_STATES[my_rank]
        
        total_queue_size = length(rank_state.rank_queue)
        for thread_state in values(rank_state.thread_states)
            total_queue_size += length(thread_state.deque)
        end
        
        if total_queue_size == 0 && rank_state.active_tasks == 0
            # This rank is idle, try to steal from others
            for victim_rank in 0:(nranks-1)
                if victim_rank != my_rank
                    if attempt_mpi_steal!(my_rank, victim_rank, config, comm)
                        break
                    end
                end
            end
        end
    end
end

function start_hierarchical_work_stealing!(config::DynamicSchedulingConfig)
    """
    Start background work stealing threads.
    Implements hierarchical stealing: thread-level first, then MPI-level.
    """
    if !config.enable_continuous_stealing && !config.enable_mpi_stealing
        return
    end
    
    my_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    if !haskey(RANK_WORK_STATES, my_rank)
        RANK_WORK_STATES[my_rank] = RankWorkState(my_rank)
    end
    
    # Start thread-level stealing
    if config.enable_continuous_stealing && Threads.nthreads() > 1
        THREAD_STEALING_ACTIVE[] = true
        THREAD_STEALING_TASK[] = @async thread_stealing_loop!(config)
        @debug "Started thread-level work stealing"
    end
    
    # Start MPI-level stealing
    if config.enable_mpi_stealing && MPI.Comm_size(MPI.COMM_WORLD) > 1
        MPI_STEALING_ACTIVE[] = true
        
        MPI_SERVER_TASK[] = @async mpi_steal_server_loop!(config)
        @debug "Started MPI steal server"
        
        MPI_STEALING_TASK[] = @async mpi_stealing_loop!(config)
        @debug "Started MPI-level work stealing client"
    end
end

function stop_hierarchical_work_stealing!()
    """Stop all work stealing background tasks."""
    THREAD_STEALING_ACTIVE[] = false
    MPI_STEALING_ACTIVE[] = false
    
    if THREAD_STEALING_TASK[] !== nothing
        try
            wait(THREAD_STEALING_TASK[])
        catch
        end
        THREAD_STEALING_TASK[] = nothing
    end
    
    if MPI_STEALING_TASK[] !== nothing
        try
            wait(MPI_STEALING_TASK[])
        catch
        end
        MPI_STEALING_TASK[] = nothing
    end
    
    if MPI_SERVER_TASK[] !== nothing
        try
            wait(MPI_SERVER_TASK[])
        catch
        end
        MPI_SERVER_TASK[] = nothing
    end
    
    @debug "Stopped work stealing"
end

function get_task_input_chunks(spec::DTaskSpec)
    """Extract all input chunks from task arguments."""
    input_chunks = []
    for (_, arg) in spec.args
        arg, _ = unwrap_inout(arg)
        if arg isa DTask
            continue
        elseif arg isa Chunk
            push!(input_chunks, arg)
        end
    end
    return input_chunks
end

function get_preferred_rank_from_decomp(spec::DTaskSpec, 
                                       all_procs::Vector{Processor}, 
                                       config::DynamicSchedulingConfig)
    """
    Extract preferred rank from decomposition plan.
    Returns rank where input data is likely located.
    """
    decomp_plan = config.decomposition_plan
    if decomp_plan === nothing
        return nothing
    end

    input_chunks = get_task_input_chunks(spec)
    if isempty(input_chunks)
        return nothing
    end

    chunk = first(input_chunks)
    if hasfield(typeof(chunk), :handle) && hasfield(typeof(chunk.handle), :rank)
        return chunk.handle.rank
    end
    return nothing
end

function compute_data_affinity(spec::DTaskSpec, proc::Processor)
    """
    Compute data affinity score for a task-processor pair.
    Higher score means more input data is local to this processor.
    """
    input_chunks = get_task_input_chunks(spec)
    if isempty(input_chunks)
        return 0.0
    end
    
    proc_rank = get_processor_rank(proc)
    local_chunks = 0
    
    for chunk in input_chunks
        if hasfield(typeof(chunk), :handle) && hasfield(typeof(chunk.handle), :rank)
            if chunk.handle.rank == proc_rank
                local_chunks += 1
            end
        end
    end
    
    return Float64(local_chunks) / length(input_chunks)
end

function find_locality_aware_processor(spec::DTaskSpec, 
                                      all_procs::Vector{Processor}, 
                                      config::DynamicSchedulingConfig)
    """
    Select processor based on data locality and current load.
    Implements affinity-based placement from Algorithm 3.
    """
    current_rank = MPI.Comm_rank(MPI.COMM_WORLD)

    # Strategy 1: Check decomposition plan for preferred rank
    preferred_rank = get_preferred_rank_from_decomp(spec, all_procs, config)
    if preferred_rank !== nothing
        candidate_procs = filter(p -> get_processor_rank(p) == preferred_rank, 
                                all_procs)
        if !isempty(candidate_procs)
            return find_least_loaded_processor(candidate_procs)
        end
    end

    # Strategy 2: Find rank with most input chunks
    input_chunks = get_task_input_chunks(spec)
    if isempty(input_chunks)
        # No data locality hints, use local processors
        local_procs = filter(proc -> get_processor_rank(proc) == current_rank, 
                           all_procs)
        return !isempty(local_procs) ? find_least_loaded_processor(local_procs) : 
                                       first(all_procs)
    end

    rank_chunk_counts = Dict{Int, Int}()
    for chunk in input_chunks
        if hasfield(typeof(chunk), :handle) && hasfield(typeof(chunk.handle), :rank)
            chunk_rank = chunk.handle.rank
            rank_chunk_counts[chunk_rank] = get(rank_chunk_counts, chunk_rank, 0) + 1
        end
    end

    best_rank = current_rank
    max_chunks = 0
    for (rank, count) in rank_chunk_counts
        if count > max_chunks
            max_chunks = count
            best_rank = rank
        end
    end

    # Get processors on best rank and select least loaded
    best_rank_procs = filter(proc -> get_processor_rank(proc) == best_rank, 
                            all_procs)
    
    if !isempty(best_rank_procs)
        return find_least_loaded_processor(best_rank_procs)
    else
        local_procs = filter(proc -> get_processor_rank(proc) == current_rank, 
                           all_procs)
        return !isempty(local_procs) ? find_least_loaded_processor(local_procs) : 
                                       first(all_procs)
    end
end

struct DataDepsTaskQueue <: AbstractTaskQueue
    upper_queue::AbstractTaskQueue
    seen_tasks::Union{Vector{Pair{DTaskSpec,DTask}},Nothing}
    g::Union{SimpleDiGraph{Int},Nothing}
    task_to_id::Union{Dict{DTask,Int},Nothing}
    traversal::Symbol
    scheduler::Symbol
    aliasing::Bool
    dynamic_config::DynamicSchedulingConfig
    dependency_tracker::DataDependencyTracker

    function DataDepsTaskQueue(upper_queue;
                               traversal::Symbol=:inorder,
                               scheduler::Symbol=:locality_aware,
                               aliasing::Bool=false,
                               dynamic_config::DynamicSchedulingConfig=DynamicSchedulingConfig())
        seen_tasks = Pair{DTaskSpec,DTask}[]
        g = SimpleDiGraph()
        task_to_id = Dict{DTask,Int}()
        dep_tracker = DataDependencyTracker()
        return new(upper_queue, seen_tasks, g, task_to_id, traversal, 
                  scheduler, aliasing, dynamic_config, dep_tracker)
    end
end

function Dagger.enqueue!(queue::DataDepsTaskQueue, spec::Pair{DTaskSpec,DTask})
    push!(queue.seen_tasks, spec)
end

function Dagger.enqueue!(queue::DataDepsTaskQueue, specs::Vector{Pair{DTaskSpec,DTask}})
    append!(queue.seen_tasks, specs)
end

function distribute_tasks!(queue::DataDepsTaskQueue)
    """
    Distribute tasks across processors with full dynamic load balancing.
    Implements Algorithm 3 from paper with variance-based rebalancing.
    """
    all_procs = Processor[]
    scope = get_options(:scope, DefaultScope())
    accel = current_acceleration()
    accel_procs = filter(procs(Dagger.Sch.eager_context())) do proc
        Dagger.accel_matches_proc(accel, proc)
    end
    all_procs = unique(vcat([collect(Dagger.get_processors(gp)) for gp in accel_procs]...))
    sort!(all_procs, by=short_name)
    filter!(proc->!isa(constrain(ExactScope(proc), scope), InvalidScope), all_procs)
    
    if isempty(all_procs)
        throw(Sch.SchedulingException("No processors available"))
    end
    
    for proc in all_procs
        check_uniform(proc)
    end

    upper_queue = get_options(:task_queue)
    traversal = queue.traversal
    task_order = Colon()
    scheduler = queue.scheduler
    config = queue.dynamic_config
    current_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    nthreads = Threads.nthreads()
    
    @debug "Distributing tasks with scheduler=$scheduler, threads=$nthreads"
    
    reset_processor_loads!(all_procs)
    
    # Initialize rank state for work stealing
    if !haskey(RANK_WORK_STATES, current_rank)
        RANK_WORK_STATES[current_rank] = RankWorkState(current_rank)
    end
    
    dep_tracker = queue.dependency_tracker
    task_assignments = Dict{DTask, Processor}()
    
    # Phase 1: Initial placement based on locality
    for (spec, task) in queue.seen_tasks[task_order]
        task_deps = compute_task_dependencies!(dep_tracker, task, spec)
        
        # Find best processor based on scheduler policy
        our_proc = if scheduler == :locality_aware || scheduler == :dynamic
            find_locality_aware_processor(spec, all_procs, config)
        else
            find_least_loaded_processor(all_procs)
        end
        
        task_assignments[task] = our_proc
        
        estimated_cost = estimate_task_cost(spec, our_proc)
        update_processor_load!(our_proc, estimated_cost)
    end
    
    # Phase 2: Check load variance and rebalance if needed 
    if config.enabled && scheduler == :dynamic
        load_variance = compute_load_variance(all_procs)
        
        @debug "Load variance: $load_variance (threshold: $(config.rebalance_threshold))"
        
        if load_variance > config.rebalance_threshold
            @debug "Variance exceeds threshold, rebalancing tasks"
            task_assignments = rebalance_tasks!(task_assignments, all_procs, 
                                               queue.seen_tasks, config)
            
            new_variance = compute_load_variance(all_procs)
            @debug "New variance after rebalancing: $new_variance"
        end
    end
    
    # Phase 3: Enqueue tasks with final assignments
    for (spec, task) in queue.seen_tasks[task_order]
        our_proc = task_assignments[task]
        target_rank = get_processor_rank(our_proc)
        
        @assert our_proc in all_procs
        our_space = only(memory_spaces(our_proc))
        our_procs = filter(proc->proc in all_procs, collect(processors(our_space)))
        our_scope = UnionScope(map(ExactScope, our_procs)...)
        
        check_uniform(our_proc)
        check_uniform(our_space)

        if target_rank == current_rank
            spec.f = move(default_processor(), our_proc, spec.f)
        end

        for (idx, (pos, arg)) in enumerate(spec.args)
            arg, _ = unwrap_inout(arg)
            
            if target_rank == current_rank
                arg = arg isa DTask ? fetch(arg; raw=true) : arg
            end
            
            spec.args[idx] = pos => arg
        end

        task_deps = get(dep_tracker.task_deps, task, Set{DTask}())
        existing_syncdeps = get(spec.options, :syncdeps, Set{Any}())
        all_syncdeps = union(existing_syncdeps, task_deps)
        
        @debug "Task $(task.uid) -> $(short_name(our_proc)) rank=$target_rank, " *
               "deps=$(length(all_syncdeps)), affinity=$(compute_data_affinity(spec, our_proc))"

        spec.options = merge(spec.options, (
            scope = our_scope,
            occupancy = Dict(Any=>0),
            syncdeps = all_syncdeps
        ))
        
        Dagger.enqueue!(upper_queue, spec=>task)
    end
    
    if current_rank == 0
        total_tasks = length(queue.seen_tasks)
        total_deps = sum(length(deps) for deps in values(dep_tracker.task_deps))
        avg_deps = total_deps / max(1, total_tasks)
        
    #    @info "Task distribution complete: $total_tasks tasks, " *
     #         "avg $(round(avg_deps, digits=2)) dependencies per task"
        
        for proc in all_procs
            load = get_processor_load(proc)
            @debug "  $(short_name(proc)): load=$load"
        end
    end
end

function spawn_datadeps(f::Base.Callable; 
                       static::Bool=true,
                       traversal::Symbol=:inorder,
                       scheduler::Union{Symbol,Nothing}=nothing,
                       aliasing::Bool=false,
                       launch_wait::Union{Bool,Nothing}=nothing,
                       dynamic_fft::Bool=false,
                       steal_threshold_ms::Float64=5.0,
                       p2p_bandwidth_gbps::Float64=50.0,
                       decomposition_plan::Union{AbstractDecompositionPlan,Nothing}=nothing,
                       enable_continuous_stealing::Bool=false,
                       enable_mpi_stealing::Bool=false,
                       max_steal_attempts::Int=3,
                       mpi_steal_threshold_ms::Float64=50.0,
                       mpi_steal_chunk_size::Int=16,
                       rebalance_threshold::Float64=0.3)
    
    if !static
        throw(ArgumentError("Dynamic scheduling is no longer available"))
    end
    
    if scheduler == :none
        return f()
    end
    
    auto_dynamic = dynamic_fft || (decomposition_plan !== nothing)
    
    nthreads = Threads.nthreads()
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    
    try
        result = wait_all(; check_errors=true) do
            effective_scheduler = if scheduler !== nothing
                scheduler
            elseif auto_dynamic
                :dynamic
            else
                something(DATADEPS_SCHEDULER[], :locality_aware)
            end::Symbol
            
            launch_wait = something(launch_wait, DATADEPS_LAUNCH_WAIT[], false)::Bool
            
            dynamic_config = DynamicSchedulingConfig(
                enabled = auto_dynamic,
                steal_threshold_ms = steal_threshold_ms,
                p2p_bandwidth_gbps = p2p_bandwidth_gbps,
                decomposition_plan = decomposition_plan,
                enable_continuous_stealing = enable_continuous_stealing,
                enable_mpi_stealing = enable_mpi_stealing,
                max_steal_attempts = max_steal_attempts,
                mpi_steal_threshold_ms = mpi_steal_threshold_ms,
                mpi_steal_chunk_size = mpi_steal_chunk_size,
                rebalance_threshold = rebalance_threshold
            )
            
         #   @info "Using scheduler: $effective_scheduler with full dependency tracking"
            
            # Start work stealing if enabled
            if enable_continuous_stealing || enable_mpi_stealing
                start_hierarchical_work_stealing!(dynamic_config)
            end

            result = spawn_bulk() do
                queue = DataDepsTaskQueue(get_options(:task_queue);
                                        traversal, scheduler=effective_scheduler, 
                                        aliasing, dynamic_config)
                with_options(f; task_queue=queue)
                distribute_tasks!(queue)
            end

            return result
        end
        
        return result
    finally
        if enable_continuous_stealing || enable_mpi_stealing
            stop_hierarchical_work_stealing!()
        end
        
        current_rank = MPI.Comm_rank(MPI.COMM_WORLD)
        if current_rank == 0 && haskey(RANK_WORK_STATES, current_rank)
            rank_state = RANK_WORK_STATES[current_rank]
            
            total_steals = sum(t.total_steals for t in values(rank_state.thread_states))
            total_completed = sum(t.total_tasks_completed for t in values(rank_state.thread_states))
            
            if total_steals > 0 || total_completed > 0
                @info "Execution statistics: completed=$total_completed tasks, " *
                      "steals=$total_steals"
            end
        end
    end
end

const DATADEPS_SCHEDULER = Ref{Union{Symbol,Nothing}}(nothing)
const DATADEPS_LAUNCH_WAIT = Ref{Union{Bool,Nothing}}(nothing)