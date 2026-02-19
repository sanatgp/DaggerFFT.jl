module DaggerFFT

using AbstractFFTs
using LinearAlgebra
using FFTW
using MPI
using LoopVectorization
using NVTX
using Graphs
using Statistics
using CUDA
using GPUArraysCore
using KernelAbstractions

using Dagger
import Dagger: DTaskSpec, DTask, Processor, MemorySpace, AbstractTaskQueue, Chunk
import Dagger: processors, memory_spaces, get_parent, move, get_options, DefaultScope
import Dagger: constrain, ExactScope, UnionScope, InvalidScope, tochunk, memory_space
import Dagger: short_name, Acceleration
import Dagger: with_options, wait_all, spawn_bulk
import Dagger: DArray, @spawn, In, Out, InOut, Deps
import Dagger.Sch
import MPI: Comm_rank, Comm_size
using Distributed: procs

import Dagger: current_acceleration, default_processor, check_uniform

include("datadeps.jl")
include("fft.jl")
include("fftgpu.jl")

export fft!, ifft!, gpu_fft!, gpu_ifft!, spawn_datadeps
export FFT, RFFT, IRFFT, IFFT, FFT!, RFFT!, IRFFT!, IFFT!
export Pencil, Slab
export create_workspace, cleanup_workspace!
export create_gpu_workspace, gpu_cleanup_workspace!
export CoalescedWorkspace, GPUFFTWorkspace, FFTWorkspace

end