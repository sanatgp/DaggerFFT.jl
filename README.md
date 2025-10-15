# DaggerFFT.jl
Scalable distributed FFT implementation for heterogeneous CPU/GPU systems, built on Dagger.jl

[pencil1_cropped (2).pdf](https://github.com/user-attachments/files/22937554/pencil1_cropped.2.pdf)

## Installation

```julia
using Pkg
Pkg.add("DaggerFFT")
```

Or from the Julia REPL:

```julia
] add DaggerFFT
```

## Usage

### 3D Complex-to-Complex FFT & IFFT (CPU)

```julia
using DaggerFFT

A = rand(ComplexF64, 128, 128, 128)
F = fft(A; decomp=Pencil(), dims=(1,2,3))
A_recon = ifft(F; decomp=Pencil(), dims=(1,2,3))
```

### 3D Real-to-Complex RFFT & IRFFT (CPU)

```julia
using DaggerFFT

A = rand(256, 256, 256)
F = rfft(A; decomp=Pencil(), dims=(1,2,3))
A_recon = irfft(F, size(A, 1); decomp=Pencil(), dims=(1,2,3))
```

### 3D Complex FFT & IFFT (GPU - CUDA)

```julia
using DaggerFFT
using CUDA

A = CUDA.rand(ComplexF64, 256, 256, 256)
F = fft(A; decomp=Slab(), dims=(1,2,3))
A_recon = ifft(F; decomp=Slab(), dims=(1,2,3))
```

### 2D Real-to-Real FFT (CPU)

```julia
using DaggerFFT
using FFTW

A = rand(256, 256)
F = fft(A; decomp=Slab(), transforms=(R2R((FFTW.REDFT10, FFTW.REDFT10)),), dims=(1,2))
A_recon = ifft(F; decomp=Slab(), transforms=(R2R((FFTW.REDFT01, FFTW.REDFT01)),), dims=(1,2))
```

