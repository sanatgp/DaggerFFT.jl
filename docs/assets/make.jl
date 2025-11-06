#!/usr/bin/env julia
using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Documenter
using DaggerFFT 

DocMeta.setdocmeta!(DaggerFFT, :DocTestSetup, :(using DaggerFFT); recursive=true)

pages = [
    "Home" => "index.md",
    "Guide" => [
        "Quickstart" => "guide/quickstart.md",
        "Pipelining & GPU" => "guide/pipelining_gpu.md",
    ],
    "API" => "api.md",
    "Benchmarks" => "benchmarks.md",
]

makedocs(
    sitename   = "DaggerFFT.jl",
    modules    = [DaggerFFT],
    format     = Documenter.HTML(prettyurls = get(ENV, "CI", "false") == "true"),
    pages      = pages,
    strict     = true,
    assets     = ["assets"],  
)

deploydocs(
    repo       = "github.com/sanatgp/DaggerFFT.jl",
    devbranch  = "main",
)
