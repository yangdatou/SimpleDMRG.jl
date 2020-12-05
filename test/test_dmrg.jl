using Printf

include("../src/SimpleDMRG.jl")
using .SimpleDMRG

nn = 100
im_model = IsingModel(1.0, 1.0, T=ComplexF64)
psi, e   = kernel(im_model, 20, verbose=5)