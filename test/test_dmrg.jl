using Printf

include("../src/SimpleDMRG.jl")
using .SimpleDMRG

nn = 100
im_model = IsingModel(1.0, 1.0)