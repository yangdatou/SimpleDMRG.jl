using Printf

include("../src/SimpleDMRG.jl")
using .SimpleDMRG

nn = 100

im_model = IsingModel(1.0, 1.0, T=ComplexF64)
im_mps   = get_randn_mps(im_model, nn)
display(im_mps)
println("im_mps = ", im_mps)
display(im_mps')
println("im_mps' = ", im_mps')
@time println("im_mps'* im_mps = ", im_mps'*im_mps)

hm_model = HeisenbergModel(1.0, 1.0, 1.0, T=ComplexF64)
hm_mps   = get_randn_mps(hm_model, nn)
display(hm_mps)
println("hm_mps = ", hm_mps)
display(hm_mps')
println("hm_mps' = ", hm_mps')
@time println("hm_mps'* hm_mps = ", hm_mps'*hm_mps)
