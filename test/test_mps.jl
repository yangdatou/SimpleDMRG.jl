using Printf

include("../src/SimpleDMRG.jl")
using .SimpleDMRG

nn = 100

im_model = IsingModel(1.0, 1.0)
im_mps   = get_randn_mps(im_model, nn, 5)
display(im_mps)
println("im_mps = ", im_mps)
display(im_mps')
println("im_mps' = ", im_mps')
@time println("im_mps'*im_mps = ", im_mps'*im_mps)

hm_model = HeisenbergModel(1.0, 1.0, 1.0)
hm_mps   = get_randn_mps(hm_model, nn, 5)
display(hm_mps)
println("hm_mps = ", hm_mps)
display(hm_mps')
println("hm_mps' = ", hm_mps')
@time println("hm_mps'*hm_mps = ", hm_mps'*hm_mps)
