using Printf

include("../src/SimpleDMRG.jl")
using .SimpleDMRG

im_model = IsingModel(1.0, 1.0)
im_mps   = get_randn_mps(im_model, 4, 5)
display(im_mps)
println("im_mps = ", im_mps)
display(im_mps')
println("im_mps' = ", im_mps')
println("im_mps'*im_mps = ", im_mps'*im_mps)
for i in 1:4
    println("im_mps[$i]  = ", size(im_mps[i]))
    println("im_mps'[$i] = ", size(im_mps'[i]))
end

hm_model = HeisenbergModel(1.0, 1.0, 1.0)
hm_mps   = get_randn_mps(hm_model, 4, 5)
display(hm_mps)
println("hm_mps = ", hm_mps)
display(hm_mps')
println("hm_mps' = ", hm_mps')
println("hm_mps'*hm_mps = ", hm_mps'*hm_mps)
for i in 1:4
    println("hm_mps[$i]  = ", size(hm_mps[i]))
    println("hm_mps'[$i] = ", size(hm_mps'[i]))
end
