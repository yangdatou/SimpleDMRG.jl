using Printf

include("../src/SimpleDMRG.jl")
using .SimpleDMRG

im_model = IsingModel(1.0, 1.0)
im_mps   = get_randn_mps(im_model, 4, 5)
im_mpo   = get_mpo(im_model, 4)

display(im_mpo)
println("im_mps = ", im_mpo)
println("im_mps'*im_mpo*im_mps   = ", im_mps'*im_mpo*im_mps)
println("im_mps'*(im_mpo*im_mps) = ", im_mps'*(im_mpo*im_mps))
println("im_mps'*((im_mpo*im_mpo)*im_mps) = ", ((im_mps'*im_mpo)*im_mpo)*im_mps)
println("im_mps'*((im_mpo*im_mpo)*im_mps) = ", ((im_mps'*im_mpo)*im_mpo)*im_mps)
println("im_mps'*((im_mpo*im_mpo)*im_mps) = ", (im_mps'*im_mpo)*(im_mpo*im_mps))
println("im_mps'*((im_mpo*im_mpo)*im_mps) = ", im_mps'*((im_mpo*im_mpo)*im_mps))
println("im_mps'*(im_mpo*(im_mpo*im_mps)) = ", im_mps'*(im_mpo*(im_mpo*im_mps)))

hm_model = HeisenbergModel(1.0, 1.0, 1.0)
hm_mps   = get_randn_mps(hm_model, 10, 5)
hm_mpo   = get_mpo(hm_model, 10)

display(hm_mpo)
println("hm_mps = ", hm_mpo)
println("hm_mps'*hm_mpo*hm_mps   = ", hm_mps'*hm_mpo*hm_mps)
println("hm_mps'*(hm_mpo*hm_mps) = ", hm_mps'*(hm_mpo*hm_mps))
println("hm_mps'*((hm_mpo*hm_mpo)*hm_mps) = ", ((hm_mps'*hm_mpo)*hm_mpo)*hm_mps)
println("hm_mps'*((hm_mpo*hm_mpo)*hm_mps) = ", ((hm_mps'*hm_mpo)*hm_mpo)*hm_mps)
println("hm_mps'*((hm_mpo*hm_mpo)*hm_mps) = ", (hm_mps'*hm_mpo)*(hm_mpo*hm_mps))
println("hm_mps'*((hm_mpo*hm_mpo)*hm_mps) = ", hm_mps'*((hm_mpo*hm_mpo)*hm_mps))
println("hm_mps'*(hm_mpo*(hm_mpo*hm_mps)) = ", hm_mps'*(hm_mpo*(hm_mpo*hm_mps)))