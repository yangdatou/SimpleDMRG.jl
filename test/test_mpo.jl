using Printf

include("../src/SimpleDMRG.jl")
using .SimpleDMRG

nn = 100

im_model = IsingModel(1.0, 1.0, T=ComplexF64)
im_mps   = get_randn_mps(im_model, nn)
im_mpo   = get_mpo(im_model,       nn)

display(im_mpo)
println("im_mps = ", im_mpo)
@time println("im_mps'*im_mpo*im_mps   = ", im_mps'*im_mpo*im_mps)
@time println("im_mps'*(im_mpo*im_mps) = ", im_mps'*(im_mpo*im_mps))
@time println("im_mps'*((im_mpo*im_mpo)*im_mps) = ", ((im_mps'*im_mpo)*im_mpo)*im_mps)
@time println("im_mps'*((im_mpo*im_mpo)*im_mps) = ", ((im_mps'*im_mpo)*im_mpo)*im_mps)
@time println("im_mps'*((im_mpo*im_mpo)*im_mps) = ", (im_mps'*im_mpo)*(im_mpo*im_mps))
@time println("im_mps'*((im_mpo*im_mpo)*im_mps) = ", im_mps'*((im_mpo*im_mpo)*im_mps))
@time println("im_mps'*(im_mpo*(im_mpo*im_mps)) = ", im_mps'*(im_mpo*(im_mpo*im_mps)))

hm_model = HeisenbergModel(1.0, 1.0, 1.0, T=ComplexF64)
hm_mps   = get_randn_mps(hm_model, nn)
hm_mpo   = get_mpo(hm_model,       nn)

display(hm_mpo)
println("hm_mps = ", hm_mpo)
@time println("hm_mps'*hm_mpo*hm_mps   = ", hm_mps'*hm_mpo*hm_mps)
@time println("hm_mps'*(hm_mpo*hm_mps) = ", hm_mps'*(hm_mpo*hm_mps))
@time println("hm_mps'*((hm_mpo*hm_mpo)*hm_mps) = ", ((hm_mps'*hm_mpo)*hm_mpo)*hm_mps)
@time println("hm_mps'*((hm_mpo*hm_mpo)*hm_mps) = ", ((hm_mps'*hm_mpo)*hm_mpo)*hm_mps)
@time println("hm_mps'*((hm_mpo*hm_mpo)*hm_mps) = ", (hm_mps'*hm_mpo)*(hm_mpo*hm_mps))
@time println("hm_mps'*((hm_mpo*hm_mpo)*hm_mps) = ", hm_mps'*((hm_mpo*hm_mpo)*hm_mps))
@time println("hm_mps'*(hm_mpo*(hm_mpo*hm_mps)) = ", hm_mps'*(hm_mpo*(hm_mpo*hm_mps)))