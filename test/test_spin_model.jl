using Printf

include("../src/SimpleDMRG.jl")
using .SimpleDMRG
                                                                                                                        
im_model = IsingModel(1.0, 1.0, T=ComplexF64)
tmp = get_local_operator_tensor(im_model)
println("\n##################################")
println("Ising modle local operator matrix:")
println("##################################")
for i in 1:3
    for j in 1:3
        @printf("\ntmp[%d,%d,:,:] = \n\n", i, j)
        display(tmp[i,j,:,:])
        println()
    end
end

hm_model = HeisenbergModel(1.0, 1.0, 1.0, T=ComplexF64)
tmp = get_local_operator_tensor(hm_model)
println("\n#######################################")
println("Heisenberg model local operator matrix:")
println("#######################################")
for i in 1:5
    for j in 1:5
        @printf("\ntmp[%d,%d,:,:] = \n", i, j)
        display(tmp[i,j,:,:])
        println()
    end
end