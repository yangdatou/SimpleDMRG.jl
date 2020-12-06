using Printf

include("../src/SimpleDMRG.jl")
using .SimpleDMRG
                                                                                                                        
ising_model = IsingModel(1.0, 1.0, T=ComplexF64)
tmp         = get_local_operator_tensor(ising_model)
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

heis_model = HeisenbergModel(1.0, 1.0, 1.0, T=ComplexF64)
tmp        = get_local_operator_tensor(heis_model)
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

hub_model = HubbardModel(1.0, 1.0, T=ComplexF64)
tmp       = get_local_operator_tensor(hub_model)
println("\n#######################################")
println("Hubbard model local operator matrix:")
println("#######################################")
for i in 1:6
    for j in 1:6
        @printf("\ntmp[%d,%d,:,:] = \n", i, j)
        display(tmp[i,j,:,:])
        println()
    end
end