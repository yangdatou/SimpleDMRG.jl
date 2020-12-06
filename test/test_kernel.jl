include("../src/SimpleDMRG.jl")
using .SimpleDMRG

nn           = 10
max_bond_dim = 1000

function test_kernel(m::ModelSystem{T}, nn::Int) where T    
    @time psi, e   = kernel(m, nn, verbose=5, max_bond_dim=max_bond_dim)
    h              = get_mpo(m,       nn)

    println("e                = ", e)
    println("(psi'*h)*psi     = ", (psi'*h)*psi)
    println("(psi'*h)*(h*psi) = ", (psi'*h)*(h*psi))
    println("(psi'*h*psi)^2   = ", (psi'*h*psi)^2)
end

ising_model = IsingModel(1.0, 1.0, T=ComplexF64)
test_kernel(ising_model, nn)
heis_model  = HeisenbergModel(1.0, 1.0, 1.0, T=ComplexF64)
test_kernel(heis_model, nn)
hub_model   = HubbardModel(1.0, 1.0, T=ComplexF64)
test_kernel(hub_model, nn)