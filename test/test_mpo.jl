include("../src/SimpleDMRG.jl")
using .SimpleDMRG

const nn           = 20
const max_bond_dim = 50

function test_mpo(m::ModelSystem{T}, nn::Int) where T
    psi     = build_randn_mps(m, nn, bond_dim=max_bond_dim)
    h       = build_mpo(m,       nn)
    
    println("h = ", h)
    display(h)
    @time println("(psi'*h)*psi = ", (psi'*h)*psi)
    @time println("psi'*(h*psi) = ", psi'*(h*psi))
    @time println("((psi'*h)*h)*psi = ", ((psi'*h)*h)*psi)
    @time println("((psi'*h)*h)*psi = ", ((psi'*h)*h)*psi)
    @time println("(psi'*h)*(h*psi) = ", (psi'*h)*(h*psi))
    @time println("psi'*((h*h)*psi) = ", psi'*((h*h)*psi))
    @time println("psi'*(h*(h*psi)) = ", psi'*(h*(h*psi)))
end

ising_model = IsingModel(1.0, 1.0, T=ComplexF64)
test_mpo(ising_model, nn)
heis_model  = HeisenbergModel(1.0, 1.0, 1.0, T=ComplexF64)
test_mpo(heis_model,  nn)
hub_model   = HubbardModel(1.0, 1.0, T=ComplexF64)
test_mpo(hub_model,   nn)