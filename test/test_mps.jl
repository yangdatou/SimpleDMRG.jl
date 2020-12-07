using Printf

include("../src/SimpleDMRG.jl")
using .SimpleDMRG

const nn           = 20
const max_bond_dim = 50

function test_mps(m::ModelSystem{T}, nn::Int) where T
    psi     = build_randn_mps(m, nn, bond_dim=max_bond_dim)
    
    println("psi  = ", psi)
    display(psi)
    println("psi' = ", psi')
    display(psi')
    @time println("psi'*psi = ", psi'*psi)
end

ising_model = IsingModel(1.0, 1.0, T=ComplexF64)
test_mps(ising_model, nn)
heis_model  = HeisenbergModel(1.0, 1.0, 1.0, T=ComplexF64)
test_mps(heis_model,  nn)
hub_model   = HubbardModel(1.0, 1.0, T=ComplexF64)
test_mps(hub_model,   nn)