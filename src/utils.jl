not(x) = ~x
dg(t::Array{T, 4}) where {T} = permutedims(conj.(t), (2, 1, 3, 4))
dg(t::Array{T, 3}) where {T} = permutedims(conj.(t), (2, 1, 3))

abstract type AbstractTensor end