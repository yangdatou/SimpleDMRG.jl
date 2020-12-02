"""
Part of matrix product state would be diagrammatically represented,

   last_sgm            this_sgm            next_sgm
       |                   |                   |
    a)-•--(last_a last_a)--•--(this_a this_a)--•--(next_a

   last_rho            this_rho            next_rho
       |                   |                   |
    b)-•--(last_b last_b)--•--(this_b this_b)--•--(next_b

"""

struct MatrixProductState{T<:Number} <: TensorProduct
    _data::Vector{StateTensor}
end

function MatrixProductState(mps_tensors::Vector{Array{T,3}}) where {T}
    return new([StateTensor(t) for t in mps_tensors])
end

function get_data(t::MatrixProductState{T}, l::Int) where {T}
    return t._data[l]._data::Array{T,3}
end

function get_sys_size(the_mps::MatrixProductState{T})::Int where {T}
    return length(the_mps._data)
end

function get_phys_dim(the_mps::MatrixProductState)::Int
    return length(the_mps[2][1, 1, :])
end

function get_bond_dims(the_mps::MatrixProductState)::Array{Tuple{Int,Int},1}
    sys_size  = get_sys_size(the_mps)
    bond_dims = [size(the_mps[i][:, :, 1]) for i in 1:sys_size]
    return bond_dims
end

function get_randn_mps(phys_dim::Int, sys_size::Int, bond_dim::Int; T=Float64)
    temp_mps_tensors = [randn(T, 1, bond_dim, phys_dim), [randn(T, bond_dim, bond_dim, phys_dim) for _ in 2:(sys_size-1)]..., randn(T, bond_dim, 1, phys_dim)]
    return MatrixProductState{T}(temp_mps_tensors) |> left_canonical |> right_canonical
end

function get_randn_mps(m::ModelSystem, sys_size::Int, bond_dim::Int; T=Float64)
    phys_dim = get_phys_dim(m)
    return get_randn_mps(phys_dim::Int, sys_size::Int, bond_dim::Int, T=T)::MatrixProductState{T}
end

function Base.show(io::IO, ::MIME"text/plain", the_mps::MatrixProductState{T}) where {T}
    sys_size  = get_sys_size(the_mps)
    phys_dim  = get_phys_dim(the_mps)
    bond_dims = get_bond_dims(the_mps)
    println(io, "\n########################################################################")
    println(io, "Matrix Product State on $sys_size sites")
    _show_mps_dims(io, sys_size, phys_dim, bond_dims)
    println(io, "\n########################################################################\n")
end

function _show_mps_dims(io::IO, sys_size::Int, phys_dim::Int, bond_dims::Array{Tuple{Int,Int},1})
    println(io, "  Physical dimension: $phys_dim")
    print(io,   "  Bond dimensions:   ")
    if sys_size > 4
        for i in 1:4
            print(io, bond_dims[i], " × ")
        end
        print(io, " ... × ", bond_dims[sys_size])
    else
        for i in 1:(sys_size-1)
            print(io, bond_dims[i], " × ")
        end
        print(io, bond_dims[sys_size])
    end
end

function Base.show(io::IO, the_mps::MatrixProductState)
    sys_size  = get_sys_size(the_mps)
    print(io, "MPS on $sys_size sites")
end

function Base.adjoint(the_mps::MatrixProductState{T}) where {T}
    return Adjoint{T, MatrixProductState{T}}(the_mps)
end

function get_sys_size(the_adj_mps::Adjoint{T, MatrixProductState{T}})::Int where {T}
    the_mps = the_adj_mps.parent
    return get_sys_size(the_mps)
end

function get_phys_dim(the_adj_mps::Adjoint{T, MatrixProductState{T}})::Int where {T}
    the_mps = the_adj_mps.parent
    return get_phys_dim(the_mps)
end

function get_bond_dims(the_adj_mps::Adjoint{T, MatrixProductState{T}})::Array{Tuple{Int,Int},1} where {T}
    the_mps = the_adj_mps.parent
    return get_bond_dims(the_mps)
end

function Base.show(io::IO, ::MIME"text/plain", the_adj_mps::Adjoint{T, MatrixProductState{T}}) where {T}
    sys_size  = get_sys_size(the_adj_mps)
    phys_dim  = get_phys_dim(the_adj_mps)
    bond_dims = get_bond_dims(the_adj_mps)
    println(io, "\n########################################################################")
    println(io, "Adjoint Matrix Product State on $sys_size sites")
    _show_mps_dims(io, sys_size, phys_dim, bond_dims)
    println(io, "\n########################################################################\n")
end

function Base.show(io::IO, the_adj_mps::Adjoint{T, MatrixProductState{T}}) where {T}
    sys_size  = get_sys_size(the_adj_mps)
    print(io, "Adjoint MPS on $sys_size sites")
end

function Base.getindex(the_mps::Adjoint{T, MatrixProductState{T}}, args...) where {T}
    out = Base.getindex(reverse(the_mps.parent.tensors), args...)
    return permutedims(conj.(out), (2, 1, 3))
end