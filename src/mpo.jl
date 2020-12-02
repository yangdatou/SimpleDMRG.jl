"""
Part of matrix product operator would be diagrammatically represented,

   last_sgm            this_sgm            next_sgm
       |                   |                   |
    a)-•--(last_a last_a)--•--(this_a this_a)--•--(next_a
       |                   |                   |
   last_rho            this_rho            next_rho
"""

struct MatrixProductOperator{T<:Number} <: TensorProduct
    _data::OperatorTensor
end

function MatrixProductOperator(w::Array{T,4}, sys_size::Int) where {T}
    sys_size >= 2 || throw(DomainError(sys_size, "sys_size shoule be larger than 2"))

    n  = size(w)[1]::Int
    m  = size(w)[3]::Int

    mpo_tensors    = Vector{OperatorTensor}(undef, sys_size)
    mpo_tensors[1] = OperatorTensor(w[end:end, :, :, :])
    for l in 2:(sys_size-1)
        mpo_tensors[l] = OperatorTensor(w)
    end
    mpo_tensors[sys_size] = OperatorTensor(w[:, 1:1, :, :])
    return MatrixProductOperator{T}(mpo_tensors)
end

function MatrixProductOperator(m::ModelSystem, sys_size::Int)
    sys_size >= 2 || throw(DomainError(sys_size, "sys_size shoule be larger than 2"))
    w = get_local_operator_tensor(m)
    return MatrixProductOperator(w, sys_size)
end

function get_data(t::MatrixProductOperator{T}, l::Int) where {T}
    return t._data[l]._data::Array{T,4}
end

function get_sys_size(the_mpo::MatrixProductOperator{T})::Int where {T}
    return length(the_mpo._data)
end

function get_phys_dim(the_mpo::MatrixProductOperator)::Int
    return length(the_mpo[2][1, 1, 1, :])
end

function get_bond_dims(the_mpo::MatrixProductOperator)::Array{Tuple{Int,Int},1}
    sys_size  = get_sys_size(the_mpo)
    bond_dims = [size(the_mpo[i][:, :, 1, 1]) for i in 1:sys_size]
    return bond_dims
end

function Base.show(io::IO, ::MIME"text/plain", the_mpo::MatrixProductOperator)
    sys_size  = get_sys_size(the_mpo)
    phys_dim  = get_phys_dim(the_mpo)
    bond_dims = get_bond_dims(the_mpo)
    println(io, "\n########################################################################")
    println(io, "Matrix Product Operator on $sys_size sites")
    _show_mpo_dims(io, sys_size, phys_dim, bond_dims)
    println(io, "\n########################################################################\n")
end

function _show_mpo_dims(io::IO, sys_size::Int, phys_dim::Int, bond_dims::Array{Tuple{Int,Int},1})
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

function Base.show(io::IO, the_mpo::MatrixProductOperator)
    sys_size  = get_sys_size(the_mpo)
    print(io, "Matrix Product Operator on $sys_size sites")
end