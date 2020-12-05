"""
    contraction(the_adj_mps::Adjoint{T, MatrixProductState{T}}, the_mps::MatrixProductState{T}) where {T}

representing

        •--(last_b last_b)--•--(this_b this_b)--•  
        |                   |                   |      
    last_sgm            this_sgm            next_sgm 
    last_sgm            this_sgm            next_sgm
        |                   |                   |    
        •--(last_a last_a)--•--(this_a this_a)--•    
"""
function contraction(the_adj_mps::Adjoint{T, MatrixProductState{T}}, the_mps::MatrixProductState{T}) where {T}
    sys_size  = get_sys_size(the_mps)

    tmp_mps_tensor1::Array{T,3} = get_data(the_adj_mps, 1)
    tmp_mps_tensor2::Array{T,3} = get_data(the_mps, 1)
    @tensor tmp[this_b, this_a]  := tmp_mps_tensor1[this_b, last_b, sgm] * tmp_mps_tensor2[last_b, this_a, sgm]

    for l in 2:sys_size-1
        tmp_mps_tensor1  = get_data(the_adj_mps, l)
        tmp_mps_tensor2  = get_data(the_mps, l)
        @tensor new[this_b, this_a] := tmp_mps_tensor1[this_b, last_b, sgm] * tmp[last_b, last_a] * tmp_mps_tensor2[last_a, this_a, sgm]
        tmp = new::Array{T,2}
    end

    tmp_mps_tensor1 = get_data(the_adj_mps, sys_size)
    tmp_mps_tensor2 = get_data(the_mps, sys_size)
    @tensor tmp[] := tmp[last_b, last_a] * tmp_mps_tensor1[this_b, last_b, sgm] * tmp_mps_tensor2[last_a, this_b, sgm]

    return tmp[1]::T
end

function Base.:(*)(the_adj_mps::Adjoint{T, MatrixProductState{T}}, the_mps::MatrixProductState{T}) where {T}
    return contraction(the_adj_mps::Adjoint{T, MatrixProductState{T}}, the_mps::MatrixProductState{T})::T
end

"""
    ontraction(the_mpo::MatrixProductOperator{T}, the_mps::MatrixProductState{T}) where {T}

representing

    last_rho            this_rho            next_rho
        |                   |                   |     
        •--(last_b last_b)--•--(this_b this_b)--•  
        |                   |                   |      
    last_sgm            this_sgm            next_sgm 
    last_sgm            this_sgm            next_sgm
        |                   |                   |    
        •--(last_a last_a)--•--(this_a this_a)--•    
"""

function contraction(the_mpo::MatrixProductOperator{T}, the_mps::MatrixProductState{T}) where {T}
    sys_size          = get_sys_size(the_mps)
    new_mps_tensors   = Vector{Array{T,3}}(undef, sys_size)

    for l in 1:sys_size
        tmp_mpo_tensor::Array{T,4} = get_data(the_mpo, l)
        tmp_mps_tensor::Array{T,3} = get_data(the_mps, l)
        @reduce tmp[(last_b, last_a), (this_b, this_a), rho] :=  sum(sgm) tmp_mpo_tensor[last_b, this_b, rho, sgm] * tmp_mps_tensor[last_a, this_a, sgm]
        new_mps_tensors[l]         = tmp::Array{T,3}
    end

    return MatrixProductState{T}(new_mps_tensors)
end

function Base.:(*)(the_mpo::MatrixProductOperator{T}, the_mps::MatrixProductState{T}) where {T}
    return contraction(the_mpo::MatrixProductOperator{T}, the_mps::MatrixProductState{T})::MatrixProductState{T}
end

"""
    contraction(the_mpo1::MatrixProductOperator{T}, the_mpo2::MatrixProductOperator{T}) where {T}

representing

    last_rho            this_rho            next_rho
        |                   |                   |     
        •--(last_b last_b)--•--(this_b this_b)--•  
        |                   |                   |      
    last_sgm            this_sgm            next_sgm 
    last_sgm            this_sgm            next_sgm
        |                   |                   |    
        •--(last_a last_a)--•--(this_a this_a)--•
        |                   |                   |     
    last_lam            this_lam            next_lam
"""

function contraction(the_mpo1::MatrixProductOperator{T}, the_mpo2::MatrixProductOperator{T}) where {T}
    sys_size        = get_sys_size(the_mpo1)
    new_mpo_tensors = Vector{Array{T,4}}(undef, sys_size)

    for l in 1:sys_size
        tmp_mpo_tensor1::Array{T,4} = get_data(the_mpo1, l)
        tmp_mpo_tensor2::Array{T,4} = get_data(the_mpo2, l)
        @reduce tmp[(last_b, last_a), (this_b, this_a), rho, lam] :=  sum(sgm) tmp_mpo_tensor1[last_b, this_b, rho, sgm] * tmp_mpo_tensor2[last_a, this_a, sgm, lam]
        new_mpo_tensors[l]          = tmp::Array{T,4}
    end

    return MatrixProductOperator{T}(new_mpo_tensors)
end

function Base.:(*)(the_mpo1::MatrixProductOperator{T}, the_mpo2::MatrixProductOperator{T}) where {T}
    return contraction(the_mpo1::MatrixProductOperator{T}, the_mpo2::MatrixProductOperator{T})::MatrixProductOperator{T}
end

"""
    contraction(the_adj_mps::Adjoint{T,MatrixProductState{T}}, the_mpo::MatrixProductOperator{T}) where {T}

representing

        •--(last_b last_b)--•--(this_b this_b)--•  
        |                   |                   |      
    last_sgm            this_sgm            next_sgm 
    last_sgm            this_sgm            next_sgm
        |                   |                   |    
        •--(last_a last_a)--•--(this_a this_a)--•
        |                   |                   |     
    last_lam            this_lam            next_lam
"""

function contraction(the_adj_mps::Adjoint{T,MatrixProductState{T}}, the_mpo::MatrixProductOperator{T}) where {T}
    sys_size        = get_sys_size(the_adj_mps)
    the_mps         = the_adj_mps.parent
    new_mps_tensors = Vector{Array{T,3}}(undef, sys_size)

    for l in 1:sys_size
        tmp_mpo_tensor::Array{T,4} = dg(get_data(the_mpo, sys_size-l+1))
        tmp_mps_tensor::Array{T,3} =    get_data(the_mps,            l)
        @reduce tmp[(last_b, last_a), (this_b, this_a), lam] := sum(sgm) tmp_mps_tensor[last_b, this_b, sgm] * tmp_mpo_tensor[last_a, this_a, sgm, lam]
        new_mps_tensors[l]         = tmp::Array{T,3}
    end

    new_mps = MatrixProductState{T}(new_mps_tensors)
    return adjoint(new_mps)
end

function Base.:*(the_adj_mps::Adjoint{T,MatrixProductState{T}}, the_mpo::MatrixProductOperator{T}) where {T}
    return contraction(the_adj_mps::Adjoint{T,MatrixProductState{T}}, the_mpo::MatrixProductOperator{T})::Adjoint{T,MatrixProductState{T}}
end