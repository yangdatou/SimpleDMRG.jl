function left_canonical!(the_mps::MatrixProductState{T}; max_bond_dim::Int=50) where {T}
    sys_size          = get_sys_size(the_mps)
    phys_dim          = get_phys_dim(the_mps)
    new_mps_tensors   = Vector{Array{T,3}}(undef, sys_size)
    
    tmp_mps_tensor     = get_data(the_mps, 1)
    @cast tmp_mps_matrix[(sgm, last_b), this_b] := tmp_mps_tensor[last_b, this_b, sgm]
    u, s, v            = psvd(tmp_mps_matrix, rank=max_bond_dim)
    @cast tmp[last_b, this_b, sgm]              := u[(sgm, last_b), this_b] (sgm:phys_dim)
    new_mps_tensors[1] = tmp
    
    for l in 2:sys_size
        tmp_mps_tensor = get_data(the_mps, l)
        @tensor tmp_mps_tensor_p[last_b, this_b, sgm] := (Diagonal(s)*v')[last_b, last_a] * tmp_mps_tensor[last_a, this_b, sgm]
        @cast   tmp_mps_matrix[(sgm, last_b), this_b] := tmp_mps_tensor_p[last_b, this_b, sgm]
        u, s, v = psvd(tmp_mps_matrix, rank=max_bond_dim)
        @cast tmp[last_b, this_b, sgm]                := u[(sgm, last_b), this_b] (sgm:phys_dim)
        new_mps_tensors[l] = tmp
    end

    return MatrixProductState{T}(new_mps_tensors)
end

function right_canonical!(the_mps::MatrixProductState{T}; max_bond_dim::Int=50) where {T}
    sys_size          = get_sys_size(the_mps)
    phys_dim          = get_phys_dim(the_mps)
    new_mps_tensors   = Vector{Array{T,3}}(undef, sys_size)
    
    tmp_mps_tensor            = get_data(the_mps, sys_size)    
    @cast tmp_mps_matrix[last_b, (sgm, this_b)] := tmp_mps_tensor[last_b, this_b, sgm]
    u, s, v                   = psvd(tmp_mps_matrix, rank=max_bond_dim)
    @cast tmp[last_b, this_b, sgm]              := v'[last_b, (sgm, this_b)] (sgm:phys_dim)
    new_mps_tensors[sys_size] = tmp
    
    for l in (sys_size-1):-1:1
        tmp_mps_tensor = get_data(the_mps, l)
        @tensor tmp_mps_tensor_p[last_b, this_b, sgm] := tmp_mps_tensor[last_b, this_a, sgm] * (u*Diagonal(s))[this_a, this_b]
        @cast   tmp_mps_matrix[last_b, (sgm, this_b)] := tmp_mps_tensor_p[last_b, this_b, sgm]
        u, s, v = psvd(tmp_mps_matrix, rank=max_bond_dim)
        @cast tmp[last_b, this_b, sgm]                := v'[last_b, (sgm, this_b)] (sgm:phys_dim)
        new_mps_tensors[l] = tmp
    end

    return MatrixProductState{T}(new_mps_tensors)
end
