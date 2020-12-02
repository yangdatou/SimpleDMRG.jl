max_int = typemax(Int)

function left_canonical(the_mps::MatrixProductState{T}; d_cut::Int=max_int) where {T}
    sys_size          = get_sys_size(the_mps)
    phys_dim          = get_phys_dim(the_mps)
    new_state_tensors = Vector{StateTensor}(undef, sys_size)
    
    tmp_mps_tensor = the_mps[1]    
    @cast tmp_mps_matrix[(sgm, 1), last_b] := tmp_mps_tensor[1, last_b, sgm]
    u, s, v = psvd(tmp_mps_matrix, rank=d_cut)

    @cast tmp_new_tensor[1, last_b, sgm] := u[(sgm, 1), last_b] (sgm:phys_dim)
    new_state_tensors[1] = StateTensor(tmp_new_tensor)
    
    for i in 2:sys_size
        tmp_mps_tensor = the_mps[i]
        @tensor tmp_mps_tensor_p[last_b, this_b, sgm] := (Diagonal(s)*v')[last_b, last_a] * tmp_mps_tensor[last_a, this_b, sgm]
        @cast tmp_mps_matrix[(sgm, last_b), this_b]   := tmp_mps_tensor_p[last_b, this_b, sgm]
        u, s, v = psvd(tmp_mps_matrix, rank=d_cut)
        @cast tmp_new_tensor[last_b, this_b, sgm]     := u[(sgm, last_b), this_b] (sgm:phys_dim)
        new_state_tensors[l] = StateTensor(tmp_new_tensor)
    end

    return MatrixProductState{T}(new_state_tensors)
end

function right_canonical(the_mps::MatrixProductState{T}; d_cut::Int=max_int) where {T}
    sys_size          = get_sys_size(the_mps)
    phys_dim          = get_phys_dim(the_mps)
    new_state_tensors = Vector{StateTensor}(undef, sys_size)
    
    tmp_mps_tensor = the_mps[sys_size]    
    @cast tmp_mps_matrix[this_b, (sgm, 1)] := tmp_mps_tensor[this_b, 1, sgm]
    u, s, v = psvd(tmp_mps_matrix, rank=d_cut)

    @cast tmp_new_tensor[this_b, 1, sgm] := u[this_b, (sgm, 1)] (sgm:phys_dim)
    new_state_tensors[sys_size] = StateTensor(tmp_new_tensor)
    
    for l in (sys_size-1):-1:1
        tmp_mps_tensor = the_mps[l]
        @tensor tmp_mps_tensor_p[last_b, this_b, sgm] := tmp_mps_tensor[last_b, this_a, sgm] * (u*Diagonal(s))[this_a, this_b]
        @cast tmp_mps_matrix[last_b, (sgm, this_b)]   := tmp_mps_tensor_p[last_b, this_b, sgm]
        u, s, v = psvd(tmp_mps_matrix, rank=d_cut)
        @cast tmp_new_tensor[last_b, this_b, sgm]     := v'[last_b, (sgm, this_b)] (sgm:phys_dim)
        new_state_tensors[l] = StateTensor(tmp_new_tensor)
    end

    return MatrixProductState{T}(new_state_tensors)
end
