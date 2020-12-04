"""
    ground_state(the_mps_init::MatrixProductState{T}, the_mpo::MatrixProductOperator{T}; iter_max=10, quiet=false, tol=1e-8) where {T}
Perform the finite system density matrix renormalization group
algorithm. First this will build up the R expressions, then do right
and left sweeps until either
 1) The state converges to an eigenstate `ϕ` such that
    ϕ' * the_mpo * the_mpo * ϕ ≈ (ϕ' * the_mpo * ϕ) 
to the requested tolerance `tol`
 2) The energy eigenvalue stops changing (possible signaling the algorithm is 
stuck in a local minimum)
 3) The number of full (right and left) sweeps exceeds `iter_max`. 
Setting `quiet=true` will suppress notifications about the algorithm's
progress but *not* warnings due to non-convergence.
"""
function is_eigen(the_mpo::MatrixProductOperator{T}, the_mps::MatrixProductState{T}; tol=1e-8)
    isapprox(the_mps' * the_mps, rtol=tol) || error("the_mps is not canonicalized")
    return isapprox(the_mps' * (the_mpo * the_mpo * the_mps), (the_mps' * (the_mpo * the_mps))^2, rtol=tol)
end

function sweep!(the_mps::MatrixProductState{T}, the_mpo::MatrixProductOperator{T},
                left_blocks::Vector{Array{T, 3}}, right_blocks::Vector{Array{T, 3}};
                verbose=3, d_cut::Int=max_int) where {T}
    e_right   = zero(T)
    e_left    = zero(T)
    sys_size  = get_sys_size(the_mps)

    if not(isdefined(left_blocks,1)) && not(isdefined(right_blocks,sys_size))
        tmp_right_block    = ones(T, 1, 1, 1)::Array{T, 3}
        for l in sys_size:-1:1
            tmp_mpo_tensor = get_data(the_mpo, l)::Array{T,4}
            tmp_mps_tensor = get_data(the_mps, l)::Array{T,3}
            @tensoropt new_right_block[last_a, last_b, last_c] := tmp_mpo_tensor[last_a, this_a, sgm, lam] * (conj.(tmp_mps_tensor))[last_b, this_b, sgm] *  tmp_mps_tensor[last_c, this_c, lam] * tmp_right_block[this_a, this_b, this_c]
            right_blocks[l]   = new_right_block::Array{T,3}
            tmp_right_block   = new_right_block::Array{T,3}
        end
    end

    verbose<5 || println(">>>>>>>>>> Performing Right Sweep >>>>>>>>>>")
    tmp_left_block = ones(T, 1, 1, 1)::Vector{Array{T, 3}}
    for l in 1:sys_size
        this_mpo_tensor = get_data(the_mpo, l)::Array{T,4}
        this_mps_tensor = get_data(the_mps, l)::Array{T,3}

        if l == sys_size
            tmp_right_block = ones(T, 1, 1, 1)::Vector{Array{T, 3}}
        elseif 1 <= l < sys_size
            tmp_right_block = right_blocks[l+1]::Array{T,3}
        end

        b1, b2, phys_dim      = size(this_mps_tensor)
        @cast this_mps_vector[(sgm, last_b, this_b)] := this_mps_tensor[last_b, this_b, sgm]

        @tensoropt hamil_matrix[(sgm, last_b, this_b), (lam, last_c, this_c)] := tmp_left_block[last_a, last_b, last_c] * this_mpo_tensor[last_a, this_a, sgm, lam] * tmp_right_block[this_a, this_b, this_c]

        eig_vals, eig_vecs    = eigs(hamil_matrix, v0=this_mps_vector, nev=1, which=:SR)
        e_left                = eig_vals[1]::T
        new_mps_vector        = eig_vecs[:,1]::Vector{T}

        @cast new_mps_matrix[(sgm, last_b), this_b] := new_mps_vector[(sgm, last_b, this_b)] (last_b:b1, this_b:b2, sgm:phys_dim)
        u, s, v               = psvd(new_mps_matrix, rank=d_cut)
        @cast new_mps_tensor[last_b, this_b, sgm] := u[(sgm, last_b), this_b] (sgm:phys_dim)
        the_mps[l]            = new_mps_tensor::Array{T,3}

        if 1 <= l < sys_size
            next_mps_tensor   = get_data(the_mps, l+1)::Array{T,3}
            @tensor next_mps_tensor_p[last_b, this_b, sgm] := (Diagonal(s)*v')[last_b, last_a] * next_mps_tensor[last_a, this_b, sgm]
            the_mps[l+1]      = next_mps_tensor_p::Array{T,3}
        end

        @tensoropt new_left_block[this_a, this_b, this_c] := tmp_mpo_tensor[last_a, this_a, sgm, lam] * (conj.(new_mps_tensor))[last_b, this_b, sgm] *  new_mps_tensor[last_c, this_c, lam] * tmp_left_block[last_a, last_b, last_c]

        left_blocks[l]        = new_left_block::Array{T,3}
        tmp_left_block        = new_left_block::Array{T,3}

        verbose<5 || @printf("site: %2d, energy: %.12f\n", l, real(e_left))
    end

    verbose<5 || println("<<<<<<<<<< Performing Left Sweep <<<<<<<<<<")
    for l in sys_size:-1:1
        this_mpo_tensor = get_data(the_mpo, l)::Array{T,4}
        this_mps_tensor = get_data(the_mps, l)::Array{T,3}

        if l == 1
            tmp_left_block = ones(T, 1, 1, 1)::Vector{Array{T, 3}}
        elseif 1 < l <= sys_size
            tmp_left_block = left_blocks[l-1]::Array{T,3}
        end

        b1, b2, phys_dim      = size(this_mps_tensor)
        @cast this_mps_vector[(sgm, last_b, this_b)] := this_mps_tensor[last_b, this_b, sgm]

        @tensoropt hamil_matrix[(sgm, last_b, this_b), (lam, last_c, this_c)] := tmp_left_block[last_a, last_b, last_c] * this_mpo_tensor[last_a, this_a, sgm, lam] * tmp_left_block[this_a, this_b, this_c]

        eig_vals, eig_vecs    = eigs(hamil_matrix, v0=this_mps_vector, nev=1, which=:SR)
        e_right               = eig_vals[1]::T
        new_mps_vector        = eig_vecs[:,1]::Vector{T}

        @cast new_mps_matrix[(sgm, last_b), this_b] := new_mps_vector[(sgm, last_b, this_b)] (last_b:b1, this_b:b2, sgm:phys_dim)
        u, s, v               = psvd(new_mps_matrix, rank=d_cut)
        @cast new_mps_tensor[last_b, this_b, sgm] := v'[(sgm, last_b), this_b] (sgm:phys_dim)
        the_mps[l]            = new_mps_tensor::Array{T,3}

        if 1 < l <= sys_size
            last_mps_tensor   = get_data(the_mps, l-1)::Array{T,3}
            @tensor last_mps_tensor_p[last_b, this_b, sgm] := (u*Diagonal(s))[last_b, last_a] * next_mps_tensor[last_a, this_b, sgm]
            the_mps[l-1]      = last_mps_tensor_p::Array{T,3}
        end

        @tensoropt new_right_block[last_a, last_b, last_c] := tmp_mpo_tensor[last_a, this_a, sgm, lam] * (conj.(new_mps_tensor))[last_b, this_b, sgm] *  new_mps_tensor[last_c, this_c, lam] * tmp_right_block[this_a, this_b, this_c]

        right_blocks[l]       = new_right_block::Array{T,3}
        tmp_right_block       = new_right_block::Array{T,3}

        verbose<5 || @printf("site: %2d, energy: %.12f\n", l, real(e_left))
    end

    return e_left, e_right
end

function kernel(the_mps_init::MatrixProductState{T}, the_mpo::MatrixProductOperator{T}; iter_max=10, verbose=3, tol=1e-8, d_cut::Int=max_int) where {T}
    the_mps      = copy(the_mps_init)

    verbose<3 || println("Computing right expressions")

    left_blocks  = Vector{Array{T,3}}(undef, sys_size)
    right_blocks = Vector{Array{T,3}}(undef, sys_size)

    is_converged = false
    iter_num     = 0
    ene0         = zero(T)
    enable_cache(maxsize=5*100000)

    while not(is_converged) && iter_num < iter_max
        e_left, e_right = sweep!(the_mps, the_mpo, left_blocks, right_blocks, verbose=verbose, d_cut=d_cut)
        iter_num += 1

        if is_eigen(the_mps, the_mpo, tol=tol)
            verbose < 3 || println("Converged in $iter_num iterations")
            is_converged = true
            ene0 = (e_left + e_right)/2
        end
    end

    clear_cache()
    return the_mps, ene0
end

function kernel(the_mpo::MatrixProductOperator{T}; iter_max::Int=10, verbose::Int=3, tol::Real=1e-8, T=Float64, d_cut::Int=max_int) where {T}
    bond_dims     = get_bond_dims(the_mpo)::Vector{Tuple{Int,Int}}
    the_mps_init  = get_randn_mps(m, bond_dims, T=T)::MatrixProductState{T}

    the_mps, ene0 = kernel(the_mps_init::MatrixProductState{T}, the_mpo::MatrixProductOperator, iter_max=iter_max, verbose=verbose, tol=tol, d_cut=d_cut)
    return the_mps, ene0
end

function kernel(m::ModelSystem, sys_size::Int; iter_max=10, verbose=3, tol=1e-8, T=Float64, d_cut::Int=max_int)
    the_mpo       = get_mpo(m, sys_size)::MatrixProductOperator{T}
    bond_dims     = get_bond_dims(the_mpo)::bond_dims::Vector{Tuple{Int,Int}}
    the_mps_init  = get_randn_mps(m, bond_dims, T=T)::MatrixProductState{T}

    the_mps, ene0 = kernel(the_mps_init::MatrixProductState{T}, the_mpo::MatrixProductOperator, iter_max=iter_max, verbose=verbose, tol=tol, d_cut=d_cut)
    return the_mps, ene0
end