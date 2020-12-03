"""
    ground_state(the_mps_init::MatrixProductState{L, T}, the_mpo::MPO{L, T}; maxiter=10, quiet=false, tol=1e-8) where {L, T}
Perform the finite system density matrix renormalization group
algorithm. First this will build up the R expressions, then do right
and left sweeps until either
 1) The state converges to an eigenstate `ϕ` such that
    ϕ' * the_mpo * the_mpo * ϕ ≈ (ϕ' * the_mpo * ϕ) 
to the requested tolerance `tol`
 2) The energy eigenvalue stops changing (possible signaling the algorithm is 
stuck in a local minimum)
 3) The number of full (right and left) sweeps exceeds `maxiter`. 
Setting `quiet=true` will suppress notifications about the algorithm's
progress but *not* warnings due to non-convergence.
"""

function sweep!(the_mps::MatrixProductState{T}, the_mpo::MatrixProductOperator{T},
                left_blocks::Vector{Array{T, 3}}, right_blocks::Vector{Array{T, 3}}; verbose=3) where {T}
    
    sys_size  = get_sys_size(the_mps)
    if not(isdefined(left_blocks,1)) && not(isdefined(right_blocks,1))
        tmp_right_block = ones(T, 1, 1, 1)
        for l in sys_size:-1:2
            tmp_mpo_tensor::Array{T,4} = get_data(the_mpo, l)
            tmp_mps_tensor::Array{T,3} = get_data(the_mps, l)
            @tensoropt tmp_right_block[last_b, last_a, last_c] := (conj.(tmp_mps_tensor))[last_a, this_a, sgm] * tmp_mpo_tensor[last_b, this_b, sgm, lam] * tmp_mps_tensor[last_c, this_c, lam] * tmp_right_block[this_b, this_a, this_c]
            right_blocks[l] = tmp_right_block
        end
    end

    verbose<3 || println("Performing right sweep")
    tmp_left_block = ones(T, 1, 1, 1)
    for l in 1:sys_size-1
        this_mpo_tensor::Array{T,4} = get_data(the_mpo, l)
        this_mps_tensor::Array{T,3} = get_data(the_mps, l)
        tmp_right_block = right_blocks[l]
    
        b1, b2, phys_dim = size(this_mps_tensor)
        @cast  this_mps_vector[(sgm, last_b, this_b)] := this_mps_tensor[last_b, this_b, sgm]
        @tensoropt hamil_matrix[(sgm, last_a, this_a), (lam, last_b, this_b)] = tmp_left_block[last_a, last_b, last_c] * this_mpo_tensor[last_c, this_c, sgm, lam] * tmp_right_block[this_a, this_b, this_c]
        eig_vals, eig_vecs = eigs(hamil_matrix, v0=this_mps_vector, nev=1, which=:SR)
        e1                   = eig_vals[1]::T
        this_mps_vector       = eig_vecs[:,1]::Vector{T}
        @cast this_mps_matrix[(sgm, last_b), this_b] := this_mps_vector[(sgm, last_b, this_b)] (last_b:b1, this_b:b2, sgm:phys_dim)
        u, s, v = svd(this_mps_matrix)
        @cast this_mps_tensor[last_b, this_b, sgm] := u[(sgm, last_b), this_b] (last_b:b1, this_b:b2, sgm:phys_dim)
        the_mps[l] = this_mps_tensor

        @tensoropt tmp_left_block[last_b, last_a, last_c] := (conj.(this_mps_tensor))[last_a, this_a, sgm] * this_mpo_tensor[last_b, this_b, sgm, lam] * this_mps_tensor[last_c, this_c, lam] * tmp_left_block[this_b, this_a, this_c]
        left_blocks[l] = tmp_left_block

        next_mps_tensor = get_data(the_mps, l+1)
        @tensor next_mps_tensor_p[last_b, this_b, sgm] := (Diagonal(s)*v')[last_b, last_a] * next_mps_tensor[last_a, this_b, sgm]
        the_mps[l+1] = next_mps_tensor_p
    end

    verbose<3 || println("Performing left sweep")
    tmp_right_block = ones(T, 1, 1, 1)
    for l in sys_size:-1:2
        this_mpo_tensor::Array{T,4} = get_data(the_mpo, l)
        this_mps_tensor::Array{T,3} = get_data(the_mps, l)
        tmp_left_block              = left_blocks[l]
    
        b1, b2, phys_dim = size(this_mps_tensor)
        @cast  this_mps_vector[(sgm, last_b, this_b)] := this_mps_tensor[last_b, this_b, sgm]
        @tensoropt hamil_matrix[(sgm, last_a, this_a), (lam, last_b, this_b)] = tmp_left_block[last_a, last_b, last_c] * this_mpo_tensor[last_c, this_c, sgm, lam] * tmp_right_block[this_a, this_b, this_c]

        eig_vals, eig_vecs = eigs(hamil_matrix, v0=this_mps_vector, nev=1, which=:SR)
        e1                    = eig_vals[1]::T
        this_mps_vector       = eig_vecs[:,1]::Vector{T}

        @cast this_mps_matrix[(sgm, last_b), this_b] := this_mps_vector[(sgm, last_b, this_b)] (last_b:b1, this_b:b2, sgm:phys_dim)
        u, s, v = svd(this_mps_matrix)

        @cast this_mps_tensor[last_b, this_b, sgm]   := v'[last_b, (sgm, this_b)] (last_b:b1, this_b:b2, sgm:phys_dim)
        the_mps[l] = this_mps_tensor

        @tensoropt tmp_right_block[last_b, last_a, last_c] := (conj.(this_mps_tensor))[last_a, this_a, sgm] * this_mpo_tensor[last_b, this_b, sgm, lam] * this_mps_tensor[last_c, this_c, lam] * tmp_right_block[this_b, this_a, this_c]
        right_blocks[l] = tmp_right_block

        last_mps_tensor = get_data(the_mps, l-1)
        @tensor last_mps_tensor_p[last_b, this_b, sgm] := (Diagonal(s)*v')[last_b, last_a] * last_mps_tensor[last_a, this_b, sgm]
        the_mps[l-1] = last_mps_tensor_p
    end


    return e1, e2
end

function kernel(the_mps_init::MatrixProductState{T}, the_mpo::MPO{L, T}; maxiter=10, verbose=3, tol=1e-8) where {T}
    the_mps = copy(the_mps_init)

    verbose<3 || println("Computing right expressions")

    left_blocks  = Vector{Array{T,3}}(undef, sys_size)
    right_blocks = Vector{Array{T,3}}(undef, sys_size)

    is_converged = false
    iter_num     = 0
    ene0         = zero(T)
    enable_cache(maxsize=5*10^9)

    while not(is_converged)
        e1, e2 = sweep!(the_mps, the_mpo, left_blocks, right_blocks, verbose=verbose)
        verbose<3 || println("Performing right sweep")
        L_exs, E₀′ = sweep!(right, ϕ, the_mpo, R_exs)

        quiet || println("Performing left sweep")
        R_exs, E₀  = sweep!(left,  ϕ, the_mpo, L_exs)

        iter_num += 1
        if iseigenstate(ϕ, the_mpo, tol=tol)
            quiet || println("Converged in $count iterations")
            converged = true
        elseif count > 1 && E₀ ≈ E₀′
                @warn """
Energy eigenvalue converged but state is not an eigenstate.
Consider either lowering your requested tolerance or 
implementing a warm-up algorithm to avoid local minima.
"""
            break
        elseif count >= maxiter
            @warn "Did not converge in $maxiter iterations"
            break
        end
    end

    clear_cache()
    ϕ, E₀
end