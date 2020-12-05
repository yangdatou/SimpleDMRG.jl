abstract type ModelSystem{T<:Number}                 end
abstract type SpinModel{T<:Number} <: ModelSystem{T} end

const zero_matrix = [0.0 0.0;  0.0  0.0]
const id_matrix   = [1.0 0.0;  0.0  1.0]
const sz_matrix   = [1.0 0.0;  0.0 -1.0]
const sx_matrix   = [0.0 1.0;  1.0  0.0]
const sp_matrix   = [0.0 1.0;  0.0  0.0]
const sm_matrix   = [0.0 0.0;  1.0  0.0]

struct IsingModel{T<:Number}      <: SpinModel{T}
    # construct local operator for Ising model
    # - h:  strength of external field
    # - J:  coupling constant
    # the local operator has shape (3, 3, 2, 2)

    phy_dim  ::Int
    bond_dim ::Int
    h_val    ::T
    j_val    ::T
end

function IsingModel(h_val, j_val; T=Float64)
    phy_dim   = 2
    bond_dim  = 3
    h_val_    = convert(T, h_val)::T
    j_val_    = convert(T, j_val)::T
    return IsingModel{T}(phy_dim, bond_dim, h_val_::T, j_val_::T)
end

struct HeisenbergModel{T<:Number} <: SpinModel{T}
    # construct local operator for Heisenberg model
    # - h:  strength of external field
    # - J:  coupling constant
    # - Jz: coupling constant
    # the local operator has shape (5, 5, 2, 2)

    phy_dim  ::Int
    bond_dim ::Int

    h_val  ::T
    j_val  ::T
    jz_val ::T
end

function HeisenbergModel(h_val, j_val, jz_val; T=Float64)
    phy_dim   = 2
    bond_dim  = 5

    h_val_    = convert(T, h_val)::T
    j_val_    = convert(T, j_val)::T
    jz_val_   = convert(T, jz_val)::T
    return HeisenbergModel{T}(phy_dim, bond_dim, h_val_::T, j_val_::T, jz_val_::T)
end

struct HubbardModel{T<:Number} <: SpinModel{T}
    # construct local operator for Heisenberg model
    # - h:  strength of external field
    # - J:  coupling constant
    # - Jz: coupling constant
    # the local operator has shape (5, 5, 2, 2)

    phy_dim  ::Int
    bond_dim ::Int

    u_val    ::T
    mu_val   ::T
end

function HubbardModel(u_val, mu_val; T=Float64)
    phy_dim   = 2
    bond_dim  = 5

    u_val_     = convert(T, u_val)::T
    mu_val_    = convert(T, mu_val)::T
    return HeisenbergModel{T}(phy_dim, bond_dim, u_val_::T, mu_val_::T)
end

function get_phys_dim(m::ModelSystem{T}) where {T}
    return m.phy_dim
end

function get_local_operator_tensor(model::IsingModel{T}) where {T}
    n  = model.bond_dim
    m  = model.phy_dim
    h  = model.h_val
    j  = model.j_val

    tmp = zeros(T,n,n,m,m)

    tmp[1,1,:,:] = id_matrix
    tmp[3,3,:,:] = id_matrix

    tmp[2,1,:,:] = sz_matrix
    tmp[3,1,:,:] = -h*sx_matrix
    tmp[3,2,:,:] = -j*sz_matrix

    return tmp::Array{T,4}
end

function get_local_operator_tensor(model::HeisenbergModel{T}) where {T}
    n  = model.bond_dim
    m  = model.phy_dim
    h  = model.h_val
    j  = model.j_val
    jz = model.jz_val

    tmp = zeros(T,n,n,m,m)

    tmp[1,1,:,:] = id_matrix
    tmp[5,5,:,:] = id_matrix

    tmp[2,1,:,:] = sp_matrix
    tmp[3,1,:,:] = sm_matrix
    tmp[4,1,:,:] = sz_matrix
    tmp[5,1,:,:] = -h*sz_matrix

    tmp[5,2,:,:] = (j/2)*sm_matrix 
    tmp[5,3,:,:] = (j/2)*sp_matrix 
    tmp[5,4,:,:] = jz   *sz_matrix

    return tmp::Array{T,4}
end

function get_local_operator_tensor(model::HubbardModel{T}) where {T}
    n   = model.bond_dim
    m   = model.phy_dim
    u   = model.u_val
    mu  = model.mu_val

    c_up = kron(sm_matrix, id_matrix)
    c_dn = kron(id_matrix, sm_matrix)
    id2  = kron(id_matrix, id_matrix)
    n_up = c_up' * c_up
    n_dn = c_dn' * c_dn

    p_up = (id2 - 2*c_up'*c_up) # Spin up parity operator
    p_dn = (id2 - 2*c_dn'*c_dn) # Spin down parity operator

    tmp = zeros(T,n,n,m,m)
    tmp[1, 1, :, :] = id2
    tmp[2, 1, :, :] = c_up'
    tmp[3, 1, :, :] = c_dn'
    tmp[4, 1, :, :] = c_up
    tmp[5, 1, :, :] = c_dn
    tmp[6, 1, :, :] = u*(n_up * n_dn) - mu*(n_up + n_dn)
    tmp[6, 2, :, :] =  c_up  * p_up  # Must multiply by the parity operator to get 
    tmp[6, 3, :, :] =  c_dn  * p_dn  # correct off-site commutation relations!
    tmp[6, 4, :, :] = -c_up' * p_up
    tmp[6, 5, :, :] = -c_dn' * p_dn
    tmp[6, 6, :, :] = id2
    
    return tmp::Array{T,4}
end