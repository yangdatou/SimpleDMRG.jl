abstract type ModelSystem              end
abstract type SpinModel <: ModelSystem end

zero_matrix = [0.0 0.0;  0.0  0.0]
id_matrix   = [1.0 0.0;  0.0  1.0]
sz_matrix   = [0.5 0.0;  0.0 -0.5]
sx_matrix   = [0.0 0.5;  0.5  0.0]
sp_matrix   = [0.0 1.0;  0.0  0.0]
sm_matrix   = [0.0 0.0;  1.0  0.0]

struct IsingModel{FloatType<:Real} <: SpinModel
    # construct local operator for Ising model
    # - h:  strength of external field
    # - J:  coupling constant
    # the local operator has shape (3, 3, 2, 2)

    phy_dim::Int
    op_dim ::Int
    h_val  ::FloatType
    j_val  ::FloatType
end

function IsingModel(h_val::Number, j_val::Number; FloatType=Float64)
    phy_dim = 2
    op_dim  = 3
    return IsingModel{FloatType}(phy_dim, op_dim, h_val::FloatType, j_val::FloatType)
end

struct HeisenbergModel{FloatType<:Real} <: SpinModel
    # construct local operator for Heisenberg model
    # - h:  strength of external field
    # - J:  coupling constant
    # - Jz: coupling constant
    # the local operator has shape (5, 5, 2, 2)

    phy_dim::Int
    op_dim ::Int

    h_val  ::FloatType
    j_val  ::FloatType
    jz_val ::FloatType
end

function HeisenbergModel(h_val::Number, j_val::Number, jz_val::Number; FloatType=Float64)
    phy_dim = 2
    op_dim  = 5
    return HeisenbergModel{FloatType}(phy_dim, op_dim, h_val::FloatType, j_val::FloatType, jz_val::FloatType)
end

function get_phys_dim(m::ModelSystem)
    return m.phy_dim
end

function get_local_operator_tensor(model::IsingModel; FloatType=Float64)
    n  = model.op_dim
    m  = model.phy_dim
    h  = model.h_val
    j  = model.j_val

    tmp = zeros(FloatType,n,n,m,m)

    tmp[1,1,:,:] = id_matrix
    tmp[3,3,:,:] = id_matrix

    tmp[2,1,:,:] = sz_matrix
    tmp[3,1,:,:] = -h*sx_matrix
    tmp[3,2,:,:] = -j*sz_matrix

    return tmp
end

function get_local_operator_tensor(model::HeisenbergModel)
    n  = model.op_dim
    m  = model.phy_dim
    h  = model.h_val
    j  = model.j_val
    jz = model.jz_val

    tmp = zeros(n,n,m,m)

    tmp[1,1,:,:] = id_matrix
    tmp[5,5,:,:] = id_matrix

    tmp[2,1,:,:] = sp_matrix
    tmp[3,1,:,:] = sm_matrix
    tmp[4,1,:,:] = sz_matrix
    tmp[5,1,:,:] = -h*sz_matrix

    tmp[5,2,:,:] = (j/2)*sm_matrix 
    tmp[5,3,:,:] = (j/2)*sp_matrix 
    tmp[5,4,:,:] = jz   *sz_matrix

    return tmp
end