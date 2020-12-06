
module SimpleDMRG
    using LinearAlgebra, TensorOperations, TensorCast
    using LowRankApprox, Arpack, Strided, SparseArrays
    using Printf

    export *, adjoint, getindex
    export MatrixProductOperator, MatrixProductState
    export ModelSystem, IsingModel, HeisenbergModel, HubbardModel
    export build_randn_mps, get_local_operator_tensor, build_mpo, get_bond_dims
    export contraction, kernel
    export left_canonical!, right_canonical!

    include("utils.jl")
    include("spin_models.jl")
    include("mpo.jl")
    include("mps.jl")
    include("kernel.jl")
    include("compression.jl")
    include("contraction.jl")

end