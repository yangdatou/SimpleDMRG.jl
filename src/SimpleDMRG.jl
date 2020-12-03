
module SimpleDMRG
    using LinearAlgebra, TensorOperations, TensorCast, LowRankApprox, Arpack, Strided, SparseArrays

    export *, adjoint, getindex
    export MatrixProductOperator, MatrixProductState
    export HeisenbergModel, IsingModel
    export get_randn_mps, get_local_operator_tensor, get_mpo
    export contraction
    export left_canonical, right_canonical

    include("utils.jl")
    include("spin_models.jl")
    include("mpo.jl")
    include("mps.jl")
    include("compression.jl")
    include("contraction.jl")

end