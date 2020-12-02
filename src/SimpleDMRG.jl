
module SimpleDMRG
    using LinearAlgebra, TensorOperations
    using TensorCast, LowRankApprox
    using Arpack, Strided, SparseArrays

    export *, adjoint, getindex, randn
    export MatrixProductOperator, MatrixProductState
    export HeisenbergModel, IsingModel
    export get_local_operator_tensor, contraction
    export left_canonical, right_canonical

    include("utils.jl")
    include("spin_models.jl")
    include("mpo.jl")
    include("mps.jl")
    include("compression.jl")
    include("contraction.jl")

end