#==================================================================================#
# Description
# -----------
#   This module provides standard routines and specialized ones for computing the
#   reduced density and entanglement entropy. 
#
#   The standard routines simply use reshape method to calculate reduced density
#   matrix for wavefunctions in Hilbert space with tensor-product stucture. The
#   specialized ones take advantange of restricted Hilbert space and U(1) symmetry
#   to speed up the calculation.
#
#   Version:  0.1
#   Created:  2024-12-31 14:54
#    Author:  Zhiyuan Yao, zhiyuan.yao@icloud.com
# Institute:  Lanzhou Center for Theoretical Physics, Lanzhou University
#==================================================================================#
__precompile__()

module ResEE

using LinearAlgebra, SparseArrays
using OffsetArrays

#--------------------------------------------------------------------------------------------------
# reudced density matrix construction and entanglement calculation routines
#--------------------------------------------------------------------------------------------------
export check_rho, get_rho, get_rho!, get_entEntropy, get_entEntropy!
include("./EE.jl")

# restricted reshape method for constrained system and U1 symmetry further speedup by blocking 
export get_rho_M_ijmap, get_rhos_Ms_ijkmap
include("./rEE.jl")

end
