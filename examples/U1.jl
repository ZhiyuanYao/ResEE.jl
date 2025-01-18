#==============================================================================#
# Description
# -----------
#   This program demonstrates the calculation of the bipartite entanglement entropy 
#   of a spin-S state with given total magnetization, i.e. with U(1) symmetry.
#
#   Version:  1.0
#   Created:  2024-12-24 16:26
#    Author:  Zhiyuan Yao, zhiyuan.yao@icloud.com
# Institute:  Lanzhou Center for Theoretical Physics, Lanzhou University
#==============================================================================#
using Random, LinearAlgebra, SparseArrays
using rEE

#------------------------------------------------------------------------------
# system parameters
#------------------------------------------------------------------------------
const S =  1/2
const L = 12
const Mz = -2

#------------------------------------------------------------------------------
# create product basis maps with given total magnetization Mz
#------------------------------------------------------------------------------
function get_state_space(S::Real, L::Int, Mz::Real=0) 
    # first check argument S and Mz
    if abs(round(Int, 2*S) - 2*S) > 1E-10 || abs(round(Int, 2*Mz) - 2*Mz) > 1E-10
        error("create_space(): argument S and Mz must be half-integer or integer")
    end
    base = round(Int, 2*S + 1)
    state_istate = Int[]
    for state in 0:(base^L-1)
        bits = digits(state, base=base, pad=L) |> reverse
        # | m1 m2 ⋯ mL⟩ = bits .- S  = bits .- (base-1)/2
        if (2*sum(bits) - L*(base-1)) == round(Int, 2*Mz)
            push!(state_istate, state)
        end
    end
    stateN = length(state_istate)
    return stateN, state_istate
end
stateN, state_istate = get_state_space(S, L, Mz)


# create a random state psi and then calculate the entanglement entropy in various ways
Random.seed!(1);
const psi = normalize(rand(stateN))

#------------------------------------------------------------------------------
# The direct way: get |ψ⟩ in tensor product basis and then calculate Sv
#------------------------------------------------------------------------------
function get_entEntropy_direct(psi::AbstractVector{T}) where T <: Union{Float64, ComplexF64}
    base = round(Int, 2*S + 1)
    PSI = spzeros(T, base^L)  # 1-index instead of zero 
    for istate in 1:stateN
        state = state_istate[istate]
        PSI[state+1] = psi[istate]
    end
    rho = get_rho(PSI)
    Sv = get_entEntropy(rho)
    return Sv
end

#------------------------------------------------------------------------------
# further improvement making use of the block diagonal structure of rho
#------------------------------------------------------------------------------
function get_entEntropy_blocking(psi::AbstractVector{T}) where T <: Union{Float64, ComplexF64}
    rhoList, MList, ij_igroup_istate = get_rhos_Ms_ijkmap(state_istate, Float64; S=S, L=L)
    Sv = get_entEntropy!(psi, rhoList, MList, ij_igroup_istate)
    return Sv
end

println("     get_entEntropy_direct(): ", get_entEntropy_direct(psi))
println("get_entEntropy_blocking(): ", get_entEntropy_blocking(psi))
