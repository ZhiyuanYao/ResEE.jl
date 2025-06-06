#==============================================================================#
# Description
# -----------
#   This program calculate the bipartite entanglement entropy spectrum of the 
#   PXP model with periodic boundary conditions. 
#
#   Version:  1.0
#   Created:  2024-12-24 11:45
#    Author:  Zhiyuan Yao, zhiyuan.yao@icloud.com
# Institute:  Lanzhou Center for Theoretical Physics, Lanzhou University
#==============================================================================#
using Random, ResEE, LinearAlgebra

const L  = 32  # system size
const LA = 16  # size of the subsystem A

# create the state space
function get_state_space(L::Int)
    nnList = [2^(n-1) +  2^(mod(n, L)) for n in 1:L]
    state_istate = Int[]
    for state in 0:(2^L-1)
        is_valid = true
        # make sure no neighboring sites are both in the excited (n=1) states
        for nn in nnList
            if count_ones(nn & state) > 1
                is_valid = false
                break
            end
        end
        if is_valid
            push!(state_istate, state)
        end
    end
    return length(state_istate), state_istate
end
stateN, state_istate = get_state_space(L)

# create a wavefunction for demo
psi = normalize(rand(stateN))

# create map table and preallocate memory for rho and M with the same element type as psi
rho, M, ij_istate = get_rho_M_ijmap(state_istate, eltype(psi); S=1/2, L=L, LA=LA);

# calculate the entanglement entropy using the restricted Hilbert space method
get_rho!(rho, psi, M, ij_istate)
Sv = get_entEntropy!(rho)
