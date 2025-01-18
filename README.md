# ResEE

This package provides efficient algorithms for computing the reduced density matrix
and the von Neumann entanglement entropy of systems with restricted Hilbert spaces
or a global U(1) symmetry.

The package name `ResEE` stands for [res]tricted/symmetry-[res]olved
[e]ntanglement [e]ntropy.

## Installation

```julia-repl
julia> ] add ResEE

julia> using ResEE

```

## Restricted Hilbert Space 
The following code demonstrates how to use the restricted Hilbert space method to
calculate the entanglement entropy of the kinetically restricted PXP model. In the
PXP model, each site can be in the ground (n=0) state or the excited (n=1) state,
but no neighboring sites can be both in the excited states.

```julia
using Random, ResEE

const L  = 32  # system size
const LA = 16  # size of the subsystem A

# create the state space
function get_state_space(L::Int)
    nnList = [2^(n-1) +  2^(mod(n, L)) for n in 1:L]
    state_istate = Int[]
    for state in 0:(2^L-1)
        is_valid = true
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

# create map table and preallocate memory for rho and M 
rho, M, ij_istate = get_rho_M_ijmap(state_istate, eltype(psi); S=1/2, L=L, LA=LA);

# calculate the entanglement entropy using the restricted Hilbert space method
get_rho!(rho, psi, M, ij_istate)
Sv = get_entEntropy!(rho)
```

## U(1) Symmetry
To illustrate the usage of the U(1) symmetry method, we consider the spin-S chain
with total magnetization conservation.

```julia
using Random, ResEE

const S  = 1   # spin
const L  = 12  # system size
const LA = 6   # size of the subsystem A
const Mz = 0   # total magnetization

function get_state_space(S::Real, L::Int, Mz::Real=0) 
    base, Mz2 = round(Int, 2*S + 1), round(Int, 2*Mz)
    state_istate = Int[]
    for state in 0:(base^L-1)
        bits = digits(state, base=base, pad=L)
        # | m1 m2 ⋯ mL⟩ = bits .- S  = bits .- (base-1)/2
        if (2*sum(bits) - L*(base-1)) == Mz2
            push!(state_istate, state)
        end
    end
    stateN = length(state_istate)
    return stateN, state_istate
end

stateN, state_istate = get_state_space(S, L, Mz)

# create a wavefunction psi with total magnetization Mz for demostration
psi = normalize(rand(stateN))

# create the one-time map table
rhos, Ms, ij_igroup_istate = get_rhos_Ms_ijkmap(state_istate, eltype(psi); S=S, L=L, LA=LA)

# use U(1) blocking method to calculate entanglement entropy Sv
Sv = get_entEntropy!(psi, rhos, Ms, igroup_ij_istate)
```

## References
For a complete description of the package, please refer to the following paper:
