#==============================================================================#
# Description
# -----------
#   Efficient algorithms for computing the reduced density matrix `rho` and
#   entanglement entropy `Sv` in systems with a restricted Hilbert space or
#   U(1) symmetry.
#
#   Version:  0.1
#   Created:  2024-12-24 10:54
#    Author:  Zhiyuan Yao, zhiyuan.yao@icloud.com
# Institute:  Lanzhou Center for Theoretical Physics, Lanzhou University
#==============================================================================#
"""
    get_istate_state(state_istate::Vector{Int}; memoryCap=2.2E9)

return the normal vector if `state=0` is not contained in `state_istate` or OffsetVector
`istate_state` otherwise (memory usage < memoryCap here). In case memory usage of above 
normal vector exceeds memoryCap (2.2GB = 2^{28+3}), then return dictionary.

# Examples
```julia-repl
julia> istate_state = get_istate_state(state_istate)
julia> istate_state = get_istate_state(state_istate; memoryCap=100*1E6)  # set memory cap as 100 MB
```
"""
function get_istate_state(state_istate::Vector{Int}; memoryCap=2.2E9)
    @assert minimum(state_istate) >= 0
    state_Max = maximum(state_istate)
    
    # if memory usage above memoryCap, then use dict 
    if 8*state_Max >= memoryCap
        return Dict(zip(state_istate, 1:length(state_istate))) 
    end

    if minimum(state_istate) == 0
        istate_state = OffsetVector(fill(typemin(0), state_Max+1), 0:state_Max)
    else
        istate_state = fill(typemin(0), state_Max)
    end
    for (istate, state) in enumerate(state_istate)
        istate_state[state] = istate 
    end
    return istate_state
end


"""
get_stateA_istateA(state_istate::Vector{Int}; S::Real, L::Int, LA::Int=0)

return the sorted `stateA_istateA` lists of the restricted Hilbert space `state_istate`. In this
way, only valid `stateA` tabled from `stateA_istateA` need to be considered when calculating `ρ`.

# Example
```julia-repl
julia> stateA_istateA = get_stateA_istateA(state_istate; S=1/2, L=L)
```
"""
function get_stateA_istateA(state_istate::Vector{Int}; S::Real, L::Int, LA::Int=0)
    @assert S > 0 && isinteger(2*S)
    base = round(Int, 2*S+1)
    LA = (LA == 0) ? L÷2 : LA   # if LA not present set LA = L÷2; () for clarity
    # [ stateA ] [ stateB ] :  state = stateA*weight + stateB
    weight = base^(L-LA)

    stateA_istateA = Int[]
    marked_stateA = falses(base^LA)
    for state in state_istate
        stateA = div(state, weight)
        if !marked_stateA[stateA+1]
            push!(stateA_istateA, stateA)
            marked_stateA[stateA+1] = true
        end
    end
    return sort(stateA_istateA)
end

"""
get_stateB_istateB(state_istate::Vector{Int}; S::Real, L::Int, LB::Int=0)

return the sorted `stateB_istateB` lists of the restricted Hilbert space `state_istate`.
In this way, only valid `stateB` tabled from `stateB_istateB` need to be considered in 
the calculation `Tr_B |stateA stateB⟩ ⟨stateA' stateB|`.

# Example
```julia-repl
julia> stateB_istateB = get_stateB_istateB(state_istate; S=1/2, L=L)
```
"""
function get_stateB_istateB(state_istate::Vector{Int}; S::Real, L::Int, LB::Int=0)
    @assert S > 0 && isinteger(2*S)
    base = round(Int, 2*S+1)
    LB = (LB == 0) ? (L-L÷2) : LB   # if LA not present set LA = L÷2; () for clarity
    weight = base^LB
    # [ stateA ] [ stateB ] :  state = stateA*weight + stateB

    stateB_istateB = Int[]
    marked_stateB = falses(base^LB)
    for state in state_istate
        stateB = rem(state, weight)
        if !marked_stateB[stateB+1]
            push!(stateB_istateB, stateB)
            marked_stateB[stateB+1] = true
        end
    end
    return sort(stateB_istateB)
end


"""
    get_stateAB_istateAB(state_istate::Vector{Int}; S::Real, L::Int, LA::Int=0)

return `stateA_istateA` and `stateB_istateB` lists of the restricted Hilbert space
`state_istate`. In this way, only valid `stateA` tabled from `stateA_istateA` need 
to be considered when calculating `ρ`.

Note the returned `stateA_istateA` and `stateB_istateB` are sorted.

# Example
```julia-repl
julia> stateA_istateA, stateB_istateB = get_stateAB_istateAB(state_istate; S=1/2, L=L)
```
"""
function get_stateAB_istateAB(state_istate::Vector{Int}; S::Real, L::Int, LA::Int=0)
    @assert S > 0 && isinteger(2*S)
    base = round(Int, 2*S+1)
    LA = (LA == 0) ? L÷2 : LA   # if LA not present set LA = L÷2; () for clarity
    LB = L - LA
    # [ stateA ] [ stateB ] :  state = stateA*weight + stateB
    weight = base^LB

    stateA_istateA, stateB_istateB = Int[], Int[]
    marked_stateA, marked_stateB = falses(base^LA), falses(base^LB)
    for state in state_istate
        stateA, stateB = divrem(state, weight)
        if !marked_stateA[stateA+1]
            push!(stateA_istateA, stateA)
            marked_stateA[stateA+1] = true
        end
        if !marked_stateB[stateB+1]
            push!(stateB_istateB, stateB)
            marked_stateB[stateB+1] = true
        end
    end
    return sort(stateA_istateA), sort(stateB_istateB)
end


"""
    get_rho_M_ijmap(state_istate::Vector{Int}; S::Real, L::Int, LA::Int=0)

return the intermediate matrix index (i, j) mapping for constrainted system.

# Example
```juila-repl
julia> rho, M, ij_istate = get_rho_M_ijmap(state_istate, Float64; S=S, L=L)
```
"""
function get_rho_M_ijmap(state_istate::Vector{Int}, T::Type=ComplexF64; S::Real, L::Int, LA::Int=0)
    @assert S > 0 && isinteger(2*S)
    LA = (LA == 0) ? L÷2 : LA
    # [ stateA ] [ stateB ] :  state = stateA*weight + stateB
    base = round(Int, 2*S+1); weight = base^(L-LA)

    stateA_istate = Int[]; stateB_istate = Int[]
    for state in state_istate
        stateA, stateB = divrem(state, weight)
        push!(stateA_istate, stateA)
        push!(stateB_istate, stateB)
    end

    # sorting is unnecessary but indexing ordered list should have performance benefits 
    stateA_istateA = sort(unique(stateA_istate))
    stateB_istateB = sort(unique(stateB_istate))
    istateA_stateA = get_istate_state(stateA_istateA)
    istateB_stateB = get_istate_state(stateB_istateB)

    stateN = length(state_istate)
    ij_istate = Vector{Tuple{Int32, Int32}}(undef, stateN) 
    for istate in 1:stateN
        stateA, stateB = stateA_istate[istate], stateB_istate[istate]
        ij_istate[istate] = (istateA_stateA[stateA], istateB_stateB[stateB])
    end

    dimA, dimB = length(stateA_istateA), length(stateB_istateB)
    println("Matrix M sparsity =", round(dimA*dimB/stateN))
    rho = zeros(T, dimA, dimA); M = zeros(T, dimA, dimB)
    return rho, M, ij_istate
end


"""
    get_rho(psi::AbstractVector{T}, M::Union{Matrix{T}, SparseMatrixCSC{T, Int64}}, ij_istate::Vector{Tuple{Int32, Int32}})

return the density matrix rho using the restricted reshape method.

# Example
```juila-repl
julia> rho, res... = get_rho_M_ijmap(state_istate, Float64; S=S, L=L); 
julia> rho = get_rho(psi, res...) 
```
"""
function get_rho(psi::AbstractVector{T}, M::Union{Matrix{T}, SparseMatrixCSC{T, Int64}}, ij_istate::Vector{Tuple{Int32, Int32}}) where T <: Union{Float64, ComplexF64}
    @assert firstindex(psi) == 1
    M .= 0 
    for istate in (issparse(psi) ? psi.nzind : eachindex(psi))
        i, j = ij_istate[istate]
        M[i, j] = psi[istate]
    end
    return M*M' 
end

"""
    get_rho!(rho::AbstractMatrix{T}, psi::AbstractVector{T}, M::Union{Matrix{T}, SparseMatrixCSC{T, Int64}}, ij_istate::Vector{Tuple{Int64, Int64}})

# Parameters
- `rho`: dimension dimA × dimA  matrix 
- `M`  : dimension dimA × dimB  matrix (for local constraint, M will NOT be sparse, and regular
         Matrix type would be better)
- `ij_istate` :: istate to intermediate matrix M index (i, j) mapping table 

# Example
```juila-repl
julia> rho, res... = get_rho_M_ijmap(state_istate, Float64; S=S, L=L); 
julia> get_rho!(rho, psi, res...)
```
"""
function get_rho!(rho::AbstractMatrix{T}, psi::AbstractVector{T}, M::Union{Matrix{T}, SparseMatrixCSC{T, Int64}}, ij_istate::Vector{Tuple{Int32, Int32}}) where T <: Union{Float64, ComplexF64}
    @assert firstindex(psi) == 1
    M .= 0 
    for istate in (issparse(psi) ? psi.nzind : eachindex(psi))
    # for (istate, c) in enumerate(psi) 
        i, j = ij_istate[istate]
        M[i, j] = psi[istate]
    end
    # 
    mul!(rho, M, M')  # rho .= M*M' creates temporary array (note @. rho = M*M' is wrong !!!)
end


#------------------------------------------------------------------------------
# create ij_istate mapping for each bitsumA group so that we can latter
# diagonalize each matrix block, i.e. of given gropu, in ρ to get eigenvalues
#------------------------------------------------------------------------------
"""
    get_rhos_Ms_ijkmap(state_istate::Vector{Int}, T::Type=ComplexF64; S::Real, L::Int, LA::Int=0)

return the block restricted reshape map `istate -> (i, j, iblock)`, block density matrix
`rho` and auxiliary block `M` using U1 convervation.

# Example
```juila-repl
julia> rhoList, MList, ij_igroup_istate = get_rhos_Ms_ijkmap(state_istate, Float64; S=S, L=L)
```
"""
function get_rhos_Ms_ijkmap(state_istate::Vector{Int}, T::Type=ComplexF64; S::Real, L::Int, LA::Int=0)
    @assert S > 0 && isinteger(2*S)
    LA = (LA == 0) ? L÷2 : LA
    # [ stateA ] [ stateB ] :  state = stateA*weight + stateB
    base = round(Int, 2*S+1); weight = base^(L-LA)

    # define a unility function here
    function bitsum_num(n::Int)::Int
        if base == 2
            return count_ones(n)
        else
            sum = 0
            while n > 0
                n, bit = divrem(n, base)
                sum += bit
            end
            return sum
        end
    end

    # bitsumA_List[igroup] give the bitsumA of igroup-th group, similarly for bitsumB_List[igroup]
    # stateA_istateA_group[igroup] gives all possible stateA of given group
    stateA_istateA_group = Vector{Int}[]
    stateB_istateB_group = Vector{Int}[]
    marked_bitsumA = falses((base-1)*LA+1)
    igroup_bitsumA = Dict{Int, Int}()

    # lightweight U(1) symmetry checking for robustness, and can be comment out 
    bitsumA_List = Int[]; bitsumB_List = Int[]   

    igroup_istate = Int[]; stateA_istate = Int[]; stateB_istate = Int[]
    counter = 0                    # igroup counter of bitsumA groups
    for state in state_istate
        stateA, stateB = divrem(state, weight)
        push!(stateA_istate, stateA)
        push!(stateB_istate, stateB)
        bitsumA = bitsum_num(stateA)
        bitsumB = bitsum_num(stateB)
        #----------------------------------------------------------------------
        # igroup_bitsumA :    1      2          igroup for each bitsumA group
        #----------------------------------------------------------------------
        if !marked_bitsumA[bitsumA+1]           # if encounter a new bitsumA group
            counter += 1                        # loop invariant, counter groups
            igroup_bitsumA[bitsumA] = counter   # update the dict
            marked_bitsumA[bitsumA+1] = true
            push!(igroup_istate, counter)
            push!(stateA_istateA_group, [stateA])
            push!(stateB_istateB_group, [stateB])
            push!(bitsumA_List, bitsumA)
            push!(bitsumB_List, bitsumB)
        else
            igroup = igroup_bitsumA[bitsumA]
            @assert bitsumB_List[igroup] == bitsumB      # make sure in a U(1) sector
            push!(igroup_istate, igroup)
            push!(stateA_istateA_group[igroup], stateA)  # duplicity to be removed later
            push!(stateB_istateB_group[igroup], stateB)  # duplicity to be removed later
        end
    end

    groupN = length(igroup_bitsumA)

    istateA_stateA_group = Dict{Int, Int}[]
    istateB_stateB_group = Dict{Int, Int}[]
    rhoList, MList = Matrix{T}[], Matrix{T}[]

    for igroup in 1:groupN
        stateA_istateA = stateA_istateA_group[igroup]
        stateB_istateB = stateB_istateB_group[igroup]
        # sorting is not necessary, but may improve performance
        sort!(unique!(stateA_istateA)); sort!(unique!(stateB_istateB))

        dA, dB = length(stateA_istateA), length(stateB_istateB)
        push!(MList, zeros(dA, dB))
        push!(rhoList, zeros(dA, dA))

        push!(istateA_stateA_group, Dict(zip(stateA_istateA, 1:dA)))
        push!(istateB_stateB_group, Dict(zip(stateB_istateB, 1:dB)))
    end

    stateN = length(state_istate)
    
    if groupN > typemax(UInt8(0))
        error("get_rhos_Ms_ijkmap(): groupN = $(groupN) too big for type UInt8")
    end
    ij_igroup_istate = Vector{Tuple{Int32, Int32, UInt8}}(undef, stateN) 

    for istate in 1:stateN
        igroup, stateA, stateB = igroup_istate[istate], stateA_istate[istate], stateB_istate[istate]
        istateA_stateA = istateA_stateA_group[igroup]
        istateB_stateB = istateB_stateB_group[igroup]
        ij_igroup_istate[istate] = (istateA_stateA[stateA], istateB_stateB[stateB], igroup)
    end

    return rhoList, MList, ij_igroup_istate
end

#------------------------------------------------------------------------------
# While rho can be sparse, each rho block is not sparse hence use Matrix{T}
#------------------------------------------------------------------------------
"""
    get_rho(psi::AbstractVector{T}, MList::Vector{Matrix{T}}, ij_igroup_istate::Vector{Tuple{Int32, Int32, UInt8}})

in place create the block diagonal sparse reduced density matrix rho leveraging the U1 conservation.

# Example
```juila-repl
julia> _, res... = get_rhos_Ms_ijkmap(state_istate, Float64; S=S, L=L)
julia> rho = get_rho(psi, res...) 
julia>
julia> rhoList, MList, ij_igroup_istate = get_rhos_Ms_ijkmap(state_istate, Float64; S=S, L=L)
julia> rho = get_rho(psi, MList, ij_igroup_istate) 
```
"""
function get_rho(psi::AbstractVector{T}, MList::Vector{Matrix{T}}, ij_igroup_istate::Vector{Tuple{Int32, Int32, UInt8}}) where T <: Union{Float64, ComplexF64}
    @assert firstindex(psi) == 1
    [M .= 0 for M in MList]
    for istate in (issparse(psi) ? psi.nzind : eachindex(psi))
        i, j, igroup = ij_igroup_istate[istate]
        MList[igroup][i, j] = psi[istate]
    end
    
    dimA = sum(size(M, 1) for M in MList); rho = spzeros(T, dimA, dimA)
    # construct the block diagonal reudced density matrix rho
    istart = 1
    for igroup in eachindex(MList) 
        M = MList[igroup]
        iend = istart + size(rho, 1) - 1
        mul!(rho[istart:iend, istart:iend], M, M')
        istart = iend + 1
    end
    return rho
end


"""
    get_entEntropy!(psi::AbstractVector{T}, rhoList::Vector{Matrix{T}}, MList::Vector{Matrix{T}}, ij_igroup_istate::Vector{Tuple{Int32, Int32, UInt8}})

get the entanglement entropy of istate wavefunction `psi` using the block restricted reshape
method for U1 conserved system.

# Example
```juila-repl
julia> res... = get_rhos_Ms_ijkmap(state_istate, Float64; S=S, L=L)
julia> entEntropy = get_entEntropy!(psi, res...) 
julia>
julia> rhoList, MList, ij_igroup_istate = get_rhos_Ms_ijkmap(state_istate, Float64; S=S, L=L)
julia> entEntropy = get_entEntropy!(psi, rhoList, MList, ij_igroup_istate) 
```
"""
function get_entEntropy!(psi::AbstractVector{T}, rhoList::Vector{Matrix{T}}, MList::Vector{Matrix{T}}, ij_igroup_istate::Vector{Tuple{Int32, Int32, UInt8}}) where T <: Union{Float64, ComplexF64}
    @assert firstindex(psi) == 1
    [M .= 0 for M in MList]
    for istate in (issparse(psi) ? psi.nzind : eachindex(psi))
        i, j, igroup = ij_igroup_istate[istate]
        MList[igroup][i, j] = psi[istate]
    end
    groupN = length(rhoList); vals = Float64[]
    for igroup in 1:groupN
        rho, M = rhoList[igroup], MList[igroup]
        mul!(rho, M, M')
        vals = vcat(vals, eigvals!(rho))
    end
    sort!(vals, rev=true); Sv=0.0
    for i in 1:length(vals)
        vals[i] < 1E-20 ? break : Sv -= vals[i]*log(vals[i])
    end
    # Sv = -sum(p*log(p) for p in sort!(vals, rev=true) if p >= 1E-20)
    return Sv
end
