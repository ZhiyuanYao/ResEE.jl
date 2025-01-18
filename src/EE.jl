#==============================================================================#
# Description
# -----------
#   Standard routines for computing the reduced density matrix `rho` and
#   entanglement entropy `Sv` in the full space of (2S+1)^L. 
#
#   Version:  0.1
#   Created:  2024-12-24 10:52
#    Author:  Zhiyuan Yao, zhiyuan.yao@icloud.com
# Institute:  Lanzhou Center for Theoretical Physics, Lanzhou University
#==============================================================================#
"""
    get_rho(psi::AbstractVector{T}; S::Real=1/2, LA::Int=0)

return the reduced density matrix ρ with subsystem size `LA` default to L÷2 if unspecified. Here we
are considering the case of non-restricted Hilbert space, i.e. dimension of `psi` is base^L.

For the case of restricted Hilbert space, such as Mz sector of spin system of PXP model, current naive
method can be quite slow. In this case, use the alternative method.

# Example

```julia-reply
julia> rho = get_rho(psi; S=1)
julia> rho = get_rho(psi; LA=4)
julia> rho = get_rho(psi; S=1/2, LA=4)
```
"""
function get_rho(psi::AbstractVector{T}; S::Real=1/2, LA::Int=0) where T <: Union{Float64, ComplexF64}
    @assert S > 0 & isinteger(2*S+1) 
    base = round(Int, 2*S+1)
    # make sure the wavefunction is normalized
    if abs(norm(psi) - 1.0) > 1E-10
        println("get_rho(): argument psi has wrong size: ", length(psi))
        exit()
    end
    #--------------------------------------------------------------------------
    # from the dimension of the psi infer the system size L
    #--------------------------------------------------------------------------
    L = round(Int, log(length(psi))/log(base))
    if Int(base^L) != length(psi)
        println("base =$base, L = $L, length = $length(psi)")
        println("get_rho(): argument psi has wrong size: ", length(psi))
        exit()
    end
    LA = (LA == 0 ? L÷2 : LA)
    LB = L - LA
    dimA = base^LA; dimB = base^LB
    #--------------------------------------------------------------------------
    # |Ψ⟩ = ∑ Ψ_ij |i_A j_B⟩  ==>  ρ_ii' = Ψ_ij ⋅ (Ψ_i'j)*
    #--------------------------------------------------------------------------
    M = transpose(reshape(Vector(psi), dimB, dimA))    # Ψ_ij
    rho = M*adjoint(M)               # ishermitian can fail for rho in this form 
    rho .+= adjoint(rho); rho ./= 2  # make sure symmetric or Hermitian
    return rho
end


"""
    check_rho(rho::T; tol=1E-10)

check whether the density matrix rho is valid or not.
"""
function check_rho(rho::T; tol=1E-10) where T <: Union{Matrix{Float64}, Matrix{ComplexF64}, SparseMatrixCSC{Float64, Int64}, SparseMatrixCSC{ComplexF64, Int64}}
    dimA = size(rho, 1)
    if dimA != size(rho, 2)
        error("check_rho(): argument size error")
    end
    if eltype(rho) == Float64
        for i in 1:dimA
            for j in 1:i-1
                if abs(rho[i, j] - rho[j, i]) > tol
                    error("check_rho(): real rho not symmetric")
                end
            end
        end
    else
        for i in 1:dimA
            for j in 1:i-1
                if abs(rho[i, j] - conj(rho[j, i])) > tol
                    error("check_rho(): complex rho off-diagonal not Hermitian")
                end
            end
            if abs(imag(rho[i, i])) > tol
                error("check_rho(): complex rho diagonal not real")
            end
        end
    end
end

#-------------------------------------------------------------------------#
# calculate von Neumann entanglement entropy of a given density matrix
#-------------------------------------------------------------------------#
"""
    get_entEntropy(rho::T; tol=1E-10, check::Bool=false)

get the von Neumann entanglement entropy of the density matrix rho.
"""
function get_entEntropy(rho::Matrix{T}; tol=1E-10, check::Bool=true) where T <: Union{Float64, ComplexF64}
    if check
        eltype(rho) == Float64 ? (@assert issymmetric(rho)) : (@assert ishermitian(rho))
    end
    vals = eigvals(rho)
    if vals[1] < -tol || vals[end] > 1.0 + tol || abs(sum(vals) - 1.0) > tol
        println("λ_1 = ", vals[1])
        println("λ_N = ", vals[end])
        println("∑ λ = ", sum(vals))
        error("entropy(): eigenvalues invalid")
    end
    Sv::Float64 = 0.0
    for i in length(vals):-1:1
        vals[i] < 1E-20 ? break : (Sv -= vals[i]*log(vals[i]))
    end
    return Sv
end


"""
    get_entEntropy!(rho::T; tol=1E-10, check::Bool=false)

get entanglement entropy of the density matrix `rho`, which is destroyed in the process.
"""
function get_entEntropy!(rho::Matrix{T}; tol=1E-10, check::Bool=true) where T <: Union{Float64, ComplexF64}
    if check
        eltype(rho) == Float64 ? (@assert issymmetric(rho)) : (@assert ishermitian(rho))
    end
    vals = eigvals!(rho)
    if vals[1] < -tol || vals[end] > 1.0 + tol || abs(sum(vals) - 1.0) > tol
        println("λ_1 = ", vals[1])
        println("λ_N = ", vals[end])
        println("∑ λ = ", sum(vals))
        error("entropy(): eigenvalues invalid")
    end
    Sv::Float64 = 0.0
    for i in length(vals):-1:1
        vals[i] < 1E-20 ? break : (Sv -= vals[i]*log(vals[i]))
    end
    return Sv
end

"""
    get_entEntropy!(rho::Matrix{T}, lambdas::AbstractVector{Float64}; tol=1E-10, check::Bool=true) where T <: Union{Float64, ComplexF64}

get the entanglement entropy Sv and in palce create the entanglement spectrum `lambdas`.

"""
function get_entEntropy!(rho::Matrix{T}, vals::AbstractVector{Float64}; tol=1E-10, check::Bool=true) where T <: Union{Float64, ComplexF64}
    if check
        eltype(rho) == Float64 ? (@assert issymmetric(rho)) : (@assert ishermitian(rho))
    end
    vals .= eigvals!(rho)
    if vals[1] < -tol || vals[end] > 1.0 + tol || abs(sum(vals) - 1.0) > tol
        println("λ_1 = ", vals[1])
        println("λ_N = ", vals[end])
        println("∑ λ = ", sum(vals))
        error("entropy(): eigenvalues invalid")
    end
    Sv::Float64 = 0.0
    for i in length(vals):-1:1
        vals[i] < 1E-20 ? break : (Sv -= vals[i]*log(vals[i]))
    end
    return Sv
end
