"""
$(TYPEDEF)

Real symmetric or complex Hermitian positive semidefinite cone of dimension
`dim` in svec format.

    $(FUNCTIONNAME){T, R}(dim::Int)
"""
mutable struct PosSemidefTri{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    dim::Int
    side::Int
    is_complex::Bool
    rt2::T

    point::Vector{T}
    dual_point::Vector{T}
    grad::Vector{T}
    dual_grad::Vector{T}
    dder3::Vector{T}
    vec1::Vector{T}
    vec2::Vector{T}
    feas_updated::Bool
    dual_feas_updated::Bool
    grad_updated::Bool
    dual_grad_updated::Bool
    hess_updated::Bool
    scal_hess_updated::Bool
    inv_scal_hess_updated::Bool
    inv_hess_updated::Bool
    is_feas::Bool
    nt_updated::Bool
    hess::Symmetric{T, Matrix{T}}
    scal_hess::Symmetric{T, Matrix{T}}
    inv_scal_hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    mat::Matrix{R}
    dual_mat::Matrix{R}
    mat2::Matrix{R}
    mat3::Matrix{R}
    mat4::Matrix{R}
    inv_mat::Matrix{R}
    inv_dual_mat::Matrix{R}
    scalmat_sqrti::Matrix{R}
    fact_mat::Cholesky{R}
    dual_fact_mat::Cholesky{R}
    nt_svd

    function PosSemidefTri{T, R}(
        dim::Int,
        ) where {T <: Real, R <: RealOrComplex{T}}
        @assert dim >= 1
        cone = new{T, R}()
        cone.dim = dim
        cone.rt2 = sqrt(T(2))
        cone.is_complex = (R <: Complex)
        cone.side = svec_side(R, dim)
        return cone
    end
end

use_dual_barrier(::PosSemidefTri) = false

use_scal(::PosSemidefTri{T, T}) where {T <: Real} = true

reset_data(cone::PosSemidefTri) = (cone.feas_updated = cone.dual_feas_updated =
    cone.grad_updated =
    cone.dual_grad_updated = cone.hess_updated = cone.scal_hess_updated =
    cone.inv_hess_updated = cone.inv_scal_hess_updated = cone.nt_updated = false)

use_sqrt_hess_oracles(::Int, cone::PosSemidefTri) = true
use_sqrt_scal_hess_oracles(::Int, cone::PosSemidefTri) = true

function setup_extra_data!(
    cone::PosSemidefTri{T, R},
    ) where {T <: Real, R <: RealOrComplex{T}}
    cone.dual_grad = zeros(T, cone.dim)
    cone.mat = zeros(R, cone.side, cone.side)
    cone.dual_mat = zero(cone.mat)
    cone.mat2 = zero(cone.mat)
    cone.mat3 = zero(cone.mat)
    cone.mat4 = zero(cone.mat)
    cone.inv_mat = zero(cone.mat)
    cone.inv_dual_mat = zero(cone.mat)
    cone.scalmat_sqrti = zero(cone.mat)
    return cone
end

get_nu(cone::PosSemidefTri) = cone.side

function set_initial_point!(arr::AbstractVector, cone::PosSemidefTri)
    incr = (cone.is_complex ? 2 : 1)
    arr .= 0
    k = 1
    @inbounds for i in 1:cone.side
        arr[k] = 1
        k += incr * i + 1
    end
    return arr
end

function update_feas(cone::PosSemidefTri)
    @assert !cone.feas_updated

    svec_to_smat!(cone.mat, cone.point, cone.rt2)
    copyto!(cone.mat2, cone.mat)
    cone.fact_mat = cholesky!(Hermitian(cone.mat2, :U), check = false)
    cone.is_feas = isposdef(cone.fact_mat)

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::PosSemidefTri)
    svec_to_smat!(cone.dual_mat, cone.dual_point, cone.rt2)
    cone.dual_fact_mat = cholesky!(Hermitian(cone.dual_mat, :U), check = false)
    cone.dual_feas_updated = true
    return isposdef(cone.dual_fact_mat)
end

function update_grad(cone::PosSemidefTri)
    @assert cone.is_feas

    inv_fact!(cone.inv_mat, cone.fact_mat)
    smat_to_svec!(cone.grad, cone.inv_mat, cone.rt2)
    cone.grad .*= -1
    copytri!(cone.mat, 'U', true)

    cone.grad_updated = true
    return cone.grad
end

function update_dual_grad(cone::PosSemidefTri)
    inv_fact!(cone.inv_dual_mat, cone.dual_fact_mat)
    smat_to_svec!(cone.dual_grad, cone.inv_dual_mat, cone.rt2)
    cone.dual_grad .*= -1

    cone.dual_grad_updated = true
    return cone.dual_grad
end

function update_hess(cone::PosSemidefTri)
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)
    copytri!(cone.inv_mat, 'U', true)
    symm_kron!(cone.hess.data, cone.inv_mat, cone.rt2)
    cone.hess_updated = true
    return cone.hess
end

function update_nt(cone::PosSemidefTri)
    (U, lambda, V) = cone.nt_svd = svd(cone.dual_fact_mat.U * cone.fact_mat.L)
    cone.scalmat_sqrti = Diagonal(sqrt.(lambda)) \ (U' * cone.dual_fact_mat.U)
    cone.nt_updated = true
    return
end

function update_scal_hess(cone::PosSemidefTri)
    @assert cone.feas_updated && cone.dual_feas_updated
    isdefined(cone, :scal_hess) || alloc_scal_hess!(cone)
    cone.nt_updated || update_nt(cone)

    (U, lambda, V) = cone.nt_svd
    scalmat_sqrti = Diagonal(sqrt.(lambda)) \ (U' * cone.dual_fact_mat.U)
    symm_kron!(cone.scal_hess.data, Hermitian(scalmat_sqrti' * scalmat_sqrti, :U), cone.rt2)
    cone.scal_hess_updated = true
    return cone.scal_hess
end

function update_inv_hess(cone::PosSemidefTri)
    @assert is_feas(cone)
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
    symm_kron!(cone.inv_hess.data, cone.mat, cone.rt2)
    cone.inv_hess_updated = true
    return cone.inv_hess
end

function update_inv_scal_hess(cone::PosSemidefTri)
    @assert cone.feas_updated && cone.dual_feas_updated
    isdefined(cone, :inv_scal_hess) || alloc_inv_scal_hess!(cone)
    cone.nt_updated || update_nt(cone)

    (U, lambda, V) = cone.nt_svd
    scalmat_sqrt = cone.dual_fact_mat.U \ (U * Diagonal(sqrt.(lambda)))
    symm_kron!(cone.inv_scal_hess.data, Hermitian((scalmat_sqrt * scalmat_sqrt'), :U), cone.rt2)
    cone.inv_scal_hess_updated = true
    return cone.inv_scal_hess
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::PosSemidefTri,
    )
    @assert is_feas(cone)

    @inbounds for i in 1:size(arr, 2)
        svec_to_smat!(cone.mat4, view(arr, :, i), cone.rt2)
        copytri!(cone.mat4, 'U', true)
        rdiv!(cone.mat4, cone.fact_mat)
        ldiv!(cone.fact_mat, cone.mat4)
        smat_to_svec!(view(prod, :, i), cone.mat4, cone.rt2)
    end

    return prod
end

function scal_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::PosSemidefTri{T, R},
    ::Bool = false,
    ) where {T <: Real, R <: RealOrComplex{T}}
    @assert cone.feas_updated && cone.dual_feas_updated
    cone.nt_updated || update_nt(cone)

    (U, lambda, V) = cone.nt_svd
    scalmat_sqrti = cone.scalmat_sqrti
    w = Hermitian(scalmat_sqrti' * scalmat_sqrti, :U)

    @inbounds for i in 1:size(arr, 2)
        svec_to_smat!(cone.mat4, view(arr, :, i), cone.rt2)
        copytri!(cone.mat4, 'U', true)
        temp = w * cone.mat4 * w
        smat_to_svec!(view(prod, :, i), temp, cone.rt2)
    end

    return prod
end

function inv_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::PosSemidefTri,
    )
    @assert is_feas(cone)

    @inbounds for i in 1:size(arr, 2)
        svec_to_smat!(cone.mat4, view(arr, :, i), cone.rt2)
        mul!(cone.mat3, Hermitian(cone.mat4, :U), cone.mat)
        mul!(cone.mat4, Hermitian(cone.mat, :U), cone.mat3)
        smat_to_svec!(view(prod, :, i), cone.mat4, cone.rt2)
    end

    return prod
end

function inv_scal_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::PosSemidefTri{T, R},
    ) where {T <: Real, R <: RealOrComplex{T}}
    @assert cone.feas_updated && cone.dual_feas_updated
    cone.nt_updated || update_nt(cone)

    (U, lambda, V) = cone.nt_svd
    scalmat_sqrt = cone.dual_fact_mat.U \ (U * Diagonal(sqrt.(lambda)))
    w = Hermitian(scalmat_sqrt * scalmat_sqrt', :U)

    @inbounds for i in 1:size(arr, 2)
        svec_to_smat!(cone.mat4, view(arr, :, i), cone.rt2)
        copytri!(cone.mat4, 'U', true)
        temp = w * cone.mat4 * w
        smat_to_svec!(view(prod, :, i), temp, cone.rt2)
    end

    return prod
end

function sqrt_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::PosSemidefTri,
    )
    @assert is_feas(cone)

    @inbounds for i in 1:size(arr, 2)
        svec_to_smat!(cone.mat4, view(arr, :, i), cone.rt2)
        copytri!(cone.mat4, 'U', true)
        rdiv!(cone.mat4, cone.fact_mat.U)
        ldiv!(cone.fact_mat.U', cone.mat4)
        smat_to_svec!(view(prod, :, i), cone.mat4, cone.rt2)
    end

    return prod
end

function inv_sqrt_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::PosSemidefTri,
    )
    @assert is_feas(cone)

    @inbounds for i in 1:size(arr, 2)
        svec_to_smat!(cone.mat4, view(arr, :, i), cone.rt2)
        copytri!(cone.mat4, 'U', true)
        rmul!(cone.mat4, cone.fact_mat.U')
        lmul!(cone.fact_mat.U, cone.mat4)
        smat_to_svec!(view(prod, :, i), cone.mat4, cone.rt2)
    end

    return prod
end

function sqrt_scal_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::PosSemidefTri{T},
    ) where {T <: Real}
    @assert cone.is_feas
    cone.nt_updated || update_nt(cone)
    cone.scalmat_sqrti = cone.scalmat_sqrti

    @inbounds for i in 1:size(arr, 2)
        svec_to_smat!(cone.mat3, view(arr, :, i), cone.rt2)
        mul!(cone.mat4, Hermitian(cone.mat3, :U), cone.scalmat_sqrti')
        mul!(cone.mat3, cone.scalmat_sqrti, cone.mat4)
        smat_to_svec!(view(prod, :, i), cone.mat3, cone.rt2)
    end
    return prod
end

function inv_sqrt_scal_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::PosSemidefTri{T, R},
    ) where {T <: Real, R <: RealOrComplex{T}}
    error()
end

function dder3(cone::PosSemidefTri, dir::AbstractVector)
    @assert cone.grad_updated

    S = copytri!(svec_to_smat!(cone.mat4, dir, cone.rt2), 'U', true)
    ldiv!(cone.fact_mat, S)
    rdiv!(S, cone.fact_mat.U)
    mul!(cone.mat3, S, S') # TODO use outer prod function
    smat_to_svec!(cone.dder3, cone.mat3, cone.rt2)

    return cone.dder3
end

function dder3(
    cone::PosSemidefTri{T},
    pdir::AbstractVector{T},
    ddir::AbstractVector{T},
    ) where {T <: Real}
    @assert cone.grad_updated
    dder3 = cone.dder3
    d = cone.side
    P = Hermitian(svec_to_smat!(zeros(T, d, d), pdir, cone.rt2), :U)
    D = Hermitian(svec_to_smat!(zeros(T, d, d), ddir, cone.rt2), :U)
    Si = Hermitian(cone.inv_mat, :U)
    PD = P * D
    smat_to_svec!(dder3, (Si * PD + PD' * Si) / -2, cone.rt2)
    return dder3
end
