"""
$(TYPEDEF)

Epigraph of Euclidean norm (AKA second-order) cone of dimension `dim`.

    $(FUNCTIONNAME){T}(dim::Int)
"""
mutable struct EpiNormEucl{T <: Real} <: Cone{T}
    dim::Int

    point::Vector{T}
    dual_point::Vector{T}
    nt_point::Vector{T}
    nt_point_sqrt::Vector{T}
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
    inv_hess::Symmetric{T, Matrix{T}}
    scal_hess::Symmetric{T, Matrix{T}}
    inv_scal_hess::Symmetric{T, Matrix{T}}

    dist::T
    dual_dist::T
    rt_dist_ratio::T
    rt_rt_dist_ratio::T

    function EpiNormEucl{T}(dim::Int) where {T <: Real}
        @assert dim >= 2
        cone = new{T}()
        cone.dim = dim
        cone.nt_point = zeros(T, dim)
        cone.nt_point_sqrt = zeros(T, dim)
        return cone
    end
end

use_dual_barrier(::EpiNormEucl) = false

use_scal(::EpiNormEucl) = true

reset_data(cone::EpiNormEucl) = (cone.feas_updated = cone.dual_feas_updated =
    cone.grad_updated = cone.dual_grad_updated = cone.hess_updated =
    cone.scal_hess_updated =  cone.inv_scal_hess_updated =
    cone.inv_hess_updated = cone.nt_updated = false)

use_sqrt_hess_oracles(::Int, cone::EpiNormEucl{T}) where {T <: Real} = true
use_sqrt_scal_hess_oracles(::Int, cone::EpiNormEucl{T}) where {T <: Real} = true

get_nu(cone::EpiNormEucl) = 2

function set_initial_point!(
    arr::AbstractVector,
    cone::EpiNormEucl{T},
    ) where {T <: Real}
    arr .= 0
    arr[1] = sqrt(T(get_nu(cone)))
    return arr
end

# TODO refac with dual feas check
function update_feas(cone::EpiNormEucl{T}) where T
    @assert !cone.feas_updated
    u = cone.point[1]

    if u > eps(T)
        @views w = cone.point[2:end]
        cone.dist = (abs2(u) - sum(abs2, w)) / 2
        cone.is_feas = (cone.dist > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::EpiNormEucl{T}) where T
    u = cone.dual_point[1]

    if u > eps(T)
        w = view(cone.dual_point, 2:cone.dim)
        @views cone.dual_dist = (abs2(u) - sum(abs2, w)) / 2
        cone.dual_feas_updated = true
        return (cone.dual_dist > eps(T))
    end

    return false
end

function update_grad(cone::EpiNormEucl)
    @assert cone.is_feas

    @. cone.grad = cone.point / cone.dist
    cone.grad[1] *= -1

    cone.grad_updated = true
    return cone.grad
end

function update_dual_grad(cone::EpiNormEucl)
    @assert cone.dual_feas_updated

    @. cone.dual_grad = cone.dual_point / cone.dual_dist
    cone.dual_grad[1] *= -1

    cone.dual_grad_updated = true
    return cone.dual_grad
end

function update_hess(cone::EpiNormEucl)
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)

    mul!(cone.hess.data, cone.grad, cone.grad')
    inv_dist = inv(cone.dist)
    @inbounds for j in eachindex(cone.grad)
        cone.hess[j, j] += inv_dist
    end
    cone.hess[1, 1] -= inv_dist + inv_dist

    cone.hess_updated = true
    return cone.hess
end

function update_nt(cone::EpiNormEucl)
    @assert cone.is_feas
    @assert cone.dual_feas_updated
    nt_point = cone.nt_point
    nt_point_sqrt = cone.nt_point_sqrt

    normalized_point = cone.point * 1 / sqrt(cone.dist * 2)
    normalized_dual_point = cone.dual_point / sqrt(cone.dual_dist * 2)
    gamma = sqrt((1 + dot(normalized_point, normalized_dual_point)) / 2)

    nt_point[1] = normalized_point[1] + normalized_dual_point[1]
    @. @views nt_point[2:end] = normalized_point[2:end] - normalized_dual_point[2:end]
    nt_point ./= 2 * gamma

    copyto!(nt_point_sqrt, nt_point)
    nt_point_sqrt[1] += 1
    nt_point_sqrt ./= sqrt(2 * nt_point_sqrt[1])

    cone.rt_dist_ratio = sqrt(cone.dist / cone.dual_dist)
    cone.rt_rt_dist_ratio = sqrt(cone.rt_dist_ratio)

    cone.nt_updated = true

    return
end

function update_scal_hess(cone::EpiNormEucl)
    @assert cone.grad_updated
    @assert cone.is_feas
    isdefined(cone, :scal_hess) || alloc_scal_hess!(cone)
    cone.nt_updated || update_nt(cone)

    mul!(cone.scal_hess.data, cone.nt_point, cone.nt_point', 2, false)
    cone.scal_hess.data[:, 1] *= -1
    cone.scal_hess += I # TODO
    cone.scal_hess.data[1, :] *= -1
    cone.scal_hess.data ./= cone.rt_dist_ratio

    cone.scal_hess_updated = true
    return cone.scal_hess
end

function update_inv_hess(cone::EpiNormEucl)
    @assert cone.is_feas
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)

    mul!(cone.inv_hess.data, cone.point, cone.point')
    @inbounds for j in eachindex(cone.grad)
        cone.inv_hess[j, j] += cone.dist
    end
    cone.inv_hess[1, 1] -= cone.dist + cone.dist

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function update_inv_scal_hess(cone::EpiNormEucl)
    @assert cone.grad_updated
    @assert cone.is_feas
    isdefined(cone, :inv_scal_hess) || alloc_inv_scal_hess!(cone)
    cone.nt_updated || update_nt(cone)

    mul!(cone.inv_scal_hess.data, cone.nt_point, cone.nt_point', 2, false)
    @inbounds cone.inv_scal_hess.data[1, 1] -= 1
    @inbounds for j in 2:cone.dim
        cone.inv_scal_hess.data[j, j] += 1
    end
    cone.inv_scal_hess.data .*= cone.rt_dist_ratio

    cone.inv_scal_hess_updated = true
    return cone.inv_scal_hess
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiNormEucl,
    )
    @assert cone.is_feas
    u = cone.point[1]
    w = @view cone.point[2:end]

    @inbounds for j in 1:size(prod, 2)
        uj = arr[1, j]
        wj = @view arr[2:end, j]
        ga = (dot(w, wj) - u * uj) / cone.dist
        prod[1, j] = -ga * u - uj
        @. @views prod[2:end, j] = ga * w + wj
    end
    @. prod ./= cone.dist

    return prod
end

function scal_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiNormEucl{T},
    ::Bool = false,
    ) where {T <: Real}
    cone.nt_updated || update_nt(cone)
    hyperbolic_householder(prod, arr, cone.nt_point, cone.rt_dist_ratio, true)
    return prod
end

function inv_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiNormEucl,
    )
    @assert cone.is_feas

    @inbounds for j in 1:size(prod, 2)
        @views pa = dot(cone.point, arr[:, j])
        @. @views prod[:, j] = pa * cone.point
    end
    @. @views prod[1, :] -= cone.dist * arr[1, :]
    @. @views prod[2:end, :] += cone.dist * arr[2:end, :]

    return prod
end

function inv_scal_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiNormEucl{T},
    ) where {T <: Real}
    cone.nt_updated || update_nt(cone)
    hyperbolic_householder(prod, arr, cone.nt_point, cone.rt_dist_ratio, false)
    return prod
end

function sqrt_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiNormEucl{T},
    ) where {T <: Real}
    @assert cone.is_feas
    u = cone.point[1]
    w = @view cone.point[2:end]

    rt2 = sqrt(T(2))
    distrt2 = cone.dist * rt2
    rtdist = sqrt(cone.dist)
    urtdist = u + rtdist * rt2
    @inbounds for j in 1:size(arr, 2)
        uj = arr[1, j]
        @views wj = arr[2:end, j]
        dotwwj = dot(w, wj)
        prod[1, j] = (u * uj - dotwwj) / distrt2
        wmulj = (dotwwj / urtdist - uj) / distrt2
        @. @views prod[2:end, j] = w * wmulj + wj / rtdist
    end

    return prod
end

function inv_sqrt_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiNormEucl{T},
    ) where {T <: Real}
    @assert cone.is_feas
    u = cone.point[1]
    w = @view cone.point[2:end]

    rt2 = sqrt(T(2))
    rtdist = sqrt(cone.dist)
    urtdist = u + rtdist * rt2
    @inbounds for j in 1:size(arr, 2)
        uj = arr[1, j]
        @views wj = arr[2:end, j]
        dotwwj = dot(w, wj)
        prod[1, j] = (u * uj + dotwwj) / rt2
        wmulj = (dotwwj / urtdist + uj) / rt2
        @. @views prod[2:end, j] = w * wmulj + wj * rtdist
    end

    return prod
end

function sqrt_scal_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiNormEucl{T},
    ) where {T <: Real}
    cone.nt_updated || update_nt(cone)
    hyperbolic_householder(prod, arr, cone.nt_point_sqrt, cone.rt_rt_dist_ratio, true)
    return prod
end

function inv_sqrt_scal_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiNormEucl{T},
    ) where {T <: Real}
    error()
end

function dder3(cone::EpiNormEucl, dir::AbstractVector)
    @assert cone.grad_updated
    dim = cone.dim
    dder3 = cone.dder3
    point = cone.point
    u = point[1]
    u_dir = dir[1]
    @views w = point[2:end]
    @views w_dir = dir[2:end]

    jdotpd = u * u_dir - dot(w, w_dir)
    hess_prod!(dder3, dir, cone)
    dotdHd = -dot(dir, dder3)
    dotpHd = dot(point, dder3)
    dder3 .*= jdotpd
    @. @views dder3[2:end] += dotdHd * w + dotpHd * w_dir
    dder3[1] += -dotdHd * u - dotpHd * u_dir
    dder3 ./= 2 * cone.dist

    return dder3
end

function dder3(
    cone::EpiNormEucl{T},
    pdir::AbstractVector{T},
    ddir::AbstractVector{T},
    ) where {T <: Real}
    @assert cone.feas_updated
    @assert cone.dual_feas_updated
    dder3 = cone.dder3
    point = cone.point

    @views jdot_p_s = point[1] * pdir[1] - dot(point[2:end], pdir[2:end])
    @. dder3 = jdot_p_s * ddir
    dot_s_z = dot(pdir, ddir)
    dot_p_z = dot(point, ddir)
    dder3[1] += dot_s_z * point[1] - dot_p_z * pdir[1]
    @. @views dder3[2:end] += -dot_s_z * point[2:end] + dot_p_z * pdir[2:end]
    dder3 ./= -cone.dist * 2

    return dder3
end

function hyperbolic_householder(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    v::AbstractVector{T},
    fact::T,
    use_inv::Bool,
    ) where {T <: Real}
    if use_inv
        v[2:end] .*= -1
    end
    for j in 1:size(prod, 2)
        @views pa = 2 * dot(v, arr[:, j])
        @. @views prod[:, j] = pa * v
    end
    @. prod[1, :] -= arr[1, :]
    @. prod[2:end, :] += arr[2:end, :]
    if use_inv
        prod ./= fact
        v[2:end] .*= -1
    else
        prod .*= fact
    end
    return prod
end
