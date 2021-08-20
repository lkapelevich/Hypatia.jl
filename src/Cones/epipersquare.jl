"""
$(TYPEDEF)

Epigraph of perspective function of halved squared Euclidean norm (AKA rotated
second-order) cone of dimension `dim`.

    $(FUNCTIONNAME){T}(dim::Int)
"""
mutable struct EpiPerSquare{T <: Real} <: Cone{T}
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
    sqrt_hess_prod_updated::Bool
    inv_sqrt_hess_prod_updated::Bool
    is_feas::Bool
    nt_updated::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    scal_hess::Symmetric{T, Matrix{T}}
    inv_scal_hess::Symmetric{T, Matrix{T}}

    dist::T
    dual_dist::T
    rtdist::T
    denom::T
    rt_dist_ratio::T
    rt_rt_dist_ratio::T
    sqrt_hess_vec::Vector{T}
    inv_sqrt_hess_vec::Vector{T}

    function EpiPerSquare{T}(dim::Int) where {T <: Real}
        @assert dim >= 3
        cone = new{T}()
        cone.dim = dim
        cone.dual_grad = zeros(T, dim)
        cone.nt_point = zeros(T, dim)
        cone.nt_point_sqrt = zeros(T, dim)
        return cone
    end
end

use_dual_barrier(::EpiPerSquare) = false

reset_data(cone::EpiPerSquare) = (cone.feas_updated = cone.dual_feas_updated =
    cone.grad_updated = cone.dual_grad_updated = cone.nt_updated =
    cone.hess_updated = cone.inv_hess_updated = cone.sqrt_hess_prod_updated =
    cone.scal_hess_updated = cone.inv_scal_hess_updated =
    cone.inv_sqrt_hess_prod_updated = false)

use_sqrt_hess_oracles(::Int, cone::EpiPerSquare{T}) where {T <: Real} = true
use_sqrt_scal_hess_oracles(::Int, cone::EpiPerSquare{T}, ::T) where {T <: Real} = true

get_nu(cone::EpiPerSquare) = 2

function set_initial_point!(arr::AbstractVector, cone::EpiPerSquare)
    @views arr[1:2] .= 1
    @views arr[3:end] .= 0
    return arr
end

# TODO refac with dual feas check
function update_feas(cone::EpiPerSquare{T}) where T
    @assert !cone.feas_updated
    u = cone.point[1]
    v = cone.point[2]

    if u > eps(T) && v > eps(T)
        @views w = cone.point[3:end]
        cone.dist = u * v - sum(abs2, w) / 2
        cone.is_feas = (cone.dist > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::EpiPerSquare{T}) where T
    u = cone.dual_point[1]
    v = cone.dual_point[2]

    if u > eps(T) && v > eps(T)
        @views w = cone.dual_point[3:end]
        cone.dual_dist = u * v - sum(abs2, w) / 2
        cone.dual_feas_updated = true
        return (cone.dual_dist > eps(T))
    end

    return false
end

function update_grad(cone::EpiPerSquare)
    @assert cone.is_feas

    @. cone.grad = cone.point / cone.dist
    g2 = cone.grad[2]
    cone.grad[2] = -cone.grad[1]
    cone.grad[1] = -g2

    cone.grad_updated = true
    return cone.grad
end

function update_dual_grad(cone::EpiPerSquare)
    @assert cone.dual_feas_updated

    @. cone.dual_grad = cone.dual_point / cone.dual_dist
    g2 = cone.dual_grad[2]
    cone.dual_grad[2] = -cone.dual_grad[1]
    cone.dual_grad[1] = -g2

    cone.dual_grad_updated = true
    return cone.dual_grad
end

function update_hess(cone::EpiPerSquare)
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)
    H = cone.hess.data

    mul!(H, cone.grad, cone.grad')
    inv_dist = inv(cone.dist)
    @inbounds for j in 3:cone.dim
        H[j, j] += inv_dist
    end
    H[1, 2] -= inv_dist

    cone.hess_updated = true
    return cone.hess
end

function update_inv_hess(cone::EpiPerSquare)
    @assert cone.is_feas
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
    Hi = cone.inv_hess.data

    mul!(Hi, cone.point, cone.point')
    @inbounds for j in 3:cone.dim
        Hi[j, j] += cone.dist
    end
    Hi[1, 2] -= cone.dist

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiPerSquare,
    )
    u = cone.point[1]
    v = cone.point[2]
    w = @view cone.point[3:end]

    @inbounds for j in 1:size(prod, 2)
        uj = arr[1, j]
        vj = arr[2, j]
        @views wj = arr[3:end, j]
        ga = (dot(w, wj) - v * uj - u * vj) / cone.dist
        prod[1, j] = -ga * v - vj
        prod[2, j] = -ga * u - uj
        @. @views prod[3:end, j] = ga * w + wj
    end
    @. prod /= cone.dist

    return prod
end

function scal_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiPerSquare{T},
    mu::T,
    ) where {T <: Real}
    cone.nt_updated || update_nt(cone, mu)
    rot_hyperbolic_householder(prod, arr, cone.nt_point, cone.rt_dist_ratio, true)
    return prod
end

function inv_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiPerSquare,
    )
    @assert cone.is_feas

    @inbounds @views for j in 1:size(prod, 2)
        pa = dot(cone.point, arr[:, j])
        @. prod[:, j] = pa * cone.point
    end
    @. @views prod[1, :] -= cone.dist * arr[2, :]
    @. @views prod[2, :] -= cone.dist * arr[1, :]
    @. @views prod[3:end, :] += cone.dist * arr[3:end, :]

    return prod
end

function inv_scal_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiPerSquare{T},
    mu::T,
    ) where {T <: Real}
    cone.nt_updated || update_nt(cone, mu)
    rot_hyperbolic_householder(prod, arr, cone.nt_point, cone.rt_dist_ratio, false)
    return prod
end

function update_sqrt_hess_prod(cone::EpiPerSquare{T}) where T
    @assert cone.is_feas
    @assert !cone.sqrt_hess_prod_updated
    if !isdefined(cone, :sqrt_hess_vec)
        cone.sqrt_hess_vec = zeros(T, cone.dim)
    end

    rtdist = cone.rtdist = sqrt(cone.dist)
    cone.denom = 2 * rtdist + cone.point[1] + cone.point[2]
    vec = cone.sqrt_hess_vec
    @. @views vec[3:end] = cone.point[3:end] / rtdist
    vec[1] = -cone.point[2] / rtdist - 1
    vec[2] = -cone.point[1] / rtdist - 1

    cone.sqrt_hess_prod_updated = true
    return
end

function update_inv_sqrt_hess_prod(cone::EpiPerSquare{T}) where T
    @assert cone.is_feas
    @assert !cone.inv_sqrt_hess_prod_updated
    if !isdefined(cone, :inv_sqrt_hess_vec)
        cone.inv_sqrt_hess_vec = zeros(T, cone.dim)
    end

    rtdist = cone.rtdist = sqrt(cone.dist)
    cone.denom = 2 * rtdist + cone.point[1] + cone.point[2]
    vec = cone.inv_sqrt_hess_vec
    copyto!(vec, cone.point)
    vec[1:2] .+= rtdist

    cone.inv_sqrt_hess_prod_updated = true
    return
end

function sqrt_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiPerSquare{T},
    ) where {T <: Real}
    if !cone.sqrt_hess_prod_updated
        update_sqrt_hess_prod(cone)
    end
    vec = cone.sqrt_hess_vec
    rtdist = cone.rtdist

    @inbounds @views for j in 1:size(arr, 2)
        dotj = dot(vec, arr[:, j]) / cone.denom
        @. prod[:, j] = dotj * vec
    end
    @. @views prod[1, :] -= arr[2, :] / rtdist
    @. @views prod[2, :] -= arr[1, :] / rtdist
    @. @views prod[3:end, :] += arr[3:end, :] / rtdist

    return prod
end

function sqrt_scal_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiPerSquare{T},
    mu::T,
    ) where {T <: Real}
    cone.nt_updated || update_nt(cone, mu)
    rot_hyperbolic_householder(prod, arr, cone.nt_point_sqrt, cone.rt_rt_dist_ratio, true)
    return prod
end

function inv_sqrt_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiPerSquare{T},
    ) where {T <: Real}
    if !cone.inv_sqrt_hess_prod_updated
        update_inv_sqrt_hess_prod(cone)
    end
    vec = cone.inv_sqrt_hess_vec
    rtdist = cone.rtdist

    @inbounds @views for j in 1:size(arr, 2)
        dotj = dot(vec, arr[:, j]) / cone.denom
        @. prod[:, j] = dotj * vec
    end
    @. @views prod[1, :] -= arr[2, :] * rtdist
    @. @views prod[2, :] -= arr[1, :] * rtdist
    @. @views prod[3:end, :] += arr[3:end, :] * rtdist

    return prod
end

function update_nt(cone::EpiPerSquare{T}, mu::T) where {T <: Real}
    @assert cone.is_feas
    @assert cone.dual_feas_updated
    nt_point = cone.nt_point
    nt_point_sqrt = cone.nt_point_sqrt
    rt2 = sqrt(T(2))

    normalized_point = cone.point ./ sqrt(cone.dist * 2)
    normalized_dual_point = cone.dual_point ./ sqrt(cone.dual_dist * 2)
    gamma = sqrt((1 + dot(normalized_point, normalized_dual_point)) / 2)

    nt_point[1] = normalized_point[2] + normalized_dual_point[1]
    nt_point[2] = normalized_point[1] + normalized_dual_point[2]
    @. @views nt_point[3:end] = -normalized_point[3:end] + normalized_dual_point[3:end]
    nt_point ./= 2 * gamma

    copyto!(nt_point_sqrt, nt_point)
    nt_point_sqrt[1] += inv(rt2)
    nt_point_sqrt[2] += inv(rt2)
    nt_point_sqrt ./= sqrt(2 + (nt_point[1] + nt_point[2]) * rt2)

    cone.rt_dist_ratio = sqrt(cone.dist / cone.dual_dist)
    cone.rt_rt_dist_ratio = sqrt(cone.rt_dist_ratio)

    cone.nt_updated = true

    return
end

function update_scal_hess(cone::EpiPerSquare{T}, mu::T) where {T}
    @assert cone.grad_updated
    cone.nt_updated || update_nt(cone, mu)
    isdefined(cone, :scal_hess) || alloc_scal_hess!(cone)
    H = cone.scal_hess.data
    nt_point = cone.nt_point

    mul!(H, nt_point, nt_point', 2, false)
    @inbounds for j in 3:cone.dim
        H[j, j] += 1
    end
    H[1, 2] -= 1
    H ./= cone.rt_dist_ratio

    cone.scal_hess_updated = true
    return cone.scal_hess
end

function dder3(cone::EpiPerSquare, dir::AbstractVector)
    @assert cone.grad_updated
    dim = cone.dim
    dder3 = cone.dder3
    point = cone.point
    u = point[1]
    v = point[2]
    u_dir = dir[1]
    v_dir = dir[2]
    @views w = point[3:end]
    @views w_dir = dir[3:end]

    jdotpd = u * v_dir + v * u_dir - dot(w, w_dir)
    hess_prod!(dder3, dir, cone)
    dotdHd = -dot(dir, dder3)
    dotpHd = dot(point, dder3)
    dder3 .*= jdotpd
    @. @views dder3[3:end] += dotdHd * w + dotpHd * w_dir
    dder3[1] += -dotdHd * v - dotpHd * v_dir
    dder3[2] += -dotdHd * u - dotpHd * u_dir
    dder3 ./= 2 * cone.dist

    return dder3
end

function dder3(
    cone::EpiPerSquare{T},
    pdir::AbstractVector{T},
    ddir::AbstractVector{T},
    ) where {T <: Real}
    @assert cone.feas_updated
    @assert cone.dual_feas_updated
    dder3 = cone.dder3
    point = cone.point

    @views jdot_p_s = pdir[1] * point[2] + pdir[2] * point[1] -
        dot(point[3:end], pdir[3:end])
    @. dder3 = jdot_p_s * ddir
    dot_s_z = dot(pdir, ddir)
    dot_p_z = dot(point, ddir)
    @. @views dder3[1:2] += dot_s_z * point[2:-1:1] - dot_p_z * pdir[2:-1:1]
    @. @views dder3[3:end] += -dot_s_z * point[3:end] + dot_p_z * pdir[3:end]
    dder3 ./= -cone.dist * 2

    return dder3
end

# function bar(::EpiPerSquare)
#     function barrier(uvw)
#         (u, v, w) = (uvw[1], uvw[2], uvw[3:end])
#         return -log(2 * u * v - sum(abs2, w))
#     end
#     return barrier
# end

function rot_hyperbolic_householder(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    v::AbstractVector{T},
    fact::T,
    use_inv::Bool,
    ) where {T <: Real}
    for j in 1:size(prod, 2)
        if use_inv
            @views pa = 2 * dot(v, arr[:, j])
            @. @views prod[:, j] = pa * v
        else
            @views pa = 2 * (v[1] * arr[2] + v[2] * arr[1] - dot(v[3:end], arr[3:end, j]))
            prod[1, j] = pa * v[2]
            prod[2, j] = pa * v[1]
            @. @views prod[3:end, j] = -pa * v[3:end]
        end
    end
    @. prod[1, :] -= arr[2, :]
    @. prod[2, :] -= arr[1, :]
    @. prod[3:end, :] += arr[3:end, :]
    if use_inv
        prod ./= fact
    else
        prod .*= fact
    end
    return prod
end
