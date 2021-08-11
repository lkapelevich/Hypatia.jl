"""
$(TYPEDEF)

Hypograph of perspective function of sum-log cone of dimension `dim`.

    $(FUNCTIONNAME){T}(dim::Int, use_dual::Bool = false)
"""
mutable struct HypoPerLog{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int

    point::Vector{T}
    dual_point::Vector{T}
    dual_grad::Vector{T}
    grad::Vector{T}
    dder3::Vector{T}
    vec1::Vector{T}
    vec2::Vector{T}
    feas_updated::Bool
    grad_updated::Bool
    dual_grad_updated::Bool
    hess_updated::Bool
    scal_hess_updated::Bool
    inv_scal_hess_updated::Bool
    inv_hess_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    scal_hess::Symmetric{T, Matrix{T}}
    inv_scal_hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_mat::Symmetric{T, Matrix{T}}
    hess_fact::Factorization{T}

    ϕ::T
    ζ::T
    dual_ϕ::T
    tempw::Vector{T}

    cone_mu::T

    function HypoPerLog{T}(
        dim::Int;
        use_dual::Bool = false,
        ) where {T <: Real}
        @assert dim >= 3
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        return cone
    end
end

reset_data(cone::HypoPerLog) = (cone.feas_updated = cone.grad_updated =
    cone.dual_grad_updated = cone.hess_updated = cone.scal_hess_updated =
    cone.inv_hess_updated = cone.inv_scal_hess_updated =
    cone.hess_fact_updated = false)

function setup_extra_data!(cone::HypoPerLog{T}) where {T <: Real}
    d = cone.dim - 2
    cone.dual_grad = zeros(T, 2 + d)
    cone.tempw = zeros(T, d)
    return cone
end

get_nu(cone::HypoPerLog) = cone.dim

use_sqrt_hess_oracles(::Int, cone::HypoPerLog) = false

function set_initial_point!(arr::AbstractVector, cone::HypoPerLog)
    (arr[1], arr[2], w) = get_central_ray_hypoperlog(cone.dim - 2)
    @views arr[3:end] .= w
    return arr
end

function update_feas(cone::HypoPerLog{T}) where {T <: Real}
    @assert !cone.feas_updated
    v = cone.point[2]
    @views w = cone.point[3:end]

    if v > eps(T) && all(>(eps(T)), w)
        u = cone.point[1]
        # bf_phi = sum(log(BigFloat(wi) / BigFloat(v)) for wi in w)
        # cone.ϕ = bf_phi
        cone.ϕ = sum(log(wi / v) for wi in w)
        cone.ζ = v * cone.ϕ - u
        cone.is_feas = (cone.ζ > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::HypoPerLog{T}) where {T <: Real}
    u = cone.dual_point[1]
    @views w = cone.dual_point[3:end]

    if all(>(eps(T)), w) && u < -eps(T)
        v = cone.dual_point[2]
        cone.dual_ϕ = sum(log(w_i / -u) for w_i in w)
        return (v - u * (cone.dual_ϕ + length(w)) > eps(T))
    end

    return false
end

function update_grad(cone::HypoPerLog)
    @assert cone.is_feas
    u = cone.point[1]
    v = cone.point[2]
    @views w = cone.point[3:end]
    d = length(w)
    ζ = cone.ζ
    g = cone.grad

    g[1] = inv(cone.ζ)
    g[2] = -(cone.ϕ - d) / ζ - inv(v)
    vζi1 = -1 - v / ζ
    @. g[3:end] = vζi1 / w

    cone.grad_updated = true
    return cone.grad
end

function update_dual_grad(cone::HypoPerLog)
    u = cone.dual_point[1]
    v = cone.dual_point[2]
    @views w = cone.dual_point[3:end]
    d = length(w)
    dg = cone.dual_grad

    β = 1 + d - v / u + cone.dual_ϕ
    # @show β / d - log(d)
    bomega = d * omegawright(β / d - log(d))
    @assert bomega + d * log(bomega) ≈ β

    dg[1] = (-d - 2 + v / u + 2 * bomega) / (u * (1 - bomega))
    dg[2] = -1 / (u * (1 - bomega))
    @. dg[3:end] = bomega / w / (1 - bomega)

    cone.dual_grad_updated = true
    return dg
end

function update_hess(cone::HypoPerLog)
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)
    v = cone.point[2]
    @views w = cone.point[3:end]
    H = cone.hess.data
    g = cone.grad
    ζ = cone.ζ
    wivζi = cone.tempw
    d = length(w)
    σζi = (cone.ϕ - d) / ζ
    vζi = v / ζ
    @. wivζi = vζi / w

    # u, v
    H[1, 1] = ζ^-2
    H[1, 2] = -σζi / ζ
    H[2, 2] = v^-2 + abs2(σζi) + d / ζ / v

    # u, v, w
    vζi2 = -vζi / ζ
    c1 = ((cone.ϕ - d) * vζi - 1) / ζ
    @. H[1, 3:end] = vζi2 / w
    @. H[2, 3:end] = c1 / w

    # w
    @inbounds for j in eachindex(wivζi)
        j2 = 2 + j
        wivζij = wivζi[j]
        for i in 1:j
            H[2 + i, j2] = wivζi[i] * wivζij
        end
        H[j2, j2] -= g[j2] / w[j]
    end

    # hess_check = ForwardDiff.hessian(bar(cone), BigFloat.(cone.point))
    # @show hess_check - cone.hess

    # @show dot(cone.point, cone.hess, cone.point) - 3

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::HypoPerLog,
    )
    v = cone.point[2]
    @views w = cone.point[3:end]
    ζ = cone.ζ
    d = length(w)
    σ = cone.ϕ - d
    rwi = cone.tempw
    vζi1 = v / ζ + 1

    @inbounds for j in 1:size(arr, 2)
        p = arr[1, j]
        q = arr[2, j]
        @. @views rwi = arr[3:end, j] / w

        qζi = q / ζ
        c0 = sum(rwi) / ζ
        # ∇ϕ[r] = v * c0
        c1 = (v * c0 - p / ζ + σ * qζi) / ζ
        c3 = c1 * v - qζi
        prod[1, j] = -c1
        prod[2, j] = c1 * σ - c0 + (qζi * d + q / v) / v
        @. prod[3:end, j] = (c3 + vζi1 * rwi) / w
    end

    return prod
end

function update_inv_hess(cone::HypoPerLog)
    @assert cone.grad_updated
    @assert !cone.inv_hess_updated
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
    v = cone.point[2]
    @views w = cone.point[3:end]
    d = length(w)
    Hi = cone.inv_hess.data
    ζ = cone.ζ
    ϕ = cone.ϕ
    ζv = ζ + v
    ζζvi = ζ / ζv
    c3 = v / (ζv + d * v)
    c0 = ϕ - d * ζζvi
    c2 = v * c3
    c4 = c2 * ζv
    c1 = v * ζζvi + c0 * c2

    Hi[1, 1] = abs2(v * ϕ) + ζ * (ζ + d * v) - d * abs2(ζ + v * ϕ) * c3
    Hi[1, 2] = c0 * c4
    Hi[2, 2] = c4

    @. Hi[1, 3:end] = c1 * w
    @. Hi[2, 3:end] = c2 * w

    @inbounds for j in eachindex(w)
        j2 = 2 + j
        Hi[j2, j2] += abs2(w[j])
    end
    @views mul!(Hi[3:end, 3:end], w, w', c2 / ζv, ζζvi)

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::HypoPerLog,
    )
    @assert cone.grad_updated
    v = cone.point[2]
    @views w = cone.point[3:end]
    d = length(w)
    ζ = cone.ζ
    ϕ = cone.ϕ
    ζv = ζ + v
    ζζvi = ζ / ζv
    c3 = v / (ζv + d * v)
    c0 = ϕ - d * ζζvi
    c4 = v * c3 * ζv
    c6 = abs2(v * ϕ) + ζ * (ζ + d * v) - d * abs2(ζ + v * ϕ) * c3
    c7 = c4 * c0
    c8 = c7 + v * ζ
    rw = cone.tempw

    @inbounds for j in 1:size(arr, 2)
        p = arr[1, j]
        q = arr[2, j]
        @. @views rw = arr[3:end, j] * w

        c1 = sum(rw) / ζv
        c5 = c0 * p + q + c1
        c2 = v * (ζζvi * p + c3 * c5)
        prod[1, j] = c6 * p + c7 * q + c8 * c1
        prod[2, j] = c4 * c5
        @. prod[3:end, j] = (c2 + ζζvi * rw) * w
    end

    return prod
end

function dder3(cone::HypoPerLog{T}, dir::AbstractVector{T}) where {T <: Real}
    @assert cone.grad_updated
    v = cone.point[2]
    @views w = cone.point[3:end]
    dder3 = cone.dder3
    p = dir[1]
    q = dir[2]
    ζ = cone.ζ
    d = length(w)
    σ = cone.ϕ - d
    viq = q / v
    viq2 = abs2(viq)
    rwi = cone.tempw
    vζi = v / ζ
    vζi1 = vζi + 1

    @. @views rwi = dir[3:end] / w
    c0 = sum(rwi)
    c7 = sum(abs2, rwi)
    ζiχ = (-p + σ * q + c0 * v) / ζ
    c4 = (viq * (-viq * d + 2 * c0) - c7) / ζ / 2
    c1 = (abs2(ζiχ) - v * c4) / ζ
    c3 = -(ζiχ + viq) / ζ
    c5 = c3 * q + vζi * viq2
    c6 = -2 * vζi * viq - c3 * v
    c8 = c5 + c1 * v

    dder3[1] = -c1
    dder3[2] = c1 * σ + (viq2 - (d * c5 + c6 * c0 + vζi * c7)) / v - c4
    @. dder3[3:end] = (c8 + rwi * (c6 + vζi1 * rwi)) / w

    return dder3
end

function bar(::HypoPerLog)
    function barrier(uvw)
        @views (u, v, w) = (uvw[1], uvw[2], uvw[3:end])
        lw = sum(log, w)
        return -log((lw - length(w) * log(v)) * v - u) - lw - log(v)
    end
    return barrier
end

# using ForwardDiff

# function dder3(
#     cone::HypoPerLog{T},
#     pdir::AbstractVector{T},
#     ddir::AbstractVector{T},
#     ) where {T <: Real}
#     @assert cone.grad_updated
#     dder3 = cone.dder3
#     d = cone.dim - 2
#     d1 = inv_hess_prod!(zeros(T, d + 2), ddir, cone)
#
#     function bar(uvw)
#         @views (u, v, w) = (uvw[1], uvw[2], uvw[3:end])
#         lw = sum(log, w)
#         return -log((lw - d * log(v)) * v - u) - lw - log(v)
#     end
#     bardir(point, s, t) = bar(point + s * d1 + t * pdir)
#     dder3 .= ForwardDiff.gradient(
#         s2 -> ForwardDiff.derivative(
#             s -> ForwardDiff.derivative(
#                 t -> bardir(s2, s, t),
#                 0),
#             0),
#         cone.point) / 2
#
#     # v = cone.point[2]
#     # @views w = cone.point[3:end]
#     # wi = inv.(w)
#     # ζ = cone.ζ
#     # τ = cone.ϕ - d
#     # σ = ζ + v * (1 + d)
#     # Tuuu = 2 / ζ^3
#     # Tuuv = -2 * τ / ζ^3
#     # Tuuw = -2 * v / ζ^3 ./ w
#     # Tuvv = 2 * τ^2 / ζ^3 + d / ζ^2 / v
#     # Tuvw = 2 * τ * v ./ (ζ^3 * w) - wi / ζ^2
#     # Tuww_c = 2 * v^2 / ζ^3
#     # Tuww_D = v / ζ^2 ./ w.^2
#     # Tvvv = -2 * τ^3 / ζ^3 - 3 * τ * d / ζ^2 / v - d / ζ / v^2 - 2 / v^3
#     # Tvvw = -2 * τ^2 * v / ζ^3 ./ w + 2 * τ / ζ^2 ./ w - d / ζ^2 ./ w
#     # Tvww_c = 2 * v / ζ^2 - 2 * v^2 * τ / ζ^3
#     # Tvww_D = -τ * v ./ w.^2 / ζ^2 + 1 ./ w.^2 / ζ
#     #
#     # Tuvw_p = dot(Tuvw, pdir[3:end])
#     # Tuvw_d = dot(Tuvw, d1[3:end])
#     # dp11 = d1[1] * pdir[1]
#     # dp12 = d1[1] * pdir[2] + d1[2] * pdir[1]
#     # dp22 = d1[2] * pdir[2]
#     # dp33 = d1[3:end] .* pdir[3:end]
#     # wip = dot(wi, pdir[3:end])
#     # wid = dot(wi, d1[3:end])
#     #
#     # dder3[1] = Tuuu * dp11 + Tuuv * dp12 + Tuvv * dp22 +
#     #     pdir[1] * dot(Tuuw, d1[3:end]) + d1[1] * dot(Tuuw, pdir[3:end]) +
#     #     pdir[2] * Tuvw_d + d1[2] * Tuvw_p +
#     #     dot(Tuww_D, dp33) +
#     #     Tuww_c * wid * wip
#     # dder3[2] = Tuuv * dp11 + Tuvv * dp12 + Tvvv * dp22 +
#     #     pdir[1] * Tuvw_d + d1[1] * Tuvw_p +
#     #     pdir[2] * dot(Tvvw, d1[3:end]) + d1[2] * dot(Tvvw, pdir[3:end]) +
#     #     dot(Tvww_D, dp33) +
#     #     Tvww_c * wid * wip
#     # dder3[3:end] .= Tuuw * dp11 + Tuvw * dp12 + Tvvw * dp22 +
#     #     Tuww_c * (d1[1] * wip + pdir[1] * wid) ./ w + Tuww_D .* (pdir[1] * d1[3:end] + d1[1] * pdir[3:end]) +
#     #     Tvww_c * (d1[2] * wip + pdir[2] * wid) ./ w + Tvww_D .* (pdir[2] * d1[3:end] + d1[2] * pdir[3:end]) +
#     #     -2 * v^3 ./ ζ^3 * wip * wid ./ w +
#     #     -abs2(v / ζ) * ((wid * pdir[3:end] + wip * d1[3:end]) ./ w .+ sum(pdir[3:end] .* d1[3:end] ./ w.^2)) ./ w +
#     #      (-2 * v / ζ - 2) ./ w.^3 .* dp33
#     #
#     # dder3 ./= 2
#
#     return dder3
# end

# see analysis in
# https://github.com/lkapelevich/HypatiaSupplements.jl/tree/master/centralpoints
function get_central_ray_hypoperlog(d::Int)
    if d <= 10
        # lookup points where x = f'(x)
        return central_rays_hypoperlog[d, :]
    end
    # use nonlinear fit for higher dimensions
    x = inv(d)
    if d <= 70
        u = 4.657876 * x ^ 2 - 3.116192 * x + 0.000647
        v = 0.424682 * x + 0.553392
        w = 0.760412 * x + 1.001795
    else
        u = -3.011166 * x - 0.000122
        v = 0.395308 * x + 0.553955
        w = 0.837545 * x + 1.000024
    end
    return [u, v, w]
end

const central_rays_hypoperlog = [
    -0.827838387  0.805102007  1.290927686
    -0.689607388  0.724605082  1.224617936
    -0.584372665  0.68128058  1.182421942
    -0.503499342  0.65448622  1.153053152
    -0.440285893  0.636444224  1.131466926
    -0.389979809  0.623569352  1.114979519
    -0.349255921  0.613978276  1.102013921
    -0.315769104  0.606589839  1.091577908
    -0.287837744  0.600745284  1.083013
    -0.264242734  0.596019009  1.075868782
    ]
