"""
$(TYPEDEF)

Generalized power cone parametrized by powers `α` in the unit simplex and
dimension `d` of the normed variables.

    $(FUNCTIONNAME){T}(α::Vector{T}, d::Int, use_dual::Bool = false)
"""
mutable struct GeneralizedPower{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    α::Vector{T}
    n::Int
    equal_powers::Bool

    point::Vector{T}
    dual_point::Vector{T}
    grad::Vector{T}
    dual_grad::Vector{T}
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
    scal_hess_fact_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    scal_hess::Symmetric{T, Matrix{T}}
    inv_scal_hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_mat::Symmetric{T, Matrix{T}}
    scal_hess_fact_mat::Symmetric{T, Matrix{T}}
    hess_fact::Factorization{T}
    scal_hess_fact::Factorization{T}

    u_idxs::UnitRange{Int}
    w_idxs::UnitRange{Int}
    z::T
    w2::T
    zw::T
    zwzwi::T
    tempu1::Vector{T}
    tempu2::Vector{T}

    function GeneralizedPower{T}(
        α::Vector{T},
        n::Int;
        use_dual::Bool = false,
        ) where {T <: Real}
        @assert n >= 1
        dim = length(α) + n
        @assert dim >= 3
        @assert all(α_i > 0 for α_i in α)
        @assert sum(α) ≈ 1
        cone = new{T}()
        cone.n = n
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        cone.α = α
        return cone
    end
end

reset_data(cone::GeneralizedPower) = (cone.feas_updated = cone.grad_updated =
    cone.dual_grad_updated = cone.hess_updated = cone.scal_hess_updated =
    cone.inv_hess_updated = cone.inv_scal_hess_updated =
    cone.hess_fact_updated = cone.scal_hess_fact_updated = false)

function setup_extra_data!(cone::GeneralizedPower{T}) where {T <: Real}
    m = length(cone.α)
    cone.u_idxs = 1:m
    cone.w_idxs = (m + 1):cone.dim
    cone.tempu1 = zeros(T, m)
    cone.tempu2 = zeros(T, m)
    cone.dual_grad = zeros(T, cone.dim)
    return cone
end

get_nu(cone::GeneralizedPower) = length(cone.α) + 1

use_sqrt_hess_oracles(::Int, cone::GeneralizedPower) = false

function set_initial_point!(arr::AbstractVector, cone::GeneralizedPower)
    @. arr[cone.u_idxs] = sqrt(1 + cone.α)
    arr[cone.w_idxs] .= 0
    return arr
end

function update_feas(cone::GeneralizedPower{T}) where {T <: Real}
    @assert !cone.feas_updated
    @views u = cone.point[cone.u_idxs]

    if all(>(eps(T)), u)
        cone.z = exp(2 * sum(α_i * log(u_i) for (α_i, u_i) in zip(cone.α, u)))
        @views cone.w2 = sum(abs2, cone.point[cone.w_idxs])
        cone.zw = cone.z - cone.w2
        cone.is_feas = (cone.zw > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::GeneralizedPower{T}) where {T <: Real}
    @views u = cone.dual_point[cone.u_idxs]

    if all(>(eps(T)), u)
        p = exp(2 * sum(α_i * log(u_i / α_i) for (α_i, u_i) in zip(cone.α, u)))
        @views w = cone.dual_point[cone.w_idxs]
        return (p - sum(abs2, w) > eps(T))
    end

    return false
end

function update_grad(cone::GeneralizedPower)
    @assert cone.is_feas
    u_idxs = cone.u_idxs
    w_idxs = cone.w_idxs
    @views u = cone.point[u_idxs]
    @views w = cone.point[w_idxs]
    g = cone.grad
    cone.zwzwi = (cone.z + cone.w2) / cone.zw

    @. g[u_idxs] = -(cone.zwzwi * cone.α + 1) / u
    @. g[w_idxs] = 2 * w / cone.zw

    cone.grad_updated = true
    return cone.grad
end

function conj_tgp(pr, alpha, k)
    (p, r) = (pr[1], pr[2:end])

    phi(w) = exp(sum(2 * alpha .* log.(w)))
    # k = phi(r)
    d = length(alpha)
    inner_bound = -1 / p - (1 + sign(p) * 1 / p * sqrt(k * (d^2 / p^2 * k + d^2 - 1))) / (p / d - d * k / p)
    gamma = abs(p) / sqrt(phi(r ./ alpha))
    outer_bound = (1 + d) * gamma / (1 - gamma) / p
    @show inner_bound, outer_bound

    fval(y) = sum(2 * ai * log(2 * ai * y^2 + (1 + ai) * 2 * y / p) for ai in alpha) -
        log(k) - log(2 * y / p + y^2) - 2 * log(2 * y / p)
    grad(y) = 2 * sum(ai^2 / (ai * y + (1 + ai) / p) for ai in alpha) -
        2 * (y + 1 / p) / y / (y + 2 / p)
    hess_bound(y) = 2 * sum(ai^3 / (ai * y + (1 + ai) / p)^2 for ai in alpha) + 2 * (y^2 + 2 / p^2) / (y^2 * (y + 2 / p)^2)

    @assert hess_bound(inner_bound) > hess_bound(outer_bound) > 0
    C = hess_bound(inner_bound) /
        # abs(1 / p / (inner_bound * (inner_bound + 2 / p)))
        (4 * abs((outer_bound + 1 / p) / (outer_bound * (outer_bound + 2 / p))))
    gap = abs(inner_bound - outer_bound)

    while (C * gap > 1) && (gap > 1e-8) # think about latter
        @show C
        @show gap
        @show fval(inner_bound)
        @show fval(outer_bound)
        # uses the fact that function has known monotonicity
        new_bound = (inner_bound + outer_bound) / 2
        if fval(new_bound) >= 0
            inner_bound = new_bound
        else
            outer_bound = new_bound
        end

        C = hess_bound(inner_bound) /
            (4 * abs((outer_bound + 1 / p) / (outer_bound * (outer_bound + 2 / p))))
        @assert hess_bound(inner_bound) > hess_bound(outer_bound) > 0
        gap = abs(inner_bound - outer_bound)
    end

    new_bound = (inner_bound + outer_bound) / 2
    iter = 0
    while abs(fval(new_bound)) > 1e-10
        new_bound -= fval(new_bound) / grad(new_bound)
        iter += 1
        @show iter
    end

    return new_bound
end

function update_dual_grad(cone::GeneralizedPower{T}) where {T <: Real}
    u_idxs = cone.u_idxs
    w_idxs = cone.w_idxs
    @views u = cone.dual_point[u_idxs]
    @views w = cone.dual_point[w_idxs][1]
    α = cone.α
    g = cone.dual_grad

    m = length(u)
    dual_z = exp(sum(2 * α .* log.(u)))
    if all(isequal(inv(T(m))), α)
        tgp = -1 / w - (1 + sign(w) * 1 / w * sqrt(dual_z * (m^2 / w^2 * dual_z +
            m^2 - 1))) / (w / m - m * dual_z / w)
    else
        tgp = conj_tgp(vcat(w, u), α, dual_z)
    end
    zeta = 2 * tgp / w[1]
    phitgr = zeta + tgp^2

    @. g[w_idxs] = [tgp]
    @. g[u_idxs] = -(2 * α * phitgr + (1 .- α) * zeta) ./ u / zeta

    cone.dual_grad_updated = true
    return cone.dual_grad
end

function update_hess(cone::GeneralizedPower)
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)
    u_idxs = cone.u_idxs
    w_idxs = cone.w_idxs
    @views u = cone.point[u_idxs]
    H = cone.hess.data
    g = cone.grad
    zw = cone.zw
    aui = cone.tempu1
    auizzwi = cone.tempu2
    @. aui = 2 * cone.α / u
    @. auizzwi = -cone.z * aui / zw
    zzwim1 = -cone.w2 / zw
    zwi = 2 / zw

    @inbounds for j in u_idxs
        auizzwim1 = auizzwi[j] * zzwim1
        for i in 1:j
            H[i, j] = aui[i] * auizzwim1
        end
        H[j, j] -= g[j] / u[j]
    end

    @inbounds for j in w_idxs
        gj = g[j]
        for i in u_idxs
            H[i, j] = auizzwi[i] * gj
        end
        for i in first(w_idxs):j
            H[i, j] = g[i] * gj
        end
        H[j, j] += zwi
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::GeneralizedPower,
    )
    @assert cone.grad_updated
    u_idxs = cone.u_idxs
    w_idxs = cone.w_idxs
    @views u = cone.point[u_idxs]
    @views w = cone.point[w_idxs]
    α = cone.α
    z = cone.z
    zw = cone.zw
    tempu1 = cone.tempu1
    @. tempu1 = 1 + cone.zwzwi * α

    @inbounds for j in 1:size(arr, 2)
        @views begin
            arr_u = arr[u_idxs, j]
            arr_w = arr[w_idxs, j]
            prod_u = prod[u_idxs, j]
            prod_w = prod[w_idxs, j]
        end
        @. prod_u = arr_u / u
        @. prod_w = 2 * arr_w / zw

        dot1 = -4 * dot(α, prod_u) * z / zw
        dot2 = (dot1 + 2 * dot(w, prod_w)) / zw
        dot3 = dot1 - dot2 * z
        prod_u .*= tempu1
        @. prod_u += dot3 * α
        prod_u ./= u
        @. prod_w += dot2 * w
    end

    return prod
end

function bar(cone::GeneralizedPower)
    function barrier(uw)
        (u, w) = (uw[cone.u_idxs], uw[cone.w_idxs])
        α = cone.α
        return -log(prod(u.^(2 * α)) - sum(abs2, w)) - sum((1 .- α) .* log.(u))
    end
    return barrier
end

function dder3(cone::GeneralizedPower, dir::AbstractVector)
    @assert cone.grad_updated
    u_idxs = cone.u_idxs
    w_idxs = cone.w_idxs
    @views u = cone.point[u_idxs]
    @views w = cone.point[cone.w_idxs]
    dder3 = cone.dder3
    @views u_dder3 = dder3[u_idxs]
    @views w_dder3 = dder3[cone.w_idxs]
    @views u_dir = dir[u_idxs]
    @views w_dir = dir[cone.w_idxs]
    α = cone.α
    z = cone.z
    zw = cone.zw
    w2 = cone.w2
    zwzwi = cone.zwzwi
    zzwi = 2 * z / zw
    zwi = 2 / zw
    udu = cone.tempu1
    @. udu = u_dir / u

    wwd = 2 * dot(w, w_dir)
    c15 = wwd / zw
    audu = dot(α, udu)
    sumaudu2 = sum(α_i * abs2(udu_i) for (α_i, udu_i) in zip(α, udu))
    c1 = 2 * zwzwi * abs2(audu) + sumaudu2
    c10 = sum(abs2, w_dir) + wwd * c15

    c13 = zzwi * (w2 * c1 - 2 * wwd * zwzwi * audu + c10) / zw
    c14 = zzwi * (2 * audu * w2 - wwd) / zw
    @. u_dder3 = (c13 * α + ((c14 + zwzwi * udu) * α + udu) * udu) / u

    c6 = zwi * (z * (4 * audu * c15 - c1) - c10) / zw
    c7 = zwi * (2 * z * audu - wwd) / zw
    @. w_dder3 = c7 * w_dir + c6 * w

    return cone.dder3
end
