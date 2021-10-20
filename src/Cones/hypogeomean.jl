"""
$(TYPEDEF)

Hypograph of geometric mean cone of dimension `dim`.

    $(FUNCTIONNAME){T}(dim::Int, use_dual::Bool = false)
"""
mutable struct HypoGeoMean{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int

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
    inv_hess_updated::Bool
    scal_hess_updated::Bool
    inv_scal_hess_updated::Bool
    hess_fact_updated::Bool
    scal_hess_fact_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    scal_hess::Symmetric{T, Matrix{T}}
    inv_scal_hess::Symmetric{T, Matrix{T}}
    hess_fact_mat::Symmetric{T, Matrix{T}}
    scal_hess_fact_mat::Symmetric{T, Matrix{T}}
    hess_fact::Factorization{T}
    scal_hess_fact::Factorization{T}

    di::T
    ϕ::T
    ζ::T
    ϕζidi::T
    dual_ϕ::T
    tempw::Vector{T}

    function HypoGeoMean{T}(
        dim::Int;
        use_dual::Bool = false,
        ) where {T <: Real}
        @assert dim >= 2
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        return cone
    end
end

use_scal(::HypoGeoMean) = true

reset_data(cone::HypoGeoMean) = (cone.feas_updated = cone.grad_updated =
    cone.dual_grad_updated = cone.hess_updated = cone.scal_hess_updated =
    cone.inv_hess_updated = cone.inv_scal_hess_updated =
    cone.hess_fact_updated = cone.scal_hess_fact_updated = false)

function setup_extra_data!(cone::HypoGeoMean{T}) where {T <: Real}
    d = cone.dim - 1
    cone.dual_grad = zeros(T, 1 + d)
    cone.di = inv(T(d))
    cone.tempw = zeros(T, d)
    return cone
end

get_nu(cone::HypoGeoMean) = cone.dim

use_sqrt_scal_hess_oracles(::Int, cone::HypoGeoMean{T}, ::T) where {T <: Real} = false

function set_initial_point!(
    arr::AbstractVector{T},
    cone::HypoGeoMean{T},
    ) where {T <: Real}
    (u, w) = get_central_ray_hypogeomean(T, cone.dim - 1)
    arr[1] = u
    arr[2:end] .= w
    return arr
end

function update_feas(cone::HypoGeoMean{T}) where {T <: Real}
    @assert !cone.feas_updated
    u = cone.point[1]
    @views w = cone.point[2:end]

    if all(>(eps(T)), w)
        cone.ϕ = exp(cone.di * sum(log, w))
        cone.ζ = cone.ϕ - u
        cone.is_feas = (cone.ζ > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::HypoGeoMean{T}) where {T <: Real}
    u = cone.dual_point[1]
    @views w = cone.dual_point[2:end]

    if (u < -eps(T)) && all(>(eps(T)), w)
        cone.dual_ϕ = exp(cone.di * sum(log, w))
        return (length(w) * cone.dual_ϕ + u > eps(T))
    end

    return false
end

function update_grad(cone::HypoGeoMean)
    @assert cone.is_feas
    @views w = cone.point[2:end]
    g = cone.grad
    ζ = cone.ζ
    cone.ϕζidi = cone.ϕ / ζ * cone.di
    ϕζidi1 = -cone.ϕζidi - 1

    g[1] = inv(ζ)
    @. g[2:end] = ϕζidi1 / w

    cone.grad_updated = true
    return cone.grad
end

function update_dual_grad(cone::HypoGeoMean)
    u = cone.dual_point[1]
    @views w = cone.dual_point[2:end]
    d = length(w)
    dg = cone.dual_grad
    dual_ϕ = cone.dual_ϕ

    @. @views dg[2:end] = -1 / w / (1 + u / d / dual_ϕ)
    dg[1] = -inv(u) - d / (d * dual_ϕ + u)

    cone.dual_grad_updated = true
    return dg
end

function update_hess(cone::HypoGeoMean)
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)
    @views w = cone.point[2:end]
    H = cone.hess.data
    ζ = cone.ζ
    ϕζidi = cone.ϕζidi
    c4 = ϕζidi - cone.di
    c1 = ϕζidi * (1 + c4) + 1

    H[1, 1] = ζ^-2

    @inbounds for j in eachindex(w)
        j1 = j + 1
        w_j = w[j]
        c3 = ϕζidi / w_j
        H[1, j1] = -c3 / ζ

        c2 = c3 * c4
        for i in 1:(j - 1)
            H[i + 1, j1] = c2 / w[i]
        end
        H[j1, j1] = c1 / w_j / w_j
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::HypoGeoMean{T},
    ) where {T <: Real}
    @assert cone.grad_updated
    @views w = cone.point[2:end]
    di = cone.di
    ζ = cone.ζ
    ϕζidi = cone.ϕζidi
    rwi = cone.tempw
    ϕζidi1 = ϕζidi + 1

    @inbounds for j in 1:size(arr, 2)
        p = arr[1, j]
        @views r = arr[2:end, j]
        @. rwi = r / w

        c0 = ϕζidi * sum(rwi)
        c1 = c0 - p / ζ
        prod[1, j] = c1 / -ζ
        c2 = ϕζidi * c1 - di * c0
        @. prod[2:end, j] = (c2 + ϕζidi1 * rwi) / w
    end

    return prod
end

function update_inv_hess(cone::HypoGeoMean{T}) where {T <: Real}
    @assert cone.grad_updated
    @assert !cone.inv_hess_updated
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
    u = cone.point[1]
    @views w = cone.point[2:end]
    Hi = cone.inv_hess.data
    ζ = cone.ζ
    ϕ = cone.ϕ
    di = cone.di
    ϕdi = ϕ * di
    c2 = inv(cone.ϕζidi + 1)
    c3 = c2 / ζ * di

    Hi[1, 1] = abs2(ζ) + ϕdi * ϕ

    @inbounds for j in eachindex(w)
        j1 = j + 1
        w_j = w[j]
        c5 = ϕdi * w_j
        Hi[1, j1] = c5

        c4 = c3 * c5
        for i in 1:(j - 1)
            Hi[i + 1, j1] = c4 * w[i]
        end
        Hi[j1, j1] = (c4 + c2 * w_j) * w_j
    end

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::HypoGeoMean{T},
    ) where {T <: Real}
    @assert cone.grad_updated
    @views w = cone.point[2:end]
    ζ = cone.ζ
    ϕ = cone.ϕ
    di = cone.di
    rw = cone.tempw
    ϕdi = ϕ * di
    c2 = inv(cone.ϕζidi + 1)
    c3 = c2 / ζ * di
    c4 = abs2(ζ) + ϕdi * ϕ

    @inbounds for j in 1:size(prod, 2)
        p = arr[1, j]
        @views r = arr[2:end, j]
        @. rw = r * w

        c5 = sum(rw)
        prod[1, j] = ϕdi * c5 + c4 * p
        c6 = ϕdi * (c3 * c5 + p)
        @. prod[2:end, j] = (c6 + c2 * rw) * w
    end

    return prod
end

function bar(::HypoGeoMean)
    function barrier(uw)
        (u, w) = (uw[1], uw[2:end])
        lw = sum(log, w)
        return -log(exp(lw / length(w)) - u) - lw
    end
    return barrier
end

function dder3(cone::HypoGeoMean{T}, dir::AbstractVector{T}) where {T <: Real}
    @assert cone.grad_updated
    @views w = cone.point[2:end]
    dder3 = cone.dder3
    p = dir[1]
    @views r = dir[2:end]
    ζ = cone.ζ
    di = cone.di
    ϕ = cone.ϕ
    ϕζidi = cone.ϕζidi
    rwi = cone.tempw

    @. rwi = r / w
    c0 = sum(rwi) * di
    c6 = sum(abs2, rwi) * di
    ζiχ = (p - ϕ * c0) / ζ
    c1 = ζiχ^2 + ϕ / ζ * (c6 - abs2(c0)) / 2
    c7 = ϕζidi * (c1 - c6 / 2 + c0 * (ζiχ + c0 / 2))
    c8 = -ϕζidi * (ζiχ + c0)
    c9 = ϕζidi + 1

    dder3[1] = c1 / -ζ
    @. dder3[2:end] = (c7 + rwi * (c8 + c9 * rwi)) / w

    return dder3
end

function dder3(
    cone::HypoGeoMean{T},
    pdir::AbstractVector{T},
    ddir::AbstractVector{T},
    ) where {T <: Real}
    dder3 = cone.dder3
    d1 = inv_hess_prod!(zeros(T, cone.dim), ddir, cone)
    di = cone.di

    p = pdir[1]
    @views r = pdir[2:end]
    x = d1[1]
    @views z = d1[2:end]
    u = cone.point[1]
    @views w = cone.point[2:end]
    ζ = -cone.ζ
    ϕ = cone.ϕ

    rwi = r ./ w
    zwi = z ./ w
    tr_rwi = sum(rwi) * di
    tr_zwi = sum(zwi) * di

    χ_1 = -p + ϕ * tr_rwi
    χ_2 = -x + ϕ * tr_zwi

    dot_rzwi = dot(rwi, zwi)
    c1 = 2 * ζ^(-3) * χ_1 * χ_2 + ζ^(-2) * ϕ * (tr_rwi * tr_zwi - di * dot_rzwi)

    dder3[1] = -c1
    rz_ζ_χ_wi = (r * χ_2 / ζ + z * χ_1 / ζ) ./ w
    rzwi = rwi .* zwi
    τ = (sum(rz_ζ_χ_wi) * di .- rz_ζ_χ_wi .+
        tr_rwi * tr_zwi .- tr_rwi * zwi .- tr_zwi * rwi .- di * dot_rzwi .+
            2 * rzwi) * ϕ * di ./ w / ζ
    dder3[2:end] .= c1 * ϕ * di ./ w + τ - 2 * rzwi ./ w

    dder3 ./= 2

    # barrier = bar(cone)
    # bardir(point, s, t) = barrier(point + s * d1 + t * pdir)
    # true_dder3 = ForwardDiff.gradient(
    #     s2 -> ForwardDiff.derivative(
    #         s -> ForwardDiff.derivative(
    #             t -> bardir(s2, s, t),
    #             0),
    #         0),
    #     cone.point) / 2
    #
    # @show true_dder3 ./ dder3

    return dder3

end

function get_central_ray_hypogeomean(::Type{T}, d::Int) where {T <: Real}
    c = sqrt(T(5 * d ^ 2 + 2 * d + 1))
    u = -sqrt((-c + 3 * d + 1) / T(2 + 2 * d))
    w = (c - d + 1) / sqrt((1 + d) * (-2 * c + 6 * d + 2))
    return (u, w)
end
