#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

epigraph of L-infinity norm
(u in R, w in R^n) : u >= norm_inf(w)

barrier from "Barrier Functions in Interior Point Methods" by Osman Guler
-sum_i(log(u - w_i^2/u)) - log(u)
=#

mutable struct EpiNormInf{T <: Real} <: Cone{T}
    use_dual::Bool
    dim::Int
    point::Vector{T}

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    inv_hess_prod_updated::Bool
    hess_prod_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    diag11::T
    diag2n::Vector{T}
    edge2n::Vector{T}
    div2n::Vector{T}
    schur::T

    function EpiNormInf{T}(dim::Int, is_dual::Bool) where {T <: Real}
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        return cone
    end
end

EpiNormInf{T}(dim::Int) where {T <: Real} = EpiNormInf{T}(dim, false)

reset_data(cone::EpiNormInf) = (cone.feas_updated = cone.grad_updated = cone.hess_updated =
    cone.inv_hess_updated = cone.inv_hess_prod_updated = cone.hess_prod_updated = false)

# TODO maybe only allocate the fields we use
function setup_data(cone::EpiNormInf{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.diag2n = zeros(T, dim - 1)
    cone.edge2n = zeros(T, dim - 1)
    cone.div2n = zeros(T, dim - 1)
    return
end

get_nu(cone::EpiNormInf) = cone.dim

function set_initial_point(arr::AbstractVector, cone::EpiNormInf)
    arr .= 0
    arr[1] = 1
    return arr
end

function update_feas(cone::EpiNormInf)
    @assert !cone.feas_updated
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    cone.is_feas = (u > 0 && u > norm(w, Inf))
    cone.feas_updated = true
    return cone.is_feas
end

# TODO maybe move the diag2n, edge2n to update hess/inv_hess functions
function update_grad(cone::EpiNormInf{T}) where {T <: Real}
    @assert cone.is_feas
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    g1 = zero(u)
    usqr = abs2(u)
    @inbounds for (j, wj) in enumerate(w)
        iuw2u = 2 / (u - abs2(wj) / u)
        g1 += iuw2u
        iu2w2 = 2 / (usqr - abs2(wj))
        iu2ww = wj * iu2w2
        cone.grad[j + 1] = iu2ww
        cone.diag2n[j] = iu2w2 + abs2(iu2ww)
        @assert cone.diag2n[j] > 0
    end
    cone.grad[1] = (cone.dim - 2) / u - g1

    cone.grad_updated = true
    return cone.grad
end

# symmetric arrow matrix
function update_hess(cone::EpiNormInf)
    @assert cone.grad_updated
    if !cone.hess_prod_updated
        update_hess_prod(cone)
    end
    H = cone.hess.data
    H[1, 1] = cone.diag11
    @inbounds for j in 2:cone.dim
        H[1, j] = cone.edge2n[j - 1]
        H[j, j] = cone.diag2n[j - 1]
    end
    cone.hess_updated = true
    return cone.hess
end

# Diag(0, inv(diag)) + xx' / schur, where x = (-1, edge ./ diag)
function update_inv_hess(cone::EpiNormInf)
    @assert cone.grad_updated
    if !cone.inv_hess_prod_updated
        update_inv_hess_prod(cone)
    end
    cone.inv_hess.data[1, 1] = 1
    @. cone.inv_hess.data[1, 2:end] = cone.div2n
    @inbounds for j in 2:cone.dim, i in 2:j
        cone.inv_hess.data[i, j] = cone.inv_hess.data[1, j] * cone.inv_hess.data[1, i]
    end
    cone.inv_hess.data ./= cone.schur
    @inbounds for j in 2:cone.dim
        cone.inv_hess.data[j, j] += inv(cone.diag2n[j - 1])
    end
    cone.inv_hess_updated = true
    return cone.inv_hess
end

function update_hess_prod(cone::EpiNormInf{T}) where {T <: Real}
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    usqr = abs2(u)
    cone.diag11 = 2 * sum((abs2(u) + abs2(wj)) / (abs2(u) - abs2(wj))^2 for wj in w) - (cone.dim - 2) / u / u
    # cone.diag11 = zero(T)
    for j in eachindex(w)
        iuw2u = 2 / (u - abs2(w[j]) / u)
        iu2w2 = 2 / (usqr - abs2(w[j]))
        iu2ww = w[j] * iu2w2
        # cone.diag11 += (abs2(u) + abs2(w[j])) / (abs2(u) - abs2(w[j]))^2
        cone.edge2n[j] = -iuw2u * iu2ww # -abs2(iu2w2) * u * w[j] #
    end
    # cone.diag11 *= 2
    # cone.diag11 -= (cone.dim - 2) / u / u
    @assert cone.diag11 > 0

    cone.hess_prod_updated = true
    return nothing
end

function update_inv_hess_prod(cone::EpiNormInf)
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    cone.schur = 2 * sum(inv(u^2 + wj^2) for wj in w) - (cone.dim - 2) / u / u
    @assert cone.schur > 0
    # -edge / diag
    for j in eachindex(w)
        cone.div2n[j] = 2 * u * w[j] / (abs2(u) + abs2(w[j]))
    end

    cone.inv_hess_prod_updated = true
    return nothing
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormInf)
    @assert cone.grad_updated
    if !cone.hess_prod_updated
        update_hess_prod(cone)
    end
    @inbounds for j in 1:size(prod, 2)
        @views prod[1, j] = cone.diag11 * arr[1, j] + dot(cone.edge2n, arr[2:end, j])
        @views @. prod[2:end, j] = cone.edge2n * arr[1, j] + cone.diag2n * arr[2:end, j]
    end
    return prod
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormInf)
    @assert cone.grad_updated
    if !cone.inv_hess_prod_updated
        update_inv_hess_prod(cone)
    end
    @inbounds for j in 1:size(prod, 2)
        @views prod[1, j] = arr[1, j] + dot(cone.div2n, arr[2:end, j])
        @. prod[2:end, j] = cone.div2n * prod[1, j]
    end
    prod ./= cone.schur
    @. @views prod[2:end, :] += arr[2:end, :] / cone.diag2n
    return prod
end
