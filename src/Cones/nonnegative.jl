"""
$(TYPEDEF)

Nonnegative cone of dimension `dim`.

    $(FUNCTIONNAME){T}(dim::Int)
"""
mutable struct Nonnegative{T <: Real} <: Cone{T}
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
    scal_hess_updated::Bool
    inv_scal_hess_updated::Bool
    inv_hess_updated::Bool
    is_feas::Bool
    hess::Diagonal{T, Vector{T}}
    scal_hess::Diagonal{T, Vector{T}}
    inv_scal_hess::Diagonal{T, Vector{T}}
    inv_hess::Diagonal{T, Vector{T}}

    function Nonnegative{T}(dim::Int) where {T <: Real}
        @assert dim >= 1
        cone = new{T}()
        cone.dim = dim
        cone.dual_grad = zeros(T, dim)
        return cone
    end
end

use_scal(::Nonnegative) = true

use_dual_barrier(::Nonnegative) = false

reset_data(cone::Nonnegative) = (cone.feas_updated = cone.grad_updated =
    cone.dual_grad_updated = cone.hess_updated = cone.scal_hess_updated =
    cone.inv_hess_updated = cone.inv_scal_hess_updated = false)

use_sqrt_hess_oracles(::Int, cone::Nonnegative) = true

use_sqrt_scal_hess_oracles(::Int, cone::Nonnegative) = true

get_nu(cone::Nonnegative) = cone.dim

set_initial_point!(arr::AbstractVector, cone::Nonnegative) = (arr .= 1)

function update_feas(cone::Nonnegative{T}) where T
    @assert !cone.feas_updated
    cone.is_feas = all(>(eps(T)), cone.point)
    cone.feas_updated = true
    return cone.is_feas
end

is_dual_feas(cone::Nonnegative{T}) where T = all(>(eps(T)), cone.dual_point)

function update_grad(cone::Nonnegative)
    @assert cone.is_feas
    @. cone.grad = -inv(cone.point)
    cone.grad_updated = true
    return cone.grad
end

function update_dual_grad(cone::Nonnegative)
    @. cone.dual_grad = -inv(cone.dual_point)
    cone.dual_grad_updated = true
    return cone.dual_grad
end

function update_hess(cone::Nonnegative{T}) where T
    cone.grad_updated || update_grad(cone)
    if !isdefined(cone, :hess)
        cone.hess = Diagonal(zeros(T, cone.dim))
    end

    @. cone.hess.diag = abs2(cone.grad)
    cone.hess_updated = true
    return cone.hess
end

function update_scal_hess(cone::Nonnegative{T}) where {T <: Real}
    if !isdefined(cone, :scal_hess)
        cone.scal_hess = Diagonal(zeros(T, cone.dim))
    end

    @. cone.scal_hess.diag = cone.dual_point / cone.point
    cone.scal_hess_updated = true
    return cone.scal_hess
end

function update_inv_scal_hess(cone::Nonnegative{T}) where T
    if !isdefined(cone, :inv_scal_hess)
        cone.inv_scal_hess = Diagonal(zeros(T, cone.dim))
    end

    @. cone.inv_scal_hess.diag = cone.point / cone.dual_point
    cone.inv_scal_hess_updated = true
    return cone.inv_scal_hess
end

function update_inv_hess(cone::Nonnegative{T}) where T
    @assert cone.is_feas
    if !isdefined(cone, :inv_hess)
        cone.inv_hess = Diagonal(zeros(T, cone.dim))
    end

    @. cone.inv_hess.diag = abs2(cone.point)
    cone.inv_hess_updated = true
    return cone.inv_hess
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::Nonnegative,
    )
    @assert cone.is_feas
    @. prod = arr / cone.point / cone.point
    return prod
end

function inv_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::Nonnegative,
    )
    @assert cone.is_feas
    @. prod = arr * cone.point * cone.point
    return prod
end

function scal_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::Nonnegative{T},
    ) where {T <: Real}
    @assert cone.is_feas
    @. prod = arr * cone.dual_point / cone.point
    return prod
end

function inv_scal_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::Nonnegative{T},
    ) where {T <: Real}
    @assert cone.is_feas
    @. prod = arr * cone.point / cone.dual_point
    return prod
end

function sqrt_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::Nonnegative,
    )
    @assert cone.is_feas
    @. prod = arr / cone.point
    return prod
end

function sqrt_scal_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::Nonnegative{T},
    ) where {T <: Real}
    @assert cone.is_feas
    @. prod = arr * sqrt(cone.dual_point) / sqrt(cone.point)
    return prod
end

function inv_sqrt_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::Nonnegative,
    )
    @assert cone.is_feas
    @. prod = arr * cone.point
    return prod
end

function inv_sqrt_scal_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::Nonnegative{T},
    ) where {T <: Real}
    @assert cone.is_feas
    @. prod = arr * sqrt(cone.point) / sqrt(cone.dual_point)
    return prod
end

function dder3(cone::Nonnegative, dir::AbstractVector)
    @. cone.dder3 = abs2(dir / cone.point) / cone.point
    return cone.dder3
end

function dder3(
    cone::Nonnegative{T},
    pdir::AbstractVector{T},
    ddir::AbstractVector{T}
    ) where {T <: Real}
    @. cone.dder3 = -pdir * ddir / cone.point
    return cone.dder3
end

hess_nz_count(cone::Nonnegative) = cone.dim
hess_nz_count_tril(cone::Nonnegative) = cone.dim
inv_hess_nz_count(cone::Nonnegative) = cone.dim
inv_hess_nz_count_tril(cone::Nonnegative) = cone.dim
hess_nz_idxs_col(cone::Nonnegative, j::Int) = [j]
hess_nz_idxs_col_tril(cone::Nonnegative, j::Int) = [j]
inv_hess_nz_idxs_col(cone::Nonnegative, j::Int) = [j]
inv_hess_nz_idxs_col_tril(cone::Nonnegative, j::Int) = [j]

# nonnegative is not primitive, so sum and max proximity measures differ
function get_proxsqr(
    cone::Nonnegative{T},
    irtmu::T,
    use_max_prox::Bool,
    ) where {T <: Real}
    aggfun = (use_max_prox ? maximum : sum)
    return aggfun(abs2(si * zi * irtmu - 1) for (si, zi) in
        zip(cone.point, cone.dual_point))
end
function get_proxcompl(
    cone::Nonnegative{T},
    mu::T,
    ) where {T <: Real}
    return minimum(si * zi / mu for (si, zi) in zip(cone.point, cone.dual_point))
end
