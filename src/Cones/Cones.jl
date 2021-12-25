"""
Proper cone definitions, oracles, and utilities.
"""
module Cones

using DocStringExtensions
using LinearAlgebra
import LinearAlgebra.copytri!
import LinearAlgebra.HermOrSym
import LinearAlgebra.BlasReal
import PolynomialRoots
using SparseArrays
import Hypatia.RealOrComplex
import Hypatia.outer_prod!
import Hypatia.update_eigen!
import Hypatia.spectral_outer!
import Hypatia.posdef_fact_copy!
import Hypatia.inv_fact!

include("arrayutilities.jl")
include("conjutilities.jl")

"""
$(TYPEDEF)

A proper cone.
"""
abstract type Cone{T <: Real} end

"""
$(SIGNATURES)

The real vector dimension of the cone.
"""
dimension(cone::Cone)::Int = cone.dim

"""
$(SIGNATURES)

The barrier parameter ``\\nu`` of the cone.
"""
get_nu(cone::Cone)::Real = cone.nu

"""
$(SIGNATURES)

Whether the cone can have scaling updates.
"""
use_scal(cone::Cone) = false

"""
$(SIGNATURES)

Set the array equal to the initial interior point for the cone.
"""
function set_initial_point!(arr::AbstractVector, cone::Cone) end

"""
$(SIGNATURES)

Returns true if and only if the currently-loaded primal point is strictly
feasible for the cone.
"""
is_feas(cone::Cone)::Bool = (cone.feas_updated ? cone.is_feas : update_feas(cone))

"""
$(SIGNATURES)

Returns false only if the currently-loaded dual point is outside the interior of
the cone's dual cone.
"""
is_dual_feas(cone::Cone)::Bool = true

"""
$(SIGNATURES)

The gradient of the cone's barrier function at the currently-loaded primal point.
"""
grad(cone::Cone) = (cone.grad_updated ? cone.grad : update_grad(cone))

"""
$(SIGNATURES)
The gradient of the cone's conjugate barrier function at the currently-loaded
dual point.
"""
dual_grad(cone::Cone) = (cone.dual_grad_updated ? cone.dual_grad :
    update_dual_grad(cone))

"""
$(SIGNATURES)

The Hessian (symmetric positive definite) of the cone's barrier function at the
currently-loaded primal point.
"""
function hess(cone::Cone)
    cone.hess_updated && return cone.hess
    return update_hess(cone)
end

"""
$(SIGNATURES)

The scaling matrix or Hessian with scaling updates of the cone's barrier
function at the currently-loaded primal point.
"""
function scal_hess(cone::Cone)
    !use_scal(cone) && return hess(cone)
    cone.scal_hess_updated && return cone.scal_hess
    return update_scal_hess(cone)
end

"""
$(SIGNATURES)

The inverse Hessian (symmetric positive definite) of the cone's barrier function
at the currently-loaded primal point.
"""
function inv_hess(cone::Cone)
    cone.inv_hess_updated && return cone.inv_hess
    return update_inv_hess(cone)
end

"""
$(SIGNATURES)

The inverse scaling matrix or Hessian with scaling updates of the cone's barrier
function at the currently-loaded primal point.
"""
function inv_scal_hess(cone::Cone)
    !use_scal(cone) && return inv_hess(cone)
    cone.inv_scal_hess_updated && return cone.inv_scal_hess
    return update_inv_scal_hess(cone)
end

"""
$(SIGNATURES)

Compute the product of the Hessian of the cone's barrier function at the
currently-loaded primal point with a vector or array, in-place.
"""
function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Cone)
    cone.hess_updated || update_hess(cone)
    mul!(prod, cone.hess, arr)
    return prod
end

"""
$(SIGNATURES)

Compute the product of the inverse Hessian of the cone's barrier function at the
currently-loaded primal point with a vector or array, in-place.
"""
function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Cone)
    update_hess_fact(cone)
    # TODO try equilibration, iterative refinement etc like posvx/sysvx
    ldiv!(prod, cone.hess_fact, arr)
    return prod
end

"""
$(SIGNATURES)

Returns true if and only if the oracle for the third-order directional
derivative oracle [`dder3`](@ref) can be computed.
"""
use_dder3(::Cone)::Bool = true

"""
$(SIGNATURES)

Compute the third-order directional derivative, in the direction `dir`, the
cone's barrier function at the currently-loaded primal point.
"""
function dder3(cone::Cone, dir::AbstractVector) end

# other oracles and helpers

use_dual_barrier(cone::Cone)::Bool = cone.use_dual_barrier

function setup_data!(cone::Cone{T}) where {T <: Real}
    reset_data(cone)
    dim = dimension(cone)
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    if hasproperty(cone, :dder3)
        cone.dder3 = zeros(T, dim)
    end
    if use_scal(cone)
        cone.dual_grad = zeros(T, dim)
    end
    cone.vec1 = zeros(T, dim)
    cone.vec2 = zeros(T, dim)
    setup_extra_data!(cone)
    return cone
end

setup_extra_data!(cone::Cone) = nothing

load_point(
    cone::Cone{T},
    point::AbstractVector{T},
    scal::T,
    ) where {T <: Real} = (@. cone.point = scal * point)

load_point(
    cone::Cone,
    point::AbstractVector,
    ) = copyto!(cone.point, point)

load_dual_point(
    cone::Cone,
    point::AbstractVector,
    ) = copyto!(cone.dual_point, point)

function alloc_hess!(cone::Cone{T}) where {T <: Real}
    dim = dimension(cone)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    return
end

function alloc_scal_hess!(cone::Cone{T}) where {T <: Real}
    @assert use_scal(cone)
    dim = dimension(cone)
    cone.scal_hess = Symmetric(zeros(T, dim, dim), :U)
    return
end

function alloc_inv_hess!(cone::Cone{T}) where {T <: Real}
    dim = dimension(cone)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    return
end

function alloc_inv_scal_hess!(cone::Cone{T}) where {T <: Real}
    @assert use_scal(cone)
    dim = dimension(cone)
    cone.inv_scal_hess = Symmetric(zeros(T, dim, dim), :U)
    return
end

reset_data(cone::Cone) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = cone.hess_fact_updated = false)

# decide whether to use sqrt oracles
function use_sqrt_hess_oracles(arr_dim::Int, cone::Cone)
    if !cone.hess_fact_updated
        (arr_dim < dimension(cone)) && return false # array is small
        update_hess_fact(cone) || return false
    end
    return (cone.hess_fact isa Cholesky)
end

function use_sqrt_scal_hess_oracles(arr_dim::Int, cone::Cone)
    !use_scal(cone) && return use_sqrt_hess_oracles(arr_dim, cone)
    if !cone.scal_hess_fact_updated
        (arr_dim < dimension(cone)) && return false # array is small
        update_scal_hess_fact(cone) || return false
    end
    return (cone.scal_hess_fact isa Cholesky)
end

# naive fallback
function update_dual_grad(cone::Cone{T}) where {T <: Real}
    quad_bound = 0.35
    s = cone.point
    z = cone.dual_point
    g = grad(cone)
    r = z + g
    Hir = inv_hess_prod!(copy(r), r, cone)
    n = n_prev = sqrt(dot(r, Hir))
    # curr = cone.dual_grad
    # curr .= s
    curr = copy(s)
    old_s = copy(s)
    # @show n
    max_iter = 40
    iter = 1
    while n > 1000eps(T)
        α = (n > quad_bound ? inv(1 + n) : 1)
        curr -= Hir * α
        load_point(cone, curr)
        reset_data(cone)
        update_feas(cone)
        g = update_grad(cone)
        r = z + g
        Hir = inv_hess_prod!(copy(r), r, cone)
        n = sqrt(dot(r, Hir))
        # @show n
        iter += 1
        if (n_prev < quad_bound) && (n > 1000(n_prev / (1 - n_prev))^2)
            break
        end
        n_prev = n
        (iter > max_iter) && break
    end
    # curr *= -1
    cone.dual_grad = -curr
    load_point(cone, old_s)
    reset_data(cone)
    update_feas(cone)
    update_grad(cone)
    cone.dual_grad_updated = true
    return cone.dual_grad
end

# only use if use_sqrt_hess_oracles is true
function sqrt_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::Cone,
    )
    @assert cone.hess_fact_updated
    mul!(prod, cone.hess_fact.U, arr)
    return prod
end

# NOTE worse convergence than no sqrts
function sqrt_scal_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::Cone,
    )
    !use_scal(cone) && return sqrt_hess_prod!(prod, arr, cone)
    @assert cone.scal_hess_fact_updated
    mul!(prod, cone.scal_hess_fact.U, arr)
    return prod
end

# only use if use_sqrt_hess_oracles is true
function inv_sqrt_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::Cone,
    )
    @assert cone.hess_fact_updated
    # TODO try equilibration, iterative refinement etc like posvx/sysvx
    ldiv!(prod, cone.hess_fact.U', arr)
    return prod
end

function inv_sqrt_scal_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::Cone,
    )
    !use_scal(cone) && return inv_sqrt_hess_prod!(prod, arr, cone)
    @assert cone.scal_hess_fact_updated
    ldiv!(prod, cone.scal_hess_fact.U', arr)
    return prod
end

update_hess_aux(cone::Cone) = nothing

function update_use_hess_prod_slow(cone::Cone{T}) where {T <: Real}
    cone.hess_updated || update_hess(cone)
    @assert cone.hess_updated
    rel_viol = abs(1 - dot(cone.point, cone.hess, cone.point) / get_nu(cone))
    # TODO tune
    cone.use_hess_prod_slow = (rel_viol > dimension(cone) * sqrt(eps(T)))
    # cone.use_hess_prod_slow && println("switching to slow hess prod")
    cone.use_hess_prod_slow_updated = true
    return
end

hess_prod_slow!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::Cone,
    ) = hess_prod!(prod, arr, cone)

function update_hess_fact(cone::Cone{T}) where {T <: Real}
    cone.hess_fact_updated && return true
    cone.hess_updated || update_hess(cone)
    if !isdefined(cone, :hess_fact_mat)
        cone.hess_fact_mat = zero(cone.hess)
    end

    # do not modify the hessian during recovery
    cone.hess_fact = posdef_fact_copy!(cone.hess_fact_mat, cone.hess, false)

    cone.hess_fact_updated = true
    return issuccess(cone.hess_fact)
end

function update_scal_hess_fact(cone::Cone{T}) where {T <: Real}
    !use_scal(cone) &&  update_hess_fact(cone)
    cone.scal_hess_fact_updated && return true
    cone.scal_hess_fact_updated = true
    if update_hess_fact(cone) && cone.hess_fact isa Cholesky
        s = cone.point
        z = cone.dual_point
        ts = -dual_grad(cone)
        tz = -grad(cone)

        nu = get_nu(cone)
        dot_sz = dot(s, z)
        cone_mu = dot_sz / nu

        tmu = dot(ts, tz) / nu
        tol = sqrt(eps(T))
        if cone_mu * tmu < 1 - tol
            @show "bad", cone_mu * tmu
        end
        ds = s - cone_mu * ts
        dz = z - cone_mu * tz
        Hts = hess_prod!(copy(ts), ts, cone)

        fact = cone.scal_hess_fact = copy(cone.hess_fact)
        fact.factors .*= sqrt(cone_mu)
        if (norm(ds) < tol) || (norm(dz) < tol) || (cone_mu * tmu - 1 < tol) ||
            (dot(ts, Hts) - nu * tmu^2 < tol)
            return true
        else
            lowrankupdate!(fact, z / sqrt(dot_sz))
            lowrankupdate!(fact, dz / sqrt(dot(ds, dz)))
            try
                lowrankdowndate!(fact, tz * sqrt(cone_mu / nu))
                c5 = cone_mu / (dot(ts, Hts) - nu * tmu^2)
                v1 = Hts - tmu * tz
                lowrankdowndate!(fact, v1 * sqrt(c5))
                return true
            catch _
                @warn "downdate failed"
            end
        end
    end
    scal_hess(cone)
    if !isdefined(cone, :scal_hess_fact_mat)
        cone.scal_hess_fact_mat = zero(cone.scal_hess)
    end
    cone.scal_hess_fact = posdef_fact_copy!(cone.scal_hess_fact_mat, cone.scal_hess, false)
    # cone.scal_hess_fact = bunchkaufman(scal_hess(cone, mu))

    return issuccess(cone.scal_hess_fact)
end

function update_inv_hess(cone::Cone)
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
    update_hess_fact(cone)
    inv_fact!(cone.inv_hess.data, cone.hess_fact)
    cone.inv_hess_updated = true
    return cone.inv_hess
end

# number of nonzeros in the Hessian and inverse
hess_nz_count(cone::Cone) = dimension(cone) ^ 2
hess_nz_count_tril(cone::Cone) = svec_length(dimension(cone))
inv_hess_nz_count(cone::Cone) = dimension(cone) ^ 2
inv_hess_nz_count_tril(cone::Cone) = svec_length(dimension(cone))
# row indices of nonzero elements in column j
hess_nz_idxs_col(cone::Cone, j::Int) = 1:dimension(cone)
hess_nz_idxs_col_tril(cone::Cone, j::Int) = j:dimension(cone)
inv_hess_nz_idxs_col(cone::Cone, j::Int) = 1:dimension(cone)
inv_hess_nz_idxs_col_tril(cone::Cone, j::Int) = j:dimension(cone)

# check numerics of some oracles used in proximity check TODO tune
function check_numerics(
    cone::Cone{T},
    gtol::T = sqrt(sqrt(eps(T))),
    Htol::T = 10sqrt(gtol),
    ) where {T <: Real}
    g = grad(cone)
    dim = length(g)
    nu = get_nu(cone)

    # grad check
    (abs(1 + dot(g, cone.point) / nu) > gtol * dim) && return false

    # inv hess check
    Hig = inv_hess_prod!(cone.vec1, g, cone)
    (abs(1 - dot(Hig, g) / nu) > Htol * dim) && return false

    return true
end

# compute squared proximity value for a cone
# NOTE if cone is not primitive (eg nonnegative), sum and max proximities differ
function get_proxsqr(
    cone::Cone{T},
    irtmu::T,
    ::Bool,
    negtol::T = sqrt(eps(T)),
    ) where {T <: Real}
    g = grad(cone)
    vec1 = cone.vec1
    vec2 = cone.vec2

    # @. vec1 = irtmu * cone.dual_point + g # correct with scaling
    # @. vec1 = cone.dual_point + g * inv(abs2(irtmu)) # wrong but works without scaling
    @. vec1 = cone.dual_point / irtmu + g * inv(abs2(irtmu)) # wrong but works with scaling
    # @. vec1 = cone.dual_point / irtmu^2 + g # correct without scaling
    inv_hess_prod!(vec2, vec1, cone)
    prox_sqr = dot(vec2, vec1)
    (prox_sqr < -negtol * length(g)) && return T(Inf) # should be positive

    return abs(prox_sqr)
end

function get_proxcompl(cone::Cone{T}, mu::T) where {T <: Real}
    g = grad(cone)
    dg = dual_grad(cone)
    nu = get_nu(cone)
    return nu / dot(g, dg) / mu
end

# NOTE if near the central path, updates should be skipped and mu*Hess should
# be used, where mu is the global mu that is not passed into these functions.
# on the first iteration, this happens but both global and local mus are
# equal to one and that's OK. in other cases, we return an approximation of
# mu*Hess by using the local mu (this doesn't happen often).
function update_scal_hess(cone::Cone{T}) where {T <: Real}
    @assert use_scal(cone)
    if !isdefined(cone, :scal_hess)
        dim = dimension(cone)
        cone.scal_hess = Symmetric(zeros(T, dim, dim), :U)
    end
    old_hess = hess(cone)
    H = cone.scal_hess.data
    s = cone.point
    z = cone.dual_point
    ts = -dual_grad(cone)
    tz = -grad(cone)

    nu = get_nu(cone)
    mu = dot(s, z) / nu
    tmu = dot(ts, tz) / nu
    tol = sqrt(eps(T))
    if mu * tmu < 1 - tol
        @show "bad", mu * tmu
    end

    ds = s - mu * ts
    dz = z - mu * tz
    Hts = hess_prod!(copy(ts), ts, cone)
    if (norm(ds) < tol) || (norm(dz) < tol) || (mu * tmu - 1 < tol) ||
        (abs(dot(ts, Hts) - nu * tmu^2) < tol)
        # @show "~~ skipping updates ~~"
        H .= old_hess * mu
    else
        v1 = z + mu * tz + dz / (mu * tmu - 1)
        v2 = Hts - tmu * tz
        M1 = dz * v1'
        H .= old_hess * mu + 1 / (2 * mu * nu) * (M1 + M1') - mu /
            (dot(ts, Hts) - nu * tmu^2) * v2 * v2'
    end
    cone.scal_hess_updated = true
    return cone.scal_hess
end

function update_inv_scal_hess(cone::Cone{T}) where {T <: Real}
    @assert use_scal(cone)
    if !isdefined(cone, :inv_scal_hess)
        dim = dimension(cone)
        cone.inv_scal_hess = Symmetric(zeros(T, dim, dim), :U)
    end
    cone.inv_scal_hess = inv(scal_hess(cone))
    cone.inv_scal_hess_updated = true
    return cone.inv_scal_hess
end

# function update_inv_scal_hess(cone::Cone{T}) where {T <: Real}
#     if !isdefined(cone, :inv_scal_hess)
#         dim = dimension(cone)
#         cone.inv_scal_hess = Symmetric(zeros(T, dim, dim), :U)
#     end
#     update_scal_hess_fact(cone)
#     inv_fact!(cone.inv_scal_hess.data, cone.scal_hess_fact)
#     cone.inv_scal_hess_updated = true
#     return cone.inv_scal_hess
# end

# function update_inv_scal_hess(cone::Cone{T}) where {T <: Real}
#     @assert use_scal(cone)
#     if !isdefined(cone, :inv_scal_hess)
#         dim = dimension(cone)
#         cone.inv_scal_hess = Symmetric(zeros(T, dim, dim), :U)
#     end
#
#     s = cone.point
#     z = cone.dual_point
#     ts = -dual_grad(cone)
#     tz = -grad(cone)
#
#     nu = get_nu(cone)
#     cone_mu = dot(s, z) / nu
#     tmu = dot(ts, tz) / nu
#
#     ds = s - cone_mu * ts
#     dz = z - cone_mu * tz
#     Hts = hess_prod!(copy(ts), ts, cone)
#     tol = sqrt(eps(T))
#     if (norm(ds) < tol) || (norm(dz) < tol) || (cone_mu * tmu - 1 < tol) ||
#         (abs(dot(ts, Hts) - nu * tmu^2) < tol)
#         # @show "~~ skipping updates ~~"
#         cone.inv_scal_hess.data .= inv_hess(cone) ./ cone_mu
#     else
#         v1 = z + cone_mu * tz + dz / (cone_mu * tmu - 1)
#         # TODO dot(ts, Hts) - nu * tmu^2 should be negative
#         v2 = sqrt(cone_mu) * (Hts - tmu * tz) / sqrt(abs(dot(ts, Hts) - nu * tmu^2))
#
#         c1 = 1 / sqrt(2 * cone_mu * nu)
#         U = hcat(c1 * dz, c1 * v1, -v2)
#         V = hcat(c1 * v1, c1 * dz, v2)'
#
#         t2 = V * inv_hess(cone) / cone_mu
#         t3 = inv_hess_prod!(copy(U), U, cone) / cone_mu
#         t4 = I + V * t3
#         t5 = t4 \ t2
#         t6 = U * t5
#         t7 = inv_hess_prod!(copy(t6), t6, cone) / cone_mu
#         cone.inv_scal_hess.data .= inv_hess(cone) ./ cone_mu - t7
#     end
#     @show cone.inv_scal_hess ./ (inv(cone.scal_hess))
#
#     cone.inv_scal_hess_updated = true
#     return cone.inv_scal_hess
# end

function scal_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::Cone{T},
    slow::Bool = false,
    ) where {T <: Real}
    if !use_scal(cone)
        slow ? hess_prod_slow!(prod, arr, cone) : hess_prod!(prod, arr, cone)
        return prod
    end

    s = cone.point
    z = cone.dual_point
    ts = -dual_grad(cone)
    tz = -grad(cone)

    dot_sz = dot(s, z)
    nu = get_nu(cone)
    cone_mu = dot_sz / nu
    tmu = dot(ts, tz) / nu

    ds = s - cone_mu * ts
    dz = z - cone_mu * tz
    if slow
        Hts = hess_prod_slow!(copy(ts), ts, cone)
        hess_prod_slow!(prod, arr, cone)
    else
        Hts = hess_prod!(copy(ts), ts, cone)
        hess_prod!(prod, arr, cone)
    end
    prod .*= cone_mu

    tol = sqrt(eps(T))
    if (norm(ds) > tol) && (norm(dz) > tol) && (cone_mu * tmu - 1 > tol) &&
        (abs(dot(ts, Hts) - nu * tmu^2) > tol)
        v1 = z + cone_mu * tz + dz / (cone_mu * tmu - 1)
        v2 = Hts - tmu * tz
        tsHts = dot(ts, Hts)
        @inbounds for j in 1:size(arr, 2)
            prod_j = view(prod, :, j)
            arr_j = view(arr, :, j)
            d1 = dot(v1, arr_j)
            d2 = dot(dz, arr_j)
            d3 = dot(v2, arr_j)
            @. prod_j += d1 / (2 * dot_sz) * dz
            @. prod_j += d2 / (2 * dot_sz) * v1
            @. prod_j -= d3 * cone_mu / (tsHts - nu * tmu^2) * v2
        end
    end

    return prod
end

function inv_scal_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::Cone,
    )
    !use_scal(cone) && return inv_hess_prod!(prod, arr, cone)
    # prod .= cholesky(scal_hess(cone, mu)) \ arr
    update_scal_hess_fact(cone)
    ldiv!(prod, cone.scal_hess_fact, arr)
    return prod
end

# function inv_scal_hess_prod!(
#     prod::AbstractVecOrMat,
#     arr::AbstractVecOrMat,
#     cone::Cone,
#     )
#
#     s = cone.point
#     z = cone.dual_point
#     ts = -dual_grad(cone)
#     tz = -grad(cone)
#
#     nu = get_nu(cone)
#     cone_mu = dot(s, z) / nu
#     tmu = dot(ts, tz) / nu
#
#     ds = s - cone_mu * ts
#     dz = z - cone_mu * tz
#     Hts = hess_prod!(copy(ts), ts, cone)
#     tol = sqrt(eps(T))
#     # tol = 1000eps(T)
#     if (norm(ds) < tol) || (norm(dz) < tol) || (cone_mu * tmu - 1 < tol) ||
#         (abs(dot(ts, Hts) - nu * tmu^2) < tol)
#         # @show "~~ skipping updates ~~"
#         inv_hess_prod!(prod, arr, cone)
#         prod ./= cone_mu
#     else
#         v1 = z + cone_mu * tz + dz / (cone_mu * tmu - 1)
#         v2 = sqrt(cone_mu) * (Hts - tmu * tz) / sqrt(abs(dot(ts, Hts) - nu * tmu^2))
#
#         c1 = 1 / sqrt(2 * cone_mu * nu)
#         U = hcat(c1 * dz, c1 * v1, -v2)
#         V = hcat(c1 * v1, c1 * dz, v2)'
#
#         t1 = inv_hess_prod!(copy(arr), arr, cone) / cone_mu
#         t2 = V * t1
#         t3 = inv_hess_prod!(copy(U), U, cone) / cone_mu
#         t4 = I + V * t3
#         t5 = t4 \ t2
#         t6 = U * t5
#         t7 = inv_hess_prod!(copy(t6), t6, cone) / cone_mu
#         prod .= t1 - t7
#     end
#
#     return prod
# end

include("nonnegative.jl")
include("possemideftri.jl")
include("doublynonnegativetri.jl")
include("possemideftrisparse/possemideftrisparse.jl")
include("linmatrixineq.jl")
include("epinorminf.jl")
include("epinormeucl.jl")
include("epipersquare.jl")
include("epinormspectraltri.jl")
include("epinormspectral.jl")
include("matrixepipersquare.jl")
include("generalizedpower.jl")
include("hypopowermean.jl")
include("hypogeomean.jl")
include("hyporootdettri.jl")
include("hypoperlog.jl")
include("hypoperlogdettri.jl")
include("epipersepspectral/epipersepspectral.jl")
include("epirelentropy.jl")
include("epitrrelentropytri.jl")
include("wsosinterpnonnegative.jl")
include("wsosinterppossemideftri.jl")
include("wsosinterpepinormone.jl")
include("wsosinterpepinormeucl.jl")

end
