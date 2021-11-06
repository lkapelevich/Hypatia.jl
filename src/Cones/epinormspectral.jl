"""
$(TYPEDEF)

Epigraph of real or complex matrix spectral norm (i.e. maximum singular value)
for a matrix (stacked column-wise) of `nrows` rows and `ncols` columns with
`nrows ≤ ncols`.

    $(FUNCTIONNAME){T, R}(nrows::Int, ncols::Int, use_dual::Bool = false)
"""
mutable struct EpiNormSpectral{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    d1::Int
    d2::Int
    is_complex::Bool

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
    hess_aux_updated::Bool
    scal_hess_updated::Bool
    inv_scal_hess_updated::Bool
    hess_fact_updated::Bool # TODO remove with closed form inverse Hessian
    scal_hess_fact_updated::Bool # TODO remove with closed form inverse Hessian
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    scal_hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    inv_scal_hess::Symmetric{T, Matrix{T}}
    hess_fact_mat::Symmetric{T, Matrix{T}}
    scal_hess_fact_mat::Symmetric{T, Matrix{T}}
    hess_fact::Factorization{T}
    scal_hess_fact::Factorization{T}

    W::Matrix{R}
    dual_W_svd
    Z::Matrix{R}
    fact_Z
    dual_z::T
    Zi::Matrix{R}
    tau::Matrix{R}
    HuW::Matrix{R}
    Huu::T
    trZi2::T
    WtauI::Matrix{R}
    Zitau::Matrix{R}
    tempd1d2::Matrix{R}
    tempd1d2b::Matrix{R}
    tempd1d2c::Matrix{R}
    tempd1d2d::Matrix{R}
    tempd1d1::Matrix{R}
    tempd2d2::Matrix{R}
    tempd2d2b::Matrix{R}

    function EpiNormSpectral{T, R}(
        d1::Int,
        d2::Int;
        use_dual::Bool = false,
        ) where {T <: Real, R <: RealOrComplex{T}}
        @assert 1 <= d1 <= d2
        cone = new{T, R}()
        cone.use_dual_barrier = use_dual
        cone.is_complex = (R <: Complex)
        cone.dim = 1 + vec_length(R, d1 * d2)
        cone.d1 = d1
        cone.d2 = d2
        return cone
    end
end

use_scal(::EpiNormSpectral{T, T}) where {T <: Real} = true

reset_data(cone::EpiNormSpectral) = (cone.feas_updated = cone.grad_updated =
    cone.dual_grad_updated =
    cone.hess_updated = cone.scal_hess_updated =
    cone.inv_hess_updated = cone.inv_scal_hess_updated = cone.hess_aux_updated =
    cone.hess_fact_updated = cone.scal_hess_fact_updated = false)

# there should eventually be an explicit hess inv prod
use_sqrt_scal_hess_oracles(::Int, cone::EpiNormSpectral{T, T}, ::T) where {T <: Real} = false

# TODO only allocate the fields we use
function setup_extra_data!(
    cone::EpiNormSpectral{T, R},
    ) where {T <: Real, R <: RealOrComplex{T}}
    dim = cone.dim
    cone.dual_grad = zeros(R, dim)
    (d1, d2) = (cone.d1, cone.d2)
    cone.W = zeros(R, d1, d2)
    cone.Z = zeros(R, d1, d1)
    cone.Zi = zeros(R, d1, d1)
    cone.tau = zeros(R, d1, d2)
    cone.HuW = zeros(R, d1, d2)
    cone.WtauI = zeros(R, d2, d2)
    cone.Zitau = zeros(R, d1, d2)
    cone.tempd1d2 = zeros(R, d1, d2)
    cone.tempd1d2b = zeros(R, d1, d2)
    cone.tempd1d2c = zeros(R, d1, d2)
    cone.tempd1d2d = zeros(R, d1, d2)
    cone.tempd1d1 = zeros(R, d1, d1)
    cone.tempd2d2 = zeros(R, d2, d2)
    cone.tempd2d2b = zeros(R, d2, d2)
    return cone
end

get_nu(cone::EpiNormSpectral) = cone.d1 + 1

function set_initial_point!(
    arr::AbstractVector,
    cone::EpiNormSpectral{T, R},
    ) where {T <: Real, R <: RealOrComplex{T}}
    arr .= 0
    arr[1] = sqrt(T(get_nu(cone)))
    return arr
end

function update_feas(cone::EpiNormSpectral{T}) where T
    @assert !cone.feas_updated
    u = cone.point[1]

    if u > eps(T)
        @views vec_copyto!(cone.W, cone.point[2:end])
        copyto!(cone.Z, abs2(u) * I)
        mul!(cone.Z, cone.W, cone.W', -1, true)
        cone.fact_Z = cholesky!(Hermitian(cone.Z, :U), check = false)
        cone.is_feas = isposdef(cone.fact_Z)
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::EpiNormSpectral{T}) where {T <: Real}
    u = cone.dual_point[1]
    if u > eps(T)
        W = @views vec_copyto!(cone.tempd1d2, cone.dual_point[2:end])
        cone.dual_W_svd = svd(W)
        cone.dual_z = u - sum(cone.dual_W_svd.S)
        return (cone.dual_z > eps(T))
    end
    return false
end

function update_grad(cone::EpiNormSpectral)
    @assert cone.is_feas
    u = cone.point[1]
    Zi = cone.Zi

    ldiv!(cone.tau, cone.fact_Z, cone.W)
    inv_fact!(Zi, cone.fact_Z)
    copytri!(Zi, 'U', true)

    cone.grad[1] = -u * tr(Hermitian(Zi, :U))
    @views vec_copyto!(cone.grad[2:end], cone.tau)
    cone.grad .*= 2
    cone.grad[1] += (cone.d1 - 1) / u

    cone.grad_updated = true
    return cone.grad
end

function update_dual_grad(
    cone::EpiNormSpectral{T, R},
    ) where {T <: Real, R <: RealOrComplex{T}}
    u = cone.dual_point[1]
    dual_W_svd = cone.dual_W_svd
    w = dual_W_svd.S

    (new_bound, zw2) = epinorminf_dg(u, w, cone.d1, cone.dual_z)

    cone.dual_grad[1] = new_bound
    cone.dual_grad[2:end] .= vec(dual_W_svd.U * Diagonal(zw2) * dual_W_svd.Vt)

    cone.dual_grad_updated = true
    return cone.dual_grad
end

function update_hess_aux(cone::EpiNormSpectral)
    @assert cone.grad_updated
    u = cone.point[1]
    tau = cone.tau
    Zitau = cone.Zitau
    WtauI = cone.WtauI

    copyto!(Zitau, tau)
    ldiv!(cone.fact_Z, Zitau)
    @. cone.HuW = -4 * u * Zitau
    cone.trZi2 = sum(abs2, cone.Zi)
    cone.Huu = 4 * abs2(u) * cone.trZi2 + (cone.grad[1] - 2 *
        (cone.d1 - 1) / u) / u
    copyto!(WtauI, I)
    mul!(WtauI, cone.W', tau, true, true)

    cone.hess_aux_updated = true
    return
end

function update_hess(cone::EpiNormSpectral)
    cone.hess_aux_updated || update_hess_aux(cone)
    isdefined(cone, :hess) || alloc_hess!(cone)
    H = cone.hess.data
    d1 = cone.d1
    d2 = cone.d2
    Zi = cone.Zi
    tau = cone.tau
    WtauI = cone.WtauI

    # H_W_W part
    # TODO parallelize loops
    idx_incr = (cone.is_complex ? 2 : 1)
    r_idx = 2
    for i in 1:d2, j in 1:d1
        c_idx = r_idx
        @inbounds for k in i:d2
            taujk = tau[j, k]
            WtauIik = WtauI[i, k]
            lstart = (i == k ? j : 1)
            @inbounds for l in lstart:d1
                term1 = Zi[l, j] * WtauIik
                term2 = tau[l, i] * taujk
                spectral_kron_element!(H, r_idx, c_idx, term1, term2)
                c_idx += idx_incr
            end
        end
        r_idx += idx_incr
    end
    H .*= 2

    # H_u_W and H_u_u parts
    @views vec_copyto!(H[1, 2:end], cone.HuW)
    H[1, 1] = cone.Huu

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiNormSpectral,
    )
    cone.hess_aux_updated || update_hess_aux(cone)
    u = cone.point[1]
    W = cone.W
    tempd1d2 = cone.tempd1d2
    tempd1d1 = cone.tempd1d1

    @inbounds for j in 1:size(prod, 2)
        arr_1j = arr[1, j]
        @views vec_copyto!(tempd1d2, arr[2:end, j])
        prod[1, j] = cone.Huu * arr_1j + real(dot(cone.HuW, tempd1d2))
        mul!(tempd1d1, tempd1d2, W')
        @inbounds for k in 1:cone.d1
            @inbounds for i in 1:k
                tempd1d1[i, k] += tempd1d1[k, i]'
            end
            tempd1d1[k, k] -= 2 * u * arr_1j
        end
        mul!(tempd1d2, Hermitian(tempd1d1, :U), cone.tau, 2, 2)
        ldiv!(cone.fact_Z, tempd1d2)
        @views vec_copyto!(prod[2:end, j], tempd1d2)
    end

    return prod
end

function dder3(cone::EpiNormSpectral, dir::AbstractVector)
    @assert cone.hess_aux_updated

    u = cone.point[1]
    W = cone.W
    u_dir = dir[1]
    @views W_dir = vec_copyto!(cone.tempd1d2, dir[2:end])
    dder3 = cone.dder3

    Zi = Hermitian(cone.Zi, :U)
    tau = cone.tau
    Zitau = cone.Zitau
    WtauI = cone.WtauI
    tempd1d2b = cone.tempd1d2b
    tempd1d2c = cone.tempd1d2c
    tempd1d2d = cone.tempd1d2d
    tempd1d1 = cone.tempd1d1
    tempd2d2 = cone.tempd2d2
    tempd2d2b = cone.tempd2d2b

    mul!(tempd2d2b, W_dir', tau)
    ldiv!(tempd1d2d, cone.fact_Z, W_dir)
    mul!(tempd1d2b, tempd1d2d, WtauI)
    mul!(tempd1d2c, tempd1d2d, tempd2d2b')
    mul!(tempd1d1, tempd1d2d, W')
    mul!(tempd2d2, tempd2d2b, tempd2d2b)

    mul!(tempd2d2, W_dir', tempd1d2b, true, true)
    mul!(tempd1d2d, tau, tempd2d2)
    mul!(tempd1d2d, tempd1d2c, WtauI, true, true)
    mul!(tempd1d2d, tempd1d2b, tempd2d2b, true, true)

    ldiv!(cone.fact_Z, tempd1d2b)
    mul!(tempd1d2b, Zitau, tempd2d2b, true, true)

    mul!(tempd1d1, tau, W_dir', true, true)
    mul!(tempd1d2b, tempd1d1, Zitau, true, true)
    tempd1d2b .*= -2 * u

    const1 = 4 * u * u_dir * u
    @. tempd1d2c = const1 * Zitau - u_dir * tau
    ldiv!(cone.fact_Z, tempd1d2c)
    tempd1d2b .+= tempd1d2c

    axpby!(-2 * u_dir, tempd1d2b, -2, tempd1d2d)
    @views vec_copyto!(dder3[2:end], tempd1d2d)

    trZi3 = sum(abs2, ldiv!(tempd1d1, cone.fact_Z.L, Zi))
    @. tempd1d2b += 3 * tempd1d2c
    dder3[1] = -real(dot(W_dir, tempd1d2b)) - u * u_dir * (6 * cone.trZi2 -
        8 * u * trZi3 * u) * u_dir - (cone.d1 - 1) * abs2(u_dir / u) / u

    return dder3
end

function dder3(
    cone::EpiNormSpectral{T, T},
    pdir::AbstractVector{T},
    ddir::AbstractVector{T},
    ) where {T <: Real}
    point = cone.point
    d1 = inv_hess_prod!(zeros(T, cone.dim), ddir, cone)
    u = cone.point[1]
    W = cone.W
    dder3 = cone.dder3

    Zi = Hermitian(cone.Zi, :U)
    tempd1d1 = cone.tempd1d1
    trZi3 = sum(abs2, ldiv!(tempd1d1, cone.fact_Z.L, Zi))
    tau = cone.tau
    Zitau = cone.Zitau
    WtauI = cone.WtauI

    p = pdir[1]
    x = d1[1]
    r = pdir[2:end]
    z = d1[2:end]
    @views r_mat = vec_copyto!(cone.tempd1d2, r)
    @views z_mat = vec_copyto!(copy(cone.tempd1d2), z)

    fact_Z = cone.fact_Z
    Zi2W = fact_Z \ (fact_Z \ W)
    Zi3W = fact_Z \ Zi2W

    tauz = tau * z_mat'
    taur = tau * r_mat'
    rztau = taur' * z_mat + tauz' * r_mat
    rzWtauI = r_mat * WtauI * z_mat'
    temp0 = p * tauz' + x * taur'
    Ziz = fact_Z \ z_mat
    Zir = fact_Z \ r_mat
    temp1 = p * Ziz + x * Zir
    rzZi = r_mat' * Ziz
    dder3_mat = p * x * (8 * u^2 * Zi3W - 2 * Zi2W) +
        Zi * (-2u * (temp0 + temp0') + rzWtauI + rzWtauI') * tau +
        Zi * (-2u * temp1 + rztau) * WtauI +
        tau * (rzZi + rzZi') * WtauI +
        tau * (rztau' - 2u * temp1') * tau

    dder3[1] =
        p * x * (6 * u * cone.trZi2 - 8 * u^3 * trZi3 + (cone.d1 - 1) / u^3) +
        2 * dot(temp1, 4 * u^2 * Zitau - tau) +
        -2u * dot(z_mat,
        Zi * Zir * WtauI + (Zir * tau' + tau * Zir' + Zi * taur) * tau
        )


    @views vec_copyto!(dder3[2:end], dder3_mat)


    return dder3
end

function bar(cone::EpiNormSpectral{T, T}) where {T <: Real}
    (d1, d2) = (cone.d1, cone.d2)
    function barrier(uw)
        (u, w) = (uw[1], uw[2:end])
        W = reshape(w, d1, d2)
        return -logdet(Symmetric(abs2(u) * I - W * W', :U)) + (d1 - 1) * log(u)
    end
    return barrier
end
