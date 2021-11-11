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

    Z::Matrix{R}
    HuW::Matrix{R}
    Huu::T
    tempd1d2::Matrix{R}
    tempd1d2b::Matrix{R}
    tempd1d2c::Matrix{R}
    tempd1d2d::Matrix{R}
    tempd1d1::Matrix{R}
    tempd2d2::Matrix{R}
    tempd2d2b::Matrix{R}

    W::Matrix{R}
    fact_Z
    Zi::Matrix{R}
    tau::Matrix{R}
    WtauI::Matrix{R}
    Zitau::Matrix{R}

    W_svd
    dual_W_svd
    s::Vector{T}
    U::Matrix{R}
    Vt::Matrix{R}
    z::Vector{T}
    dual_z::T
    uzi::Vector{T}
    Urzi::Matrix{R}
    Vtsrzi::Matrix{R}
    cu::T
    zti1::T
    tzi::Vector{T}
    usti::Vector{T}
    zszidd::Matrix{T}
    zstidd::Matrix{T}

    w1::Matrix{R}
    w2::Matrix{R}
    w3::Matrix{R}
    s1::Vector{T}
    s2::Vector{T}
    U1::Matrix{R}
    U2::Matrix{R}
    U3::Matrix{R}

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
    # cone.tempd1d2b = zeros(R, d1, d2)
    # cone.tempd1d2c = zeros(R, d1, d2)
    # cone.tempd1d2d = zeros(R, d1, d2)
    cone.tempd1d1 = zeros(R, d1, d1)
    # cone.tempd2d2 = zeros(R, d2, d2)
    # cone.tempd2d2b = zeros(R, d2, d2)

    cone.z = zeros(T, d1)
    cone.uzi = zeros(T, d1)
    cone.Urzi = zeros(R, d1, d1)
    cone.Vtsrzi = zeros(R, d1, d2)
    cone.tzi = zeros(T, d1)
    cone.usti = zeros(T, d1)
    cone.zszidd = zeros(T, d1, d1)
    cone.zstidd = zeros(T, d1, d1)
    cone.w1 = zeros(R, d1, d2)
    cone.w2 = zeros(R, d1, d2)
    cone.w3 = zeros(R, d1, d2)
    cone.s1 = zeros(T, d1)
    cone.s2 = zeros(T, d1)
    cone.U1 = zeros(R, d1, d1)
    cone.U2 = zeros(R, d1, d1)
    cone.U3 = zeros(R, d1, d1)
    return cone
end

get_nu(cone::EpiNormSpectral) = 1 + cone.d1

function set_initial_point!(
    arr::AbstractVector{T},
    cone::EpiNormSpectral{T},
    ) where {T <: Real}
    arr .= 0
    arr[1] = sqrt(T(get_nu(cone)))
    return arr
end

function update_feas(cone::EpiNormSpectral{T}) where {T <: Real}
    # TODO speed up using norm bounds, cholesky of Z?
    @assert !cone.feas_updated
    u = cone.point[1]

    if u > eps(T)
        @views vec_copyto!(cone.w1, cone.point[2:end])
        cone.W_svd = svd(cone.w1, full = false) # TODO in place
        cone.is_feas = (u - maximum(cone.W_svd.S) > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::EpiNormSpectral{T}) where {T <: Real}
    u = cone.dual_point[1]
    if u > eps(T)
        W = @views vec_copyto!(cone.w1, cone.dual_point[2:end])
        cone.dual_W_svd = svd(W)
        cone.dual_z = u - sum(cone.dual_W_svd.S)
        return (cone.dual_z > eps(T))
    end
    return false
end

function update_grad(cone::EpiNormSpectral{T}) where T
    @assert cone.is_feas
    u = cone.point[1]
    U = cone.U = cone.W_svd.U
    Vt = cone.Vt = cone.W_svd.Vt
    s = cone.s = cone.W_svd.S
    z = cone.z
    uzi = cone.uzi
    s1 = cone.s1
    rz = cone.s2
    w1 = cone.w1
    g = cone.grad

    cone.cu = (cone.d1 - 1) / u
    @. z = (u - s) * (u + s)
    @. uzi = 2 * u / z
    @. rz = sqrt(z)
    @. s1 = inv(rz)
    mul!(cone.Urzi, U, Diagonal(s1))
    @. s1 = s / rz
    mul!(cone.Vtsrzi, Diagonal(s1), Vt)

    g[1] = cone.cu - sum(uzi)
    mul!(w1, cone.Urzi, cone.Vtsrzi, 2, false)
    @views vec_copyto!(g[2:end], w1)

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

function update_hess_aux(cone::EpiNormSpectral{T}) where T
    @assert !cone.hess_aux_updated
    @assert cone.grad_updated
    d1 = cone.d1
    u = cone.point[1]
    s = cone.s
    z = cone.z
    tzi = cone.tzi
    usti = cone.usti
    zszidd = cone.zszidd
    zstidd = cone.zstidd

    u2 = abs2(u)
    zti1 = one(u)
    @inbounds for j in 1:d1
        s_j = s[j]
        z_j = z[j]
        for i in 1:(j - 1)
            s_i = s[i]
            z_i = z[i]
            s_ij = s_i * s_j
            z_ij = u2 - s_ij
            t_ij = u2 + s_ij
            # zszidd and zstidd are nonsymmetric
            zszidd[i, j] = z_i / z_ij * s_j
            zszidd[j, i] = z_j / z_ij * s_i
            zstidd[i, j] = z_i / t_ij * s_j
            zstidd[j, i] = z_j / t_ij * s_i
        end
        t_j = u2 + abs2(s_j)
        zt_ij = z_j / t_j
        zti1 += zt_ij
        tzi[j] = t_j / z_j
        usti[j] = 2 * u * s_j / t_j
        zszidd[j, j] = s_j
        zstidd[j, j] = zt_ij * s_j
    end
    cone.zti1 = zti1

    cone.hess_aux_updated = true
end

function update_hess(cone::EpiNormSpectral)
    cone.hess_aux_updated || update_hess_aux(cone)
    isdefined(cone, :hess) || alloc_hess!(cone)
    d1 = cone.d1
    d2 = cone.d2
    u = cone.point[1]
    z = cone.z
    uzi = cone.uzi
    Urzi = cone.Urzi
    Vtsrzi = cone.Vtsrzi
    tzi = cone.tzi
    w1 = cone.w1
    w2 = cone.w2
    w3 = cone.w3
    U1 = cone.U1
    U2 = cone.U2
    H = cone.hess.data

    # u, u
    @inbounds H[1, 1] = -cone.cu / u + 2 * sum(tzi[i] / z[i] for i in 1:d1)

    # u, w
    mul!(U1, Urzi, Diagonal(uzi), -2, false)
    mul!(w1, U1, Vtsrzi)
    @views vec_copyto!(H[1, 2:end], w1)

    # w, w
    Urzit = copyto!(w1, Urzi') # accessing columns below
    c_idx = 2
    reim1s = (cone.is_complex ? [1, im] : [1,])
    @inbounds for j in 1:d2, i in 1:d1, reim1 in reim1s
        @views Urzi_i = Urzit[:, i]
        @views mul!(U1, Urzi_i, Vtsrzi[:, j]', reim1, false)
        @. U2 = U1 + U1'
        mul!(w2, Hermitian(U2, :U), Vtsrzi)
        @. @views w2[:, j] += reim1 * Urzi_i
        mul!(w3, Urzi, w2, 2, false)
        @views vec_copyto!(H[2:end, c_idx], w3)
        c_idx += 1
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiNormSpectral,
    )
    cone.hess_aux_updated || update_hess_aux(cone)
    d1 = cone.d1
    u = cone.point[1]
    z = cone.z
    Urzi = cone.Urzi
    Vtsrzi = cone.Vtsrzi
    cu = cone.cu
    tzi = cone.tzi
    w1 = cone.w1
    w2 = cone.w2
    U1 = cone.U1
    U2 = cone.U2
    Duzi = Diagonal(cone.uzi)

    @inbounds for j in 1:size(prod, 2)
        p = arr[1, j]
        @views vec_copyto!(w1, arr[2:end, j])
        mul!(w2, Urzi', w1)
        mul!(U1, w2, Vtsrzi')
        @. U2 = U1 + U1'

        prod[1, j] = -cu * p / u + 2 * sum((p * tzi[i] -
            u * real(U2[i, i])) / z[i] for i in 1:d1)

        @. U2 -= p * Duzi
        mul!(w2, Hermitian(U2, :U), Vtsrzi, true, true)
        mul!(w1, Urzi, w2, 2, false)
        @views vec_copyto!(prod[2:end, j], w1)
    end

    return prod
end

function update_inv_hess(cone::EpiNormSpectral{T}) where {T <: Real}
    cone.hess_aux_updated || update_hess_aux(cone)
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
    d1 = cone.d1
    d2 = cone.d2
    u = cone.point[1]
    U = cone.U
    Vt = cone.Vt
    z = cone.z
    zti1 = cone.zti1
    usti = cone.usti
    zszidd = cone.zszidd
    zstidd = cone.zstidd
    s1 = cone.s1
    w1 = cone.w1
    w2 = cone.w2
    w3 = cone.w3
    U1 = cone.U1
    U2 = cone.U2
    Hi = cone.inv_hess.data

    # u, u
    huu = u / zti1 * u
    Hi[1, 1] = huu

    # u, w
    @. s1 = huu * usti
    mul!(U1, U, Diagonal(s1))
    mul!(w1, U1, Vt)
    @views vec_copyto!(Hi[1, 2:end], w1)

    # w, w
    Ut = copyto!(w1, U') # accessing columns below
    c_idx = 2
    reim1s = (cone.is_complex ? [1, im] : [1,])
    @inbounds for j in 1:d2, i in 1:d1, reim1 in reim1s
        @views U_i = Ut[:, i]
        @views mul!(U1, U_i, Vt[:, j]', reim1, false)
        U1 .*= zszidd
        @. U2 = (U1 + U1') * zstidd
        mul!(w2, U2, Vt, -1, false)
        @. @views w2[:, j] += reim1 * z * U_i
        mul!(w3, U, w2, T(0.5), false)
        @views vec_copyto!(Hi[2:end, c_idx], w3)
        c_idx += 1
    end

    rthuu = sqrt(huu)
    @. s1 = rthuu * usti
    mul!(U1, U, Diagonal(s1))
    mul!(w1, U1, Vt)
    @views Hiuw2vec = Hi[2:end, 1]
    vec_copyto!(Hiuw2vec, w1)
    @views mul!(Hi[2:end, 2:end], Hiuw2vec, Hiuw2vec', true, true)

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiNormSpectral{T},
    ) where T
    cone.hess_aux_updated || update_hess_aux(cone)
    d1 = cone.d1
    u = cone.point[1]
    U = cone.U
    Vt = cone.Vt
    z = cone.z
    zti1 = cone.zti1
    usti = cone.usti
    zszidd = cone.zszidd
    zstidd = cone.zstidd
    w1 = cone.w1
    w2 = cone.w2
    U1 = cone.U1
    U2 = cone.U2
    Dusti = Diagonal(usti)

    @inbounds for j in 1:size(prod, 2)
        p = arr[1, j]
        @views vec_copyto!(w1, arr[2:end, j])
        mul!(w2, U', w1)
        mul!(U1, w2, Vt')

        c1 = u * (p + sum(usti[i] * real(U1[i, i]) for i in 1:d1)) / zti1 * u
        prod[1, j] = c1

        U1 .*= zszidd
        @. U2 = (U1 + U1') * zstidd - 2 * c1 * Dusti

        lmul!(Diagonal(z), w2)
        mul!(w2, U2, Vt, -1, true)
        mul!(w1, U, w2, T(0.5), false)
        @views vec_copyto!(prod[2:end, j], w1)
    end

    return prod
end

function dder3(cone::EpiNormSpectral{T}, dir::AbstractVector{T}) where T
    cone.hess_aux_updated || update_hess_aux(cone)
    d1 = cone.d1
    u = cone.point[1]
    z = cone.z
    uzi = cone.uzi
    Urzi = cone.Urzi
    Vtsrzi = cone.Vtsrzi
    w1 = cone.w1
    w2 = cone.w2
    s1 = cone.s1
    s2 = cone.s2
    U1 = cone.U1
    U2 = cone.U2
    U3 = cone.U3
    dder3 = cone.dder3
    Ds1 = Diagonal(s1)

    p = dir[1]
    @views vec_copyto!(w1, dir[2:end])

    @. s1 = p * uzi
    @. s2 = 2 * u * s1 - p

    mul!(w2, Urzi', w1)
    mul!(U3, w2, Vtsrzi')
    @. U1 = U3 + U3'

    mul!(U3, Diagonal(uzi), U1)
    @. U2 = U3 + U3'

    mul!(U3, w2, w2')
    mul!(U3, Hermitian(U1), U1, true, true)
    @inbounds tr1 = sum(real(U3[i, i]) * uzi[i] for i in 1:d1)

    @. U1 -= Ds1
    @inbounds tr2 = sum((s1[i] * (s2[i] + 2 * p) +
        2 * s2[i] * real(U1[i, i])) / z[i] for i in 1:d1)

    dder3[1] = -cone.cu * abs2(p / u) + tr1 - tr2

    @. s1 = s2 / z
    @. U3 += p * (Ds1 - U2)

    mul!(w1, Hermitian(U1, :U), w2)
    mul!(w1, Hermitian(U3, :U), Vtsrzi, true, true)
    mul!(w2, Urzi, w1, -2, false)
    @views vec_copyto!(dder3[2:end], w2)

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
    tau = cone.tau
    Zitau = cone.Zitau
    WtauI = cone.WtauI

    # from feas check
    @views vec_copyto!(W, cone.point[2:end])
    copyto!(cone.Z, abs2(u) * I)
    mul!(cone.Z, W, W', -1, true)
    cone.fact_Z = cholesky!(Hermitian(cone.Z, :U), check = false)
    # from grad
    Zi = cone.Zi
    ldiv!(tau, cone.fact_Z, cone.W)
    inv_fact!(Zi, cone.fact_Z)
    copytri!(Zi, 'U', true)
    # from hess aux
    copyto!(Zitau, tau)
    ldiv!(cone.fact_Z, Zitau)
    trZi2 = sum(abs2, cone.Zi)
    copyto!(WtauI, I)
    mul!(WtauI, cone.W', tau, true, true)

    Zi = Hermitian(cone.Zi, :U)
    tempd1d1 = cone.tempd1d1
    trZi3 = sum(abs2, ldiv!(tempd1d1, cone.fact_Z.L, Zi))

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
        p * x * (6 * u * trZi2 - 8 * u^3 * trZi3 + (cone.d1 - 1) / u^3) +
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
