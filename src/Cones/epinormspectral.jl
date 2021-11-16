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
    inv_hess_aux_updated::Bool
    hess_fact_updated::Bool
    scal_hess_updated::Bool
    inv_scal_hess_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    scal_hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    inv_scal_hess::Symmetric{T, Matrix{T}}
    hess_fact_mat::Symmetric{T, Matrix{T}}
    scal_hess_fact_mat::Symmetric{T, Matrix{T}}
    hess_fact::Factorization{T}
    scal_hess_fact::Factorization{T}

    # TODO remove eventually
    W::Matrix{R}
    Z::Matrix{R}
    Zi::Matrix{R}
    tau::Matrix{R}
    WtauI::Matrix{R}
    Zitau::Matrix{R}
    tempd1d1::Matrix{R}

    W_svd
    s::Vector{T}
    U::Matrix{R}
    Vt::Matrix{R}
    mu::Vector{T}
    zeta::Vector{T}
    Urzi::Matrix{R}
    mrziVt::Matrix{R}
    cu::T
    Zu::T
    Uz::Matrix{R}
    sVt::Matrix{R}
    umzdd::Matrix{T}
    simdot::Matrix{T}

    w1::Matrix{R}
    w2::Matrix{R}
    s1::Vector{T}
    s2::Vector{T}
    U1::Matrix{R}
    U2::Matrix{R}
    U3::Matrix{R}
    VVt::Matrix{R}

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
    cone.hess_updated = cone.inv_hess_updated = cone.hess_aux_updated =
    cone.inv_hess_aux_updated = cone.hess_fact_updated =
    cone.dual_grad_updated = cone.scal_hess_updated =
    cone.inv_scal_hess_updated = false)

use_sqrt_scal_hess_oracles(::Int, cone::EpiNormSpectral) = false

function setup_extra_data!(
    cone::EpiNormSpectral{T, R},
    ) where {T <: Real, R <: RealOrComplex{T}}
    (d1, d2) = (cone.d1, cone.d2)
    cone.mu = zeros(T, d1)
    cone.zeta = zeros(T, d1)
    cone.Urzi = zeros(R, d1, d1)
    cone.mrziVt = zeros(R, d1, d2)
    cone.Uz = zeros(R, d1, d1)
    cone.sVt = zeros(R, d1, d2)
    cone.umzdd = zeros(T, d1, d1)
    cone.simdot = zeros(T, d1, d1)
    cone.w1 = zeros(R, d1, d2)
    cone.w2 = zeros(R, d1, d2)
    cone.s1 = zeros(T, d1)
    cone.s2 = zeros(T, d1)
    cone.U1 = zeros(R, d1, d1)
    cone.U2 = zeros(R, d1, d1)
    cone.U3 = zeros(R, d1, d1)
    # TODO remove
    cone.W = zeros(R, d1, d2)
    cone.Z = zeros(R, d1, d1)
    cone.Zi = zeros(R, d1, d1)
    cone.tau = zeros(R, d1, d2)
    cone.WtauI = zeros(R, d2, d2)
    cone.Zitau = zeros(R, d1, d2)
    cone.tempd1d1 = zeros(R, d1, d1)
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
    @assert !cone.feas_updated
    u = cone.point[1]
    cone.feas_updated = true
    cone.is_feas = false
    (u > eps(T)) || return false
    W = @views vec_copyto!(cone.w1, cone.point[2:end])

    # fast bounds:
    # spec <= frob <= spec * rtd1
    # opinf / rtd2 <= spec <= opinf * rtd1
    # op1 / rtd1 <= spec <= op1 * rtd2
    # spec <= sqrt(op1 * opinf)
    frob = norm(W, 2)
    op1 = opnorm(W, 1)
    opinf = opnorm(W, Inf)
    rtd1 = sqrt(T(cone.d1))
    rtd2 = sqrt(T(cone.d2))

    # lower bounds
    lb = max(frob / rtd1, opinf / rtd2, op1 / rtd1)
    (u - lb > eps(T)) || return false

    # upper bounds
    ub = min(frob, opinf * rtd1, op1 * rtd2, sqrt(op1 * opinf))
    if u - ub < eps(T)
        # use fast Cholesky-based feasibility check, rescale W*W' by inv(u)
        rtu = sqrt(u)
        Wrui = cone.w2
        @. Wrui = W / rtu
        Z = mul!(cone.U1, Wrui, Wrui', -1, false)
        @inbounds for i in 1:cone.d1
            Z[i, i] += u
        end
        isposdef(cholesky!(Hermitian(Z, :U), check = false)) || return false
    end

    # compute SVD and final feasibility check
    cone.W_svd = svd(W, full = false) # TODO in place
    cone.is_feas = (u - maximum(cone.W_svd.S) > eps(T))

    return cone.is_feas
end

function is_dual_feas(cone::EpiNormSpectral{T}) where {T <: Real}
    u = cone.dual_point[1]
    (u > eps(T)) || return false
    W = @views vec_copyto!(cone.w1, cone.dual_point[2:end])

    # fast bounds: frob <= nuc <= frob * rtd1
    frob = norm(W, 2)
    (u - sqrt(T(cone.d1)) * frob > eps(T)) && return true
    (u - frob > eps(T)) || return false

    # nuc = tr(sqrt(W*W')), rescale W*W' by inv(u)
    rtu = sqrt(u)
    W ./= rtu
    WWui = mul!(cone.U1, W, W')
    λ = eigvals!(Hermitian(WWui, :U))
    return (rtu - sum(sqrt(abs(λ_i)) for λ_i in λ) > eps(T))
end

function update_grad(cone::EpiNormSpectral{T}) where T
    @assert cone.is_feas
    u = cone.point[1]
    U = cone.U = cone.W_svd.U
    Vt = cone.Vt = cone.W_svd.Vt
    s = cone.s = cone.W_svd.S
    mu = cone.mu
    zeta = cone.zeta
    s1 = cone.s1
    w1 = cone.w1
    U1 = cone.U1
    g = cone.grad

    @. mu = s / u
    @. zeta = T(0.5) * (u - mu * s)
    cone.cu = (cone.d1 - 1) / u

    g[1] = cone.cu - sum(inv, zeta)

    @. s1 = mu / zeta
    mul!(U1, U, Diagonal(s1))
    mul!(w1, U1, Vt)
    @views vec_copyto!(g[2:end], w1)

    cone.grad_updated = true
    return cone.grad
end

function update_dual_grad(
    cone::EpiNormSpectral{T, R},
    ) where {T <: Real, R <: RealOrComplex{T}}
    u = cone.dual_point[1]
    W = @views vec_copyto!(cone.w1, cone.dual_point[2:end])
    dual_W_svd = svd(W)
    dual_zeta = u - sum(dual_W_svd.S)
    w = dual_W_svd.S

    (new_bound, zw2) = epinorminf_dg(u, w, cone.d1, dual_zeta)

    cone.dual_grad[1] = new_bound
    cone.dual_grad[2:end] .= vec(dual_W_svd.U * Diagonal(zw2) * dual_W_svd.Vt)

    cone.dual_grad_updated = true
    return cone.dual_grad
end

function update_hess_aux(cone::EpiNormSpectral)
    @assert !cone.hess_aux_updated
    @assert cone.grad_updated
    s1 = cone.s1
    s2 = cone.s2

    @. s1 = sqrt(cone.zeta)
    @. s2 = cone.mu / s1
    mul!(cone.mrziVt, Diagonal(s2), cone.Vt)
    @. s2 = inv(s1)
    mul!(cone.Urzi, cone.U, Diagonal(s2))

    cone.hess_aux_updated = true
end

function update_hess(cone::EpiNormSpectral{T, R}) where {T, R}
    cone.hess_aux_updated || update_hess_aux(cone)
    isdefined(cone, :hess) || alloc_hess!(cone)
    d1 = cone.d1
    d2 = cone.d2
    isdefined(cone, :VVt) || (cone.VVt = zeros(R, d2, d2))
    u = cone.point[1]
    zeta = cone.zeta
    Urzi = cone.Urzi
    mrziVt = cone.mrziVt
    s1 = cone.s1
    w1 = cone.w1
    U1 = cone.U1
    H = cone.hess.data

    # u, u
    ui = inv(u)
    H[1, 1] = sum((inv(z_i) - ui) / z_i for z_i in zeta) - cone.cu / u

    # u, w
    @. s1 = -inv(zeta)
    mul!(U1, Urzi, Diagonal(s1))
    mul!(w1, U1, mrziVt)
    @views vec_copyto!(H[1, 2:end], w1)

    # w, w
    Zi = mul!(U1, Urzi, Urzi')
    ZiW = mul!(w1, Urzi, mrziVt)
    WZiWI = mul!(cone.VVt, mrziVt', mrziVt, T(0.5), false)
    @inbounds for i in 1:d2
        WZiWI[i, i] += ui
    end

    idx_incr = (cone.is_complex ? 2 : 1)
    for i in 1:d2, j in 1:d1
        c_idx = r_idx = 2 + idx_incr * ((i - 1) * d1 + j - 1)
        @inbounds for k in i:d2
            ZiWjk = T(0.5) * ZiW[j, k]
            WZiWIik = WZiWI[i, k]
            lstart = (i == k ? j : 1)
            @inbounds for l in lstart:d1
                term1 = Zi[l, j] * WZiWIik
                term2 = ZiW[l, i] * ZiWjk
                spectral_kron_element!(H, r_idx, c_idx, term1, term2)
                c_idx += idx_incr
            end
        end
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiNormSpectral{T},
    ) where T
    cone.hess_aux_updated || update_hess_aux(cone)
    d1 = cone.d1
    u = cone.point[1]
    zeta = cone.zeta
    Urzi = cone.Urzi
    mrziVt = cone.mrziVt
    r = w1 = cone.w1
    simU = w2 = cone.w2
    sim = cone.U1
    S1 = cone.U2
    @views S1diag = S1[diagind(S1)]

    @inbounds for j in 1:size(prod, 2)
        p = arr[1, j]
        @views vec_copyto!(r, arr[2:end, j])

        pui = p / u
        mul!(simU, Urzi', r)
        mul!(sim, simU, mrziVt')
        @. S1 = T(0.5) * (sim + sim')
        @. S1diag -= p / zeta

        prod[1, j] = -sum((pui + real(S1[i, i])) / zeta[i] for i in 1:d1) -
            cone.cu * pui

        mul!(w2, Hermitian(S1, :U), mrziVt, true, inv(u))
        mul!(w1, Urzi, w2)
        @views vec_copyto!(prod[2:end, j], w1)
    end

    return prod
end

function update_inv_hess_aux(cone::EpiNormSpectral{T}) where T
    @assert !cone.inv_hess_aux_updated
    @assert cone.grad_updated
    u = cone.point[1]
    s = cone.s
    zeta = cone.zeta
    umzdd = cone.umzdd
    simdot = cone.simdot

    mul!(cone.sVt, Diagonal(s), cone.Vt)
    mul!(cone.Uz, cone.U, Diagonal(zeta))

    cone.Zu = -cone.cu + sum(inv, u - z_i for z_i in zeta)

    # umzdd = 0.5 * (u .+ mu * s')
    # simdot = zeta ./ (u .- mu * s')
    @inbounds for j in 1:cone.d1
        mu_j = cone.mu[j]
        z_j = zeta[j]
        for i in 1:(j - 1)
            mus_ij = mu_j * s[i]
            umzdd[i, j] = umzdd[j, i] = T(0.5) * (u + mus_ij)
            umus_ij = u - mus_ij
            simdot[i, j] = zeta[i] / umus_ij
            simdot[j, i] = z_j / umus_ij
        end
        umzdd[j, j] = u - z_j
        simdot[j, j] = T(0.5)
    end

    cone.inv_hess_aux_updated = true
end

function update_inv_hess(cone::EpiNormSpectral)
    cone.inv_hess_aux_updated || update_inv_hess_aux(cone)
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
    d1 = cone.d1
    d2 = cone.d2
    u = cone.point[1]
    U = cone.U
    zeta = cone.zeta
    umzdd = cone.umzdd
    Uz = cone.Uz
    sVt = cone.sVt
    s1 = cone.s1
    w1 = cone.w1
    w2 = cone.w2
    U1 = cone.U1
    U2 = cone.U2
    Hi = cone.inv_hess.data

    # u, u
    hiuu = Hi[1, 1] = u / cone.Zu

    # w, w
    Ut = copyto!(cone.U3, U')
    c_idx = 2
    @inbounds for j in 1:d2, i in 1:d1
        @views U_i = Ut[:, i]
        @views @. U1 = U_i * sVt[:, j]' / umzdd * cone.simdot

        @. U2 = U1 + U1'
        mul!(w2, U2, sVt, -1, false)
        @. @views w2[:, j] += u * U_i
        mul!(w1, Uz, w2)
        @views vec_copyto!(Hi[2:end, c_idx], w1)
        c_idx += 1

        if cone.is_complex
            U1 .*= im
            @. U2 = U1 + U1'
            mul!(w2, U2, sVt, -1, false)
            @. @views w2[:, j] += u * im * U_i
            mul!(w1, Uz, w2)
            @views vec_copyto!(Hi[2:end, c_idx], w1)
            c_idx += 1
        end
    end

    # u, w and w, w
    rthiuu = sqrt(hiuu)
    @inbounds for i in 1:d1
        s1[i] = rthiuu / umzdd[i, i]
    end
    mul!(U1, U, Diagonal(s1))
    mul!(w1, U1, sVt)
    @views Hiuwvec = Hi[1, 2:end]
    vec_copyto!(Hiuwvec, w1)
    @views mul!(Hi[2:end, 2:end], Hiuwvec, Hiuwvec', true, true)
    Hiuwvec .*= rthiuu

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiNormSpectral,
    )
    cone.inv_hess_aux_updated || update_inv_hess_aux(cone)
    d1 = cone.d1
    u = cone.point[1]
    zeta = cone.zeta
    umzdd = cone.umzdd
    sVt = cone.sVt
    r = w1 = cone.w1
    simU = w2 = cone.w2
    sim = cone.U1
    S1 = cone.U2
    @views S1diag = S1[diagind(S1)]

    @inbounds for j in 1:size(prod, 2)
        p = arr[1, j]
        @views vec_copyto!(r, arr[2:end, j])

        mul!(simU, cone.U', r)
        mul!(sim, simU, sVt')

        c1 = u * (p + sum(real(sim[i, i]) / umzdd[i, i] for i in 1:d1)) / cone.Zu
        prod[1, j] = c1

        sim .*= cone.simdot
        @. S1 = sim + sim'
        @. S1diag -= c1 / zeta
        S1 ./= umzdd

        mul!(w2, Hermitian(S1, :U), sVt, -1, u)
        mul!(w1, cone.Uz, w2)
        @views vec_copyto!(prod[2:end, j], w1)
    end

    return prod
end

# remove if becomes fallback
function inv_scal_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiNormSpectral{T, T},
    ) where {T <: Real}

    s = cone.point
    z = cone.dual_point
    ts = -dual_grad(cone)
    tz = -grad(cone)

    nu = get_nu(cone)
    cone_mu = dot(s, z) / nu
    tmu = dot(ts, tz) / nu

    ds = s - cone_mu * ts
    dz = z - cone_mu * tz
    Hts = hess_prod!(copy(ts), ts, cone)
    tol = sqrt(eps(T))
    # tol = 1000eps(T)
    if (norm(ds) < tol) || (norm(dz) < tol) || (cone_mu * tmu - 1 < tol) ||
        (abs(dot(ts, Hts) - nu * tmu^2) < tol)
        # @show "~~ skipping updates ~~"
        inv_hess_prod!(prod, arr, cone)
        prod ./= cone_mu
    else
        v1 = z + cone_mu * tz + dz / (cone_mu * tmu - 1)
        # TODO dot(ts, Hts) - nu * tmu^2 should be negative
        v2 = sqrt(cone_mu) * (Hts - tmu * tz) / sqrt(abs(dot(ts, Hts) - nu * tmu^2))

        c1 = 1 / sqrt(2 * cone_mu * nu)
        U = hcat(c1 * dz, c1 * v1, -v2)
        V = hcat(c1 * v1, c1 * dz, v2)'

        t1 = inv_hess_prod!(copy(arr), arr, cone) / cone_mu
        t2 = V * t1
        t3 = inv_hess_prod!(copy(U), U, cone) / cone_mu
        t4 = I + V * t3
        t5 = t4 \ t2
        t6 = U * t5
        t7 = inv_hess_prod!(copy(t6), t6, cone) / cone_mu
        prod .= t1 - t7
    end

    return prod
end

function dder3(cone::EpiNormSpectral{T}, dir::AbstractVector{T}) where T
    cone.hess_aux_updated || update_hess_aux(cone)
    u = cone.point[1]
    zeta = cone.zeta
    Urzi = cone.Urzi
    mrziVt = cone.mrziVt
    r = w1 = cone.w1
    simU = w2 = cone.w2
    sim = cone.U1
    S1 = cone.U2
    S2 = cone.U3
    @views S1diag = S1[diagind(S1)]
    @views S2diag = S2[diagind(S2)]
    dder3 = cone.dder3

    p = dir[1]
    @views vec_copyto!(r, dir[2:end])

    pui = p / u
    mul!(simU, Urzi', r)
    mul!(sim, simU, mrziVt')
    @. S1 = T(-0.5) * (sim + sim')
    @. S1diag += p / zeta

    mul!(S2, simU, simU', T(-0.5) / u, false)
    @. S2diag += T(0.5) * p / zeta * pui
    mul!(S2, Hermitian(S1, :U), S1, -1, true)

    @inbounds dder3[1] = -sum((real(S1[i, i]) * pui + real(S2[i, i])) / zeta[i]
        for i in 1:cone.d1) - cone.cu * abs2(pui)

    mul!(w1, Hermitian(S2, :U), mrziVt)
    mul!(w1, Hermitian(S1, :U), simU, inv(u), true)
    mul!(w2, Urzi, w1)
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
    fact_Z = cholesky!(Hermitian(cone.Z, :U), check = false)
    # from grad
    Zi = cone.Zi
    ldiv!(tau, fact_Z, cone.W)
    inv_fact!(Zi, fact_Z)
    copytri!(Zi, 'U', true)
    # from hess aux
    copyto!(Zitau, tau)
    ldiv!(fact_Z, Zitau)
    trZi2 = sum(abs2, cone.Zi)
    copyto!(WtauI, I)
    mul!(WtauI, cone.W', tau, true, true)

    Zi = Hermitian(cone.Zi, :U)
    tempd1d1 = cone.tempd1d1
    trZi3 = sum(abs2, ldiv!(tempd1d1, fact_Z.L, Zi))

    p = pdir[1]
    x = d1[1]
    r = pdir[2:end]
    z = d1[2:end]
    @views r_mat = vec_copyto!(zeros(T, cone.d1, cone.d2), r)
    @views z_mat = vec_copyto!(zeros(T, cone.d1, cone.d2), z)

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
