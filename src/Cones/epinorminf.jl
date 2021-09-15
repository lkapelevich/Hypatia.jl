"""
$(TYPEDEF)

Epigraph of real or complex infinity norm cone of dimension `dim`.

    $(FUNCTIONNAME){T, R}(dim::Int, use_dual::Bool = false)
"""
mutable struct EpiNormInf{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    n::Int
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
    scal_hess_updated::Bool
    inv_scal_hess_updated::Bool
    hess_aux_updated::Bool
    inv_hess_aux_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, SparseMatrixCSC{T, Int}}
    scal_hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    inv_scal_hess::Symmetric{T, Matrix{T}}
    # scal_hess_fact_mat::Symmetric{T, Matrix{T}} # TODO write inv scal hess
    # scal_hess_fact::Factorization{T}

    w::AbstractVector{R}
    den::AbstractVector{T}
    dual_den::T
    uden::Vector{R}
    wden::Vector{R}
    Huu::T
    Hure::Vector{T}
    Huim::Vector{T}
    Hrere::Vector{T}
    Hreim::Vector{T}
    Himim::Vector{T}
    Hiure::Vector{T}
    Hiuim::Vector{T}
    schur::T
    idet::Vector{T}

    function EpiNormInf{T, R}(
        dim::Int;
        use_dual::Bool = false,
        ) where {T <: Real, R <: RealOrComplex{T}}
        @assert dim >= 2
        cone = new{T, R}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        cone.is_complex = (R <: Complex)
        cone.n = (cone.is_complex ? div(dim - 1, 2) : dim - 1)
        return cone
    end
end

reset_data(cone::EpiNormInf) = (cone.feas_updated = cone.grad_updated =
    cone.dual_grad_updated =
    cone.hess_updated = cone.inv_hess_updated = cone.hess_aux_updated =
    cone.inv_hess_aux_updated =
    cone.scal_hess_updated = cone.inv_scal_hess_updated = false)

use_sqrt_hess_oracles(::Int, cone::EpiNormInf) = false

use_sqrt_scal_hess_oracles(::Int, cone::EpiNormInf{T, T}, ::T) where {T <: Real} = false

function setup_extra_data!(
    cone::EpiNormInf{T, R},
    ) where {T <: Real, R <: RealOrComplex{T}}
    n = cone.n
    cone.w = zeros(R, n)
    cone.wden = zeros(R, n)
    cone.den = zeros(T, n)
    cone.uden = zeros(R, n)
    cone.Hure = zeros(T, n)
    cone.Hrere = zeros(T, n)
    cone.Hiure = zeros(T, n)
    if cone.is_complex
        cone.Huim = zeros(T, n)
        cone.Hreim = zeros(T, n)
        cone.Himim = zeros(T, n)
        cone.Hiuim = zeros(T, n)
        cone.idet = zeros(T, n)
    end
    cone.dual_grad = zeros(R, cone.dim)
    return cone
end

get_nu(cone::EpiNormInf) = cone.n + 1

function set_initial_point!(
    arr::AbstractVector{T},
    cone::EpiNormInf{T},
    ) where {T <: Real}
    arr .= 0
    arr[1] = sqrt(T(get_nu(cone)))
    return arr
end

function update_feas(cone::EpiNormInf{T}) where T
    @assert !cone.feas_updated
    u = cone.point[1]
    @views vec_copyto!(cone.w, cone.point[2:end])

    cone.is_feas = (u > eps(T) && u - norm(cone.w, Inf) > eps(T))

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::EpiNormInf{T}) where T
    dp = cone.dual_point
    u = dp[1]

    if u > eps(T)
        if cone.is_complex
            @inbounds norm1 = sum(hypot(dp[2i], dp[2i + 1]) for i in 1:cone.n)
        else
            @views norm1 = norm(dp[2:end], 1)
        end
        cone.dual_den = u - norm1
        return (cone.dual_den > eps(T))
    end

    return false
end

function update_grad(
    cone::EpiNormInf{T, R},
    ) where {T <: Real, R <: RealOrComplex{T}}
    @assert cone.is_feas
    u = cone.point[1]
    w = cone.w
    den = cone.den

    usqr = abs2(u)
    @. den = usqr - abs2(w)
    den .*= T(0.5)
    @. cone.uden = u / den
    @. cone.wden = w / den
    cone.grad[1] = (cone.n - 1) / u - sum(cone.uden)
    @views vec_copyto!(cone.grad[2:end], cone.wden)

    cone.grad_updated = true
    return cone.grad
end

# TODO refactor with power cones
function update_dual_grad(
    cone::EpiNormInf{T, R},
    ) where {T <: Real, R <: RealOrComplex{T}}
    u = cone.dual_point[1]
    w = vec_copyto!(copy(cone.w), cone.dual_point[2:end])
    n = cone.n

    h(y) = u * y + sum(sqrt(1 + abs2(w[i]) * y^2) for i in 1:n) + 1
    hp(y) = u + sum(abs2(w[i]) * y * (1 + abs2(w[i]) * y^2)^(-1/2) for i in 1:n)
    hpp(y) = sum(abs2(w[i]) / (1 + abs2(w[i]) * y^2)^(3 / 2) for i in 1:n)

    lower_bound = -(n + 1) / cone.dual_den
    upper_bound = -inv(cone.dual_den)
    C = hpp(upper_bound) / hp(lower_bound) / 2
    gap = upper_bound - lower_bound
    while C * gap > 1
        new_bound = (lower_bound + upper_bound) / 2
        if h(new_bound) > 0
            upper_bound = new_bound
        else
            lower_bound = new_bound
        end
        C = hpp(upper_bound) / hp(lower_bound) / 2
        @assert lower_bound < upper_bound < 0
        gap = upper_bound - lower_bound
        # @show gap
    end

    new_bound = (lower_bound + upper_bound) / 2
    iter = 0
    while abs(h(new_bound)) > 1000eps(T)
        # new_bound -= h(new_bound) / hp(new_bound)
        new_bound -= h(BigFloat(new_bound)) / hp(BigFloat(new_bound))
        iter += 1
        # @show iter
    end
    new_bound = Float64(new_bound)

    cone.dual_grad[1] = new_bound
    # z * w / 2
    zw2 = copy(w)
    for i in eachindex(w)
        if abs(w[i]) .< 100eps(T)
            zw2[i] = new_bound^2 * w[i] / 2
        else
            zw2[i] = (-1 + sqrt(1 + abs2(w[i]) * new_bound^2)) / w[i]
        end
    end
    @views vec_copyto!(cone.dual_grad[2:end], zw2)

    cone.dual_grad_updated = true
    return cone.dual_grad
end

function update_hess_aux(cone::EpiNormInf{T}) where {T <: Real}
    @assert cone.grad_updated
    u = cone.point[1]
    w = cone.w
    uden = cone.uden

    @inbounds for (j, wdenj) in enumerate(cone.wden)
        udenj = uden[j]
        invdenj = inv(cone.den[j])
        if cone.is_complex
            (wdre, w_dim) = reim(wdenj)
            cone.Hure[j] = -wdre * udenj
            cone.Huim[j] = -w_dim * udenj
            cone.Hrere[j] = abs2(wdre) + invdenj
            cone.Himim[j] = abs2(w_dim) + invdenj
            cone.Hreim[j] = wdre * w_dim
        else
            cone.Hure[j] = -wdenj * udenj
            cone.Hrere[j] = abs2(wdenj) + invdenj
        end
    end
    cone.Huu = sum(abs2, uden) - ((cone.n - 1) / u + sum(uden)) / u

    cone.hess_aux_updated = true
    return
end

function alloc_hess!(cone::EpiNormInf{T, T}) where {T <: Real}
    # initialize sparse idxs for upper triangle of Hessian
    dim = cone.dim
    nnz_tri = 2 * dim - 1
    I = Vector{Int}(undef, nnz_tri)
    J = Vector{Int}(undef, nnz_tri)
    idxs1 = 1:dim
    @views I[idxs1] .= 1
    @views J[idxs1] .= idxs1
    idxs2 = (dim + 1):(2 * dim - 1)
    @views I[idxs2] .= 2:dim
    @views J[idxs2] .= 2:dim
    V = ones(T, nnz_tri)
    cone.hess = Symmetric(sparse(I, J, V, dim, dim), :U)
    return
end

function alloc_hess!(cone::EpiNormInf{T, Complex{T}}) where {T <: Real}
    # initialize sparse idxs for upper triangle of Hessian
    dim = cone.dim
    nnz_tri = 2 * dim - 1 + cone.n
    I = Vector{Int}(undef, nnz_tri)
    J = Vector{Int}(undef, nnz_tri)
    idxs1 = 1:dim
    @views I[idxs1] .= 1
    @views J[idxs1] .= idxs1
    idxs2 = (dim + 1):(2 * dim - 1)
    @views I[idxs2] .= 2:dim
    @views J[idxs2] .= 2:dim
    idxs3 = (2 * dim):nnz_tri
    @views I[idxs3] .= 2:2:dim
    @views J[idxs3] .= 3:2:dim
    V = ones(T, nnz_tri)
    cone.hess = Symmetric(sparse(I, J, V, dim, dim), :U)
    return
end

function update_hess(cone::EpiNormInf{T, T}) where {T <: Real}
    cone.hess_aux_updated || update_hess_aux(cone)
    isdefined(cone, :hess) || alloc_hess!(cone)

    # modify nonzeros of upper triangle of Hessian
    nzval = cone.hess.data.nzval
    nzval[1] = cone.Huu
    nz_idx = 2
    @inbounds for i in 1:cone.n
        nzval[nz_idx] = cone.Hure[i]
        nzval[nz_idx + 1] = cone.Hrere[i]
        nz_idx += 2
    end

    cone.hess_updated = true
    return cone.hess
end

function update_hess(cone::EpiNormInf{T, Complex{T}}) where {T <: Real}
    cone.hess_aux_updated || update_hess_aux(cone)
    isdefined(cone, :hess) || alloc_hess!(cone)

    # modify nonzeros of upper triangle of Hessian
    nzval = cone.hess.data.nzval
    nzval[1] = cone.Huu
    nz_idx = 1
    @inbounds for i in 1:cone.n
        @. nzval[nz_idx .+ (1:5)] = (cone.Hure[i], cone.Hrere[i],
            cone.Huim[i], cone.Hreim[i], cone.Himim[i])
        nz_idx += 5
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiNormInf{T, T},
    ) where {T <: Real}
    cone.hess_aux_updated || update_hess_aux(cone)

    @inbounds @views begin
        u_arr = arr[1, :]
        w_arr = arr[2:end, :]
        u_prod = prod[1, :]
        w_prod = prod[2:end, :]
        copyto!(u_prod, u_arr)
        mul!(u_prod, w_arr', cone.Hure, true, cone.Huu)
        mul!(w_prod, cone.Hure, u_arr')
        @. w_prod += cone.Hrere * w_arr
    end

    return prod
end

function hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiNormInf{T, Complex{T}},
    ) where {T <: Real}
    cone.hess_aux_updated || update_hess_aux(cone)

    @inbounds @views begin
        u_arr = arr[1, :]
        v_arr = arr[2:2:end, :]
        w_arr = arr[3:2:end, :]
        u_prod = prod[1, :]
        v_prod = prod[2:2:end, :]
        w_prod = prod[3:2:end, :]
        @. u_prod = cone.Huu * u_arr
        mul!(u_prod, v_arr', cone.Hure, true, true)
        mul!(u_prod, w_arr', cone.Huim, true, true)
        mul!(v_prod, cone.Hure, u_arr')
        mul!(w_prod, cone.Huim, u_arr')
        @. v_prod += cone.Hrere * v_arr + cone.Hreim * w_arr
        @. w_prod += cone.Hreim * v_arr + cone.Himim * w_arr
    end

    return prod
end

function update_inv_hess_aux(cone::EpiNormInf{T}) where {T <: Real}
    cone.hess_aux_updated || update_hess_aux(cone)
    @assert !cone.inv_hess_aux_updated
    wden = cone.wden
    u = cone.point[1]

    usqr = abs2(u)
    schur = (1 - cone.n) / usqr
    @inbounds for (j, wj) in enumerate(cone.w)
        u2pwj2 = T(0.5) * (usqr + abs2(wj))
        iedge = u / u2pwj2 * wj
        if cone.is_complex
            (cone.Hiure[j], cone.Hiuim[j]) = reim(iedge)
        else
            cone.Hiure[j] = iedge
        end
        schur += inv(u2pwj2)
    end
    cone.schur = schur
    if schur < zero(T)
        @warn("bad schur $schur")
    end

    if cone.is_complex
        @. cone.idet = cone.Hrere * cone.Himim - abs2(cone.Hreim)
    end

    cone.inv_hess_aux_updated = true
    return
end

function update_inv_hess(cone::EpiNormInf{T}) where {T <: Real}
    cone.inv_hess_aux_updated || update_inv_hess_aux(cone)
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
    Hi = cone.inv_hess.data
    wden = cone.wden
    u = cone.point[1]
    schur = cone.schur

    Hi[1, 1] = inv(schur)
    @inbounds for j in 1:cone.n
        if cone.is_complex
            Hi[2j, 1] = cone.Hiure[j]
            Hi[2j + 1, 1] = cone.Hiuim[j]
        else
            Hi[j + 1, 1] = cone.Hiure[j]
        end
    end
    @. Hi[1, 2:end] = Hi[2:end, 1] / schur

    @inbounds for j in 2:cone.dim, i in 2:j
        Hi[i, j] = Hi[j, 1] * Hi[1, i]
    end

    if cone.is_complex
        @inbounds for j in 1:cone.n
            detj = cone.idet[j]
            vj = 2j
            wj = vj + 1
            Hi[vj, vj] += cone.Himim[j] / detj
            Hi[wj, wj] += cone.Hrere[j] / detj
            Hi[vj, wj] -= cone.Hreim[j] / detj
        end
    else
        @inbounds for (j, rerej) in enumerate(cone.Hrere)
            vj = j + 1
            Hi[vj, vj] += inv(rerej)
        end
    end

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiNormInf{T, T},
    ) where {T <: Real}
    cone.inv_hess_aux_updated || update_inv_hess_aux(cone)

    @inbounds @views begin
        w_arr = arr[2:end, :]
        u_prod = prod[1, :]
        copyto!(u_prod, arr[1, :])
        mul!(u_prod, w_arr', cone.Hiure, true, true)
        u_prod ./= cone.schur
        @. prod[2:end, :] = cone.Hiure * u_prod' + w_arr / cone.Hrere
    end

    return prod
end

function inv_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiNormInf{T, Complex{T}},
    ) where {T <: Real}
    cone.inv_hess_aux_updated || update_inv_hess_aux(cone)

    @inbounds @views begin
        u_prod = prod[1, :]
        re_arr = arr[2:2:end, :]
        im_arr = arr[3:2:end, :]
        copyto!(u_prod, arr[1, :])
        mul!(u_prod, re_arr', cone.Hiure, true, true)
        mul!(u_prod, im_arr', cone.Hiuim, true, true)
        u_prod ./= cone.schur
        @. prod[2:2:end, :] = cone.Hiure * u_prod' +
            (cone.Himim * re_arr - cone.Hreim * im_arr) / cone.idet
        @. prod[3:2:end, :] = cone.Hiuim * u_prod' +
            (cone.Hrere * im_arr - cone.Hreim * re_arr) / cone.idet
    end

    return prod
end

# remove if becomes fallback
function inv_scal_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiNormInf{T, T},
    mu::T,
    ) where {T <: Real}

    hess(cone)

    old_hess = copy(cone.hess)
    s = cone.point
    z = cone.dual_point
    ts = -dual_grad(cone)
    tz = -grad(cone)

    nu = get_nu(cone)
    cone_mu = dot(s, z) / nu
    tmu = dot(ts, tz) / nu

    ds = s - cone_mu * ts
    dz = z - cone_mu * tz
    Hts = old_hess * ts
    tol = sqrt(eps(T))
    # tol = 1000eps(T)
    if (norm(ds) < tol) || (norm(dz) < tol) || (cone_mu * tmu - 1 < tol) ||
        (abs(dot(ts, Hts) - nu * tmu^2) < tol)
        # @show "~~ skipping updates ~~"
        inv_hess_prod!(prod, arr, cone)
        prod ./= mu
    else
        v1 = z + cone_mu * tz + dz / (cone_mu * tmu - 1)
        v2 = Hts - tmu * tz

        c1 = 1 / sqrt(2 * cone_mu * nu)
        c2 = sqrt(cone_mu / (dot(ts, Hts) - nu * tmu^2))
        U = hcat(c1 * dz, c1 * v1, -c2 * v2)
        V = hcat(c1 * v1, c1 * dz, c2 * v2)'

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

function dder3(
    cone::EpiNormInf{T},
    dir::AbstractVector{T},
    ) where {T <: Real}
    @assert cone.grad_updated
    u = cone.point[1]
    udir = dir[1]
    dder3 = cone.dder3

    u3 = T(1.5) / u
    udu = udir / u
    dder3[1] = -udir * sum(z * (u3 - z) * z for z in cone.uden) * udir -
        udu * (cone.n - 1) / u * udu

    @inbounds for i in 1:cone.n
        deni = -4 * cone.den[i]
        udeni = 2 * cone.uden[i]
        suuw = udir * (-1 + udeni * u)
        wi = cone.w[i]
        wdeni = 2 * cone.wden[i]

        if cone.is_complex
            (wdenire, wdeniim) = reim(wdeni)
            (wire, wiim) = reim(wi)
            (dire, diim) = (dir[2i], dir[2i + 1])
            uuwre = suuw * wdenire
            uuwim = suuw * wdeniim
            uimimre = 1 + wdenire * wire
            uimimim = 1 + wdeniim * wiim
            uimimrere = -udeni * uimimre * dire
            uimimimim = -udeni * uimimim * diim
            uimimimre = -udeni * wdeniim * wire
            imimwrerere = wdenire * (2 + uimimre)
            imimwimimim = wdeniim * (2 + uimimim)
            imimwrereim = wdeniim * uimimre * dire
            imimwimimre = wdenire * uimimim * diim

            dder3[1] += (2 * (uuwre * dire + uuwim * diim) + uimimrere * dire +
                uimimimim * diim + 2 * uimimimre * diim * dire) / deni
            dder3[2i] = (udir * (2 * (uimimrere + uimimimre * diim) + uuwre) +
                (abs2(dire) * imimwrerere + diim *
                (2 * imimwrereim + imimwimimre))) / deni
            dder3[2i + 1] = (udir * (2 * (uimimimim + uimimimre * dire) + uuwim) +
                (abs2(diim) * imimwimimim + dire *
                (2 * imimwimimre + imimwrereim))) / deni
        else
            di = dir[1 + i]
            uuw = suuw * wdeni
            uimim = 1 + wdeni * wi
            uimim2 = -udeni * uimim * di

            dder3[1] += di * (2 * uuw + uimim2) / deni
            dder3[1 + i] = (udir * (uuw + 2 * uimim2) +
                di * wdeni * (2 + uimim) * di) / deni
        end
    end

    return dder3
end

function dder3(
    cone::EpiNormInf{T},
    pdir::AbstractVector{T},
    ddir::AbstractVector{T},
    ) where {T <: Real}
    dder3 = cone.dder3
    point = cone.point
    d1 = inv_hess_prod!(zeros(T, cone.dim), ddir, cone)
    d2 = zero(d1)
    d2[1] -= (cone.n - 1) / point[1]^2 * d1[1]

    # do SOC for each i
    for i in 2:(cone.n + 1)
        dist = (abs2(point[1]) - abs2(point[i])) / 2
        d2[1] += (abs2(point[1]) + abs2(point[i])) / abs2(dist) * d1[1] / 2
        d2[1] += -point[1] * point[i] * d1[i] / abs2(dist)
        d2[i] = -point[1] * point[i] / abs2(dist) * d1[1] +
            point[i] * dot(point[i], d1[i]) / abs2(dist) + d1[i] / dist

        @views jdot_p_s = point[1] * pdir[1] - point[i] * pdir[i]
        @. dder3[[1, i]] = jdot_p_s * ddir[[1, i]]
        dot_s_z = pdir[1] * ddir[1] + pdir[i] * ddir[i]
        dot_p_z = point[1] * ddir[1] + point[i] * ddir[i]
        dder3[1] += dot_s_z * point[1] - dot_p_z * pdir[1]
        dder3[i] += -dot_s_z * point[i] + dot_p_z * pdir[i]
        dder3[[1, i]] ./= -dist * 2
    end
    # @show d2 ./ ddir
    u = point[1]
    w = cone.w
    # H = zero(cone.hess).data
    z = cone.den * 2
    #
    #
    # # H[1, 1] = sum(pdir[2:end] .* 4w .* (4u^2 ./ z.^3 - 1 ./ z.^2)) +
    # #     pdir[1] * (sum(4u ./ z.^3 .* (3z .- 4u^2)) + 2 * (cone.n - 1) / u^3)
    # H[1, 1] = pdir[1] * 2 * (cone.n - 1) / u^3 + sum(
    #     16 * (pdir[2:end] .* w * u^2 .- u^3 * pdir[1]) ./ z.^3 +
    #     4 * (-pdir[2:end] .* w .+ 3u * pdir[1]) ./ z.^2
    #     )
    # H[1, 2:end] .= 16 * (pdir[1] * w * u^2 - pdir[2:end] * u .* w.^2) ./ z.^3 +
    #     4 * (-pdir[1] * w - pdir[2:end] * u) ./ z.^2
    # H[2:end, 2:end] = Diagonal(
    #     16 * w.^2 .* (-pdir[1] * u .+ pdir[2:end] .* w) ./ z.^3 +
    #     4 * (-pdir[1] * u .+ 3w .* pdir[2:end]) ./ z.^2
    #     )

    dder3[1] = pdir[1] * 2 * (cone.n - 1) / u^3 * d1[1] +
        sum(
        16u * (
            (-d1[1] * u .+ d1[2:end] .* w) .* (pdir[1] * u .- pdir[2:end] .* w)
        ) ./ z.^3 +
        4 * (
            u * (d1[1] * pdir[1] * 3 .- pdir[2:end] .* d1[2:end]) +
            w .* (-d1[1] .* pdir[2:end] - pdir[1] * d1[2:end] )
            # d1[1] * -pdir[2:end] .* w .+ d1[1] * 3u * pdir[1] +
            # -pdir[1] * w .* d1[2:end] - pdir[2:end] .* d1[2:end] * u
            ) ./ z.^2
        )

    dder3[2:end] .= (
        16 * w .* (
            (d1[1] * u .- d1[2:end] .* w) .* (pdir[1] * u .- pdir[2:end] .* w)
            ) ./ z.^3 +
        4 * (
            -pdir[1] * w * d1[1] - pdir[2:end] * u * d1[1] +
            -pdir[1] .* d1[2:end] * u .+ 3w .* pdir[2:end] .* d1[2:end]
            ) ./ z.^2
        )
    dder3 ./= 2
    # @show Symmetric(H) ./ ()

    #
    # d1 = inv_hess_prod!(zeros(T, cone.dim), ddir, cone)
    # barrier = bar(cone)
    # bardir(point, s, t) = barrier(point + s * d1 + t * pdir)
    # true_deriv = ForwardDiff.gradient(
    #     s2 -> ForwardDiff.derivative(
    #         s -> ForwardDiff.derivative(
    #             t -> bardir(s2, s, t),
    #             0),
    #         0),
    #     cone.point) / 2
    # @show true_deriv ./ dder3


    return dder3
end

function bar(cone::EpiNormInf{T, T}) where {T <: Real}
    function barrier(uw)
        (u, w) = (uw[1], uw[2:end])
        d = length(w)
        return -sum(wi -> log(u^2 - abs2(wi)), w) + (d - 1) * log(u)
    end
    return barrier
end

# TODO remove this in favor of new hess_nz_count etc functions
# that directly use uu, uw, ww etc
hess_nz_count(cone::EpiNormInf{<:Real, <:Real}) =
    3 * cone.dim - 2

hess_nz_count(cone::EpiNormInf{<:Real, <:Complex}) =
    3 * cone.dim - 2 + 2 * cone.n

hess_nz_count_tril(cone::EpiNormInf{<:Real, <:Real}) =
    2 * cone.dim - 1

hess_nz_count_tril(cone::EpiNormInf{<:Real, <:Complex}) =
    2 * cone.dim - 1 + cone.n

hess_nz_idxs_col(cone::EpiNormInf{<:Real, <:Real}, j::Int) =
    (j == 1 ? (1:cone.dim) : [1, j])

hess_nz_idxs_col(cone::EpiNormInf{<:Real, <:Complex}, j::Int) =
    (j == 1 ? (1:cone.dim) : (iseven(j) ? [1, j, j + 1] : [1, j - 1, j]))

hess_nz_idxs_col_tril(cone::EpiNormInf{<:Real, <:Real}, j::Int) =
    (j == 1 ? (1:cone.dim) : [j])

hess_nz_idxs_col_tril(cone::EpiNormInf{<:Real, <:Complex}, j::Int) =
    (j == 1 ? (1:cone.dim) : (iseven(j) ? [j, j + 1] : [j]))
