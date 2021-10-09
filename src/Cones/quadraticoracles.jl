# for the problematic problems np is big, q is 3
# but the oracles have >>3 operations that are np^2
# so they are slower than other methods

use_quadratic_oracles(::HypoPerLog) = true

function quadratic_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat,
    cone::HypoPerLog{T},
    ) where {T <: Real}
    np = size(arr, 1)
    sd = cone.dim - 2
    temp_np_np = zeros(T, np, np)
    temp_sd_np = zeros(T, sd, np)
    temp_np = zeros(T, np)
    temp_np_2 = zeros(T, np)
    ζ = cone.ζ
    ϕ = cone.ϕ
    v = cone.point[2]
    @views w = cone.point[3:end]
    wi = inv.(w)
    d = length(w)
    σ = ϕ - d
    vζi = v / ζ

    @views A_u = arr[:, 1]
    @views A_v = arr[:, 2]
    @views A_w = arr[:, 3:end]

    # uu
    @. temp_np = A_u / ζ
    mul!(prod, temp_np, temp_np')

    # uv
    @. temp_np_2 = A_v * -σ / ζ
    mul!(temp_np_np, temp_np, temp_np_2')
    prod .+= temp_np_np
    # TODO hmm ponder
    # mul!(temp_np_np, temp_np_2, temp_np')
    # prod .+= temp_np_np
    prod .+= copy(temp_np_np')

    # vv
    c1 = inv(v) / v + (d / v + abs2(σ) / ζ) / ζ
    mul!(prod, A_v, A_v', c1, true)

    # uw
    @views mul!(temp_np, A_w, wi)
    mul!(temp_np_np, temp_np, A_u', -vζi / ζ, false)
    @. prod += temp_np_np
    # TODO hmm ponder
    mul!(temp_np_np, A_u, temp_np', -vζi / ζ, false)
    @. prod += temp_np_np
    # prod .+= copy(temp_np_np')

    # vw
    mul!(temp_np_np, temp_np, A_v', (σ * vζi - 1) / ζ, false)
    @. prod += temp_np_np
    # TODO hmm ponder
    mul!(temp_np_np, A_v, temp_np', (σ * vζi - 1) / ζ, false)
    @. prod += temp_np_np
    # prod .+= copy(temp_np_np')

    # ww- 1
    @. temp_np *= vζi
    mul!(prod, temp_np, temp_np', true, true)

    # ww- 2
    @. @views temp_sd_np = A_w' / w
    mul!(prod, temp_sd_np', temp_sd_np, vζi + 1, true)

    return prod
end

use_quadratic_oracles(::HypoGeoMean) = true


function quadratic_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat,
    cone::HypoGeoMean{T},
    ) where {T <: Real}
    np = size(arr, 1)
    sd = cone.dim - 1
    temp_np_np = zeros(T, np, np)
    temp_sd_np = zeros(T, sd, np)
    temp_np = zeros(T, np)
    ζ = cone.ζ
    ϕ = cone.ϕ
    u = cone.point[1]
    @views w = cone.point[2:end]
    wi = inv.(w)
    di = cone.di
    ϕdiζ = ϕ * di / ζ

    @views A_u = arr[:, 1]
    @views A_w = arr[:, 2:end]

    @. temp_np = A_u / ζ
    mul!(prod, temp_np, temp_np')

    @views mul!(temp_np, A_w, wi)
    mul!(temp_np_np, temp_np, A_u', -ϕdiζ / ζ, false)
    @. prod += temp_np_np
    @. prod += temp_np_np'

    @. temp_np *= di / ζ
    mul!(prod, temp_np, temp_np', ϕ * u, true)

    @. @views temp_sd_np = A_w' / w
    mul!(prod, temp_sd_np', temp_sd_np, ϕdiζ + 1, true)

    return prod
end

use_quadratic_oracles(::Cone) = false

function quadratic_scal_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat,
    cone::Cone{T},
    mu::T
    ) where {T <: Real}
    quadratic_hess_prod!(prod, arr, cone)
    temp_np_1 = zeros(T, size(prod, 1))
    temp_np_2 = zeros(T, size(prod, 1))

    rtmu = sqrt(mu)
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
    if (norm(ds) < tol) || (norm(dz) < tol) || (cone_mu * tmu - 1 < tol) ||
        (abs(dot(ts, Hts) - nu * tmu^2) < tol)
        prod .*= mu
    else
        v1 = z + cone_mu * tz + dz / (cone_mu * tmu - 1)
        v2 = Hts - tmu * tz
        M1 = dz * v1'
        prod .*= cone_mu
        tsHts = dot(ts, Hts)

        mul!(temp_np_1, arr, v1)
        mul!(temp_np_2, arr, dz)
        prod .+= temp_np_1 * temp_np_2' / (2 * cone_mu * nu)
        prod .+= temp_np_2 * temp_np_1' / (2 * cone_mu * nu)
        mul!(temp_np_1, arr, v2)
        prod .-= temp_np_1 * temp_np_1' * cone_mu / (tsHts - nu * tmu^2)
    end

    return prod
end
