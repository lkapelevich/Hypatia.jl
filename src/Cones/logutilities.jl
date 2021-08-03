function omegawright(z::T) where T
    if z < -2
        # TODO find out what others do for large negative z (unlikely in our case
        # because z is bounded by 1/d-log(d))
        t = exp(z)
        w = t * (1 + t * (-1 + t * (T(3) / 2 + t * (-T(8) / 3 + T(125) / 24 * t))))
    elseif z < 1 + π
        z1 = z - 1
        w = 1 + z1 / 2 * (1 + z1 / 8 * (1 + z1 / 12 * (-1 + z1 / 16 *
            (-1 + z1 * T(13) / 20))))
    else
        # TODO factorize, Horner-like
        lz = log(z)
        w = z - lz + lz / z + lz / z^2 * (lz / 2 - 1) + lz / z^3 * (
            (lz)^2 / 3 - lz * T(3) / 2 + 1)
    end
    r = z - w - log(w)
    for k in 1:2
        t = (1 + w) * (1 + w + 2 / 3 * r)
        w = w * (1 + r / (1 + w) * (t - r / 2) / (t - r))
        r = (2 * w^2 - 8 * w - 1) / 72 / (1 + w)^6 * r^4
    end
    # @show r
    # @show z - w - log(w)
    return w
end

using LambertW
function omegawright(z::BigFloat)
    return lambertw(exp(z))
end
