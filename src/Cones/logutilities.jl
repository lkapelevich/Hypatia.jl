function omegawright(z::Real)
    if z < -2
        # TODO find out what others do for large negative z (unlikely in our case
        # because z is bounded by 1/d-log(d))
        t = exp(z)
        w = t * (1 + t * (-1 + t * (3 / 2 + t * (-8 / 3 + 125 / 24 * t))))
    elseif z < 1 + π
        # TODO factorize, Horner-like
        w = 1 + (z - 1) / 2 + (z - 1)^2 / 16 - (z - 1)^3 / 192 -
            (z - 1)^4 / 3072 + (z - 1)^5 * 13 / 61440
    else
        # TODO factorize, Horner-like
        w = z - log(z) + log(z) / z^2 * (log(z) / 2 - 1) + log(z) / z^3 * (
            (log(z))^2 / 3 - log(z) * 3 / 2 + 1)
    end
    r = z - w - log(w)
    # TODO more rigorous refinement
    # TODO compare to higher precision
    for k in 1:3
        t = (1 + w) * (1 + w + 2 / 3 * r)
        w = w * (1 + r / (1 + w) * (t - r / 2) / (t - r))
        r = (2 * w^2 - 8 * w - 1) / 72 / (1 + w)^6 * r^4
    end
    @show r
    @show z - w - log(w)
    return w
end
