
# approximation algorithm from ``Algorithm 917: Complex Double-Precision
# Evaluation of the Wright ω Function'' by Piers W. Laurence, Robert M. Corless
# and David J. Jeffrey
# Series approximations can be found in the book chapter ``The Wright ω Function''
# by Corless, R. M. and Jeffrey, D. J.

# need to be in the right type or rationals
# consts_1 = # annoying
# consts_infty = [(-1)^i // factorial(j) * length(Combinatorics.partitions(i + j, i + 1)) for j in 1:20 for i in 1:j] # wrong
# consts_neginfty = [(-i)^(i - 1) / factorial(i) for i in 1:20]

function omegawright(z::T) where T
    z = float(z)
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
        lz = log(z)
        w = z + lz * (-1 + (1 + (lz / 2 - 1 + (lz * (lz / 3 - T(3) / 2) + 1) / z) / z) / z)
    end
    r = z - w - log(w)
    for k in 1:2
        t = (1 + w) * (1 + w + 2 / 3 * r)
        w = w * (1 + r / (1 + w) * (t - r / 2) / (t - r))
        r = (2 * w * (w - 4) - 1) / 72 / (1 + w)^6 * r^4
    end
    # @show r
    # @show z - w - log(w)
    return w::float(T)
end

# TODO instead of using LambertW, handle higher precision by including more
# terms in the initial guess. e.g. around -infty:
# heuristic for number of initial terms to include
# 5 terms works well for Float64
# num_terms = div(precision(float(T)), 53) * 5
# w = sum((-i)^(i - 1) / factorial(i) * t^i for i in 1:num_terms)
using LambertW
function omegawright(z::BigFloat)
    return lambertw(exp(z))
end
