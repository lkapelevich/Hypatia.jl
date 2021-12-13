# approximation algorithm from ``Algorithm 917: Complex Double-Precision
# Evaluation of the Wright ω Function'' by Piers W. Laurence, Robert M. Corless
# and David J. Jeffrey
# Series approximations can be found in the book chapter ``The Wright ω Function''
# by Corless, R. M. and Jeffrey, D. J.

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

# using Combinatorics
# using DelimitedFiles
# # Eulerian numbers of the second kind
# # complicated explicit formula in "Explicit formulas for the Eulerian
# # numbers of the second kind"
# # see also http://luschny.de/math/oeis/A340556.html
# function eulerian2(n::Int, m::Int)
#     if n < 0
#         return 0
#     elseif iszero(m)
#         return m^n
#     else
#         return (2n - m) * eulerian2(n - 1, m - 1) + m * eulerian2(n - 1, m)
#     end
# end
# # function one_consts()
# #     num_terms = 20
# #     one_consts = zeros(BigFloat, num_terms)
# #     for i in 1:num_terms
# #         one_consts[i] = sum(eulerian2(i - 1, k) * (-1)^k for k in 0:(i - 1)) / big(2)^(2i - 1) / big(factorial(i))
# #     end
# #     one_consts = vcat(0.5, -one_consts[2:end]) # FIXME
# #     writedlm("oneconsts.txt", one_consts)
# # end
#
# one_consts = readdlm("one_consts.txt", BigFloat)
# inf_consts = readdlm("inf_consts.txt", BigFloat)
# neginf_consts = readdlm("neginf_consts.txt", BigFloat)
#
# function omegawright(z::T) where {T <: Real}
#     # heuristic
#     num_terms = div(precision(float(z)), 53) * 5
#     FT = float(T)
#     if z < -2
#         t = exp(z)
#         # w = sum((-T(i))^(i - 1) / T(factorial(i)) * t^i for i in 1:num_terms)
#         @views w = @Base.Math.horner(t, zero(FT), FT.(neginf_consts[1:num_terms])...) # TODO figure out proper way to read constants in right precision
#     elseif z < 1 + π
#         # imprecise in BF
#         z1 = z - 1
#         @views w = @Base.Math.horner(z1, one(FT), FT.(one_consts[1:num_terms])...)
#         # w = 1 + sum(
#         #     sum(eulerian2(i - 1, k) * (-1)^k for k in 0:(i - 1)) /
#         #     T(2)^(2i - 1) / T(factorial(i)) * z1^i
#         #     for i in 1:num_terms)
#     else
#         # imprecise in BF
#         lz = log(z)
#         function consts_infty(i::Int, j::Int)
#             (i == j == 0) && return 0
#             return (-1)^i / T(factorial(j)) * Combinatorics.stirlings1(i + j, i + 1)
#         end
#         # w = z - lz + @Base.Math.horner((lz / z), [@Base.Math.horner(inv(z),
#         #     [inf_consts[i, j] for i in 1:(num_terms - 1)]...) for j in 1:(num_terms - 1)]...)
#         # w = z - lz + @Base.Math.horner((lz / z), [@Base.Math.horner(inv(z),
#         #     [consts_infty(i, j) for i in 0:(num_terms - 2)]...) for j in 0:(num_terms - 2)]...)
#         w = z - lz + sum(consts_infty(i, j) * lz^j / z^(i + j) for
#             i in 0:(num_terms - 2) for j in 0:(num_terms - 2))
#     end
#     r = z - w - log(w)
#     for k in 1:2
#         t = (1 + w) * (1 + w + 2 / T(3) * r)
#         w = w * (1 + r / (1 + w) * (t - r / 2) / (t - r))
#         r = (2 * w * (w - 4) - 1) / 72 / (1 + w)^6 * r^4
#     end
#     return w
# end

function rootnewton(
    f::Function,
    g::Function;
    lower::T = -Inf,
    upper::T = Inf,
    init::T = (lower + upper) / 2,
    increasing::Bool = true,
    ) where {T <: Real}
    curr = init
    f_new = f(BigFloat(curr))
    iter = 0
    while abs(f_new) > 1000eps(T)
        candidate = curr - f_new / g(BigFloat(curr))
        if (candidate < lower) || (candidate > upper)
            curr = (lower + upper) / 2
        else
            curr = candidate
        end
        f_new = f(BigFloat(curr))
        if (f_new < 0 && increasing) || (f_new >= 0 && !increasing)
            lower = curr
        else
            upper = curr
        end
        iter += 1
        if iter > 200
            @warn "too many iters in dual grad"
            break
        end
    end
    return T(curr)
end
