#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

epigraph of Euclidean (2-)norm (AKA second-order cone)
(u in R, w in R^n) : u >= norm_2(w)

barrier from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd, halved
-log(u^2 - norm_2(w)^2) / 2

contains some adapted code from https://github.com/embotech/ecos
TODO all oracles assumed we are using dense version of the Nesterov Todd scaling
TODO factor out repetitions in products, hessians, inverse hessians
TODO factor out eta
TODO probably undo (a-b)(a+b) dist calculations
=#

mutable struct EpiNormEucl{T <: Real} <: Cone{T}
    use_scaling::Bool
    use_3order_corr::Bool
    dim::Int
    point::Vector{T}
    dual_point::Vector{T}

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    scaling_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    normalized_point::Vector{T}
    normalized_dual_point::Vector{T}
    w::Vector{T} # TODO think about naming, it's w_bar in cvxopt paper
    # experimental
    lambda::Vector{T} # TODO think about naming, it's w_bar in cvxopt paper
    v::Vector{T}
    q::Vector{T}

    dist::T
    dual_dist::T
    gamma::T
    ww::T
    b::T
    c::T
    correction::Vector{T}

    function EpiNormEucl{T}(
        dim::Int;
        use_scaling::Bool = true,
        use_3order_corr::Bool = true,
        ) where {T <: Real}
        @assert dim >= 2
        cone = new{T}()
        cone.dim = dim
        cone.use_scaling = use_scaling
        cone.use_3order_corr = use_3order_corr
        return cone
    end
end

use_dual(cone::EpiNormEucl) = false # self-dual

use_scaling(cone::EpiNormEucl) = cone.use_scaling

use_3order_corr(cone::EpiNormEucl) = cone.use_3order_corr

# try_scaled_updates(cone::EpiNormEucl) = cone.try_scaled_updates # TODO

load_dual_point(cone::EpiNormEucl, dual_point::AbstractVector) = copyto!(cone.dual_point, dual_point)

reset_data(cone::EpiNormEucl) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.scaling_updated = false)

# TODO only allocate the fields we use
function setup_data(cone::EpiNormEucl{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.normalized_point = zeros(T, dim)
    cone.normalized_dual_point = zeros(T, dim)
    cone.w = zeros(T, dim)
    cone.v = zeros(T, dim)
    cone.correction = zeros(T, dim)
    return
end

get_nu(cone::EpiNormEucl) = 1

function set_initial_point(arr::AbstractVector, cone::EpiNormEucl)
    arr .= 0
    arr[1] = 1
    return arr
end

# calculates u ^ 2 - norm(w) ^ 2 for a vector (u, w)
calc_dist(x::Vector) = @views abs2(x[1]) - sum(abs2, x[2:end])

function update_feas(cone::EpiNormEucl)
    @assert !cone.feas_updated
    u = cone.point[1]
    if u > 0
        w = view(cone.point, 2:cone.dim)
        cone.dist = calc_dist(cone.point)
        cone.is_feas = (cone.dist > 0)
    else
        cone.is_feas = false
    end
    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::EpiNormEucl)
    @assert cone.is_feas
    @. cone.grad = cone.point / cone.dist
    cone.grad[1] *= -1
    cone.grad_updated = true
    return cone.grad
end

jdot(x::AbstractVector, y::AbstractVector) = @views x[1] * y[1] - dot(x[2:end], y[2:end])

function update_scaling(cone::EpiNormEucl)
    @assert !cone.scaling_updated
    @assert cone.feas_updated
    dual_dist = cone.dual_dist = calc_dist(cone.dual_point)
    @assert dual_dist >= 0
    normalized_point = cone.normalized_point .= cone.point ./ sqrt(cone.dist)
    normalized_dual_point = cone.normalized_dual_point .= cone.dual_point ./ sqrt(dual_dist)
    gamma = cone.gamma = sqrt((1 + dot(normalized_point, normalized_dual_point)) / 2)

    w = cone.w
    w[1] = normalized_point[1] + normalized_dual_point[1]
    @. @views w[2:end] = normalized_point[2:end] - normalized_dual_point[2:end]
    w ./= 2 # NOTE probably not /2 for unhalved barrier
    w ./= gamma

    # different code
    # v = cone.v
    # vs = dot(normalized_point, v)
    # vz = jdot(normalized_dual_point, v)
    # vq = (vs + vz) / gamma / 2 # this is just v'q since q = (s + Jz) / 2 gamma
    # vu = vs - vz # dunno what this is
    #
    # lambda = cone.lambda
    # lambda[1] = gamma
    #
    # w[1] = 2 * v[1] * vq - (normalized_point[1] + normalized_dual_point[1]) / gamma / 2
    # d = (v[1] * vu - normalized_point[1] / 2 + normalized_dual_point[1] / 2) / (1 + w[1])
    #
    # @views begin
    #     copyto!(lambda[2:end], v[2:end])
    #     @. lambda[2:end] *= 2 * (-d * vq + 0.5 * vu)
    #     @. lambda[2:end] +=  (1 - d / gamma) * normalized_point[2:end] / 2
    #     @. lambda[2:end] +=  (1 + d / gamma) * normalized_dual_point[2:end] / 2
    # end
    # @. lambda .*= sqrt(cone.dist / dual_dist)
    #
    # v .*= 2 * vq
    # v[1] -= normalized_point[1] / 2 / gamma
    # @views @. v[2:end] += normalized_point / gamma / 2
    # @views @. v[2:end] -= normalized_dual_point /  gamma / 2
    # v[1] += 1
    # v ./=  sqrt(2 * v[1])

    v = cone.v
    copyto!(v, w)
    v[1] += 1
    v ./= sqrt(2 * (w[1] + 1))

    w2nw2n = sum(abs2, w[2:end])
    cone.ww = abs2(w[1]) + w2nw2n
    cone.b = 1 + 2 / (1 + w[1]) + w2nw2n / abs2(1 + w[1])
    cone.c = 1 + w[1] + w2nw2n / (1 + w[1])

    cone.scaling_updated = true
    return cone.scaling_updated
end

# J(q) = (q[1] *= -1; q)
#
# function update_scaling(cone::EpiNormEucl)
#     @assert !cone.scaling_updated
#     @assert cone.feas_updated
#     dual_dist = cone.dual_dist = calc_dist(cone.dual_point)
#     @assert dual_dist >= 0
#     normalized_point = cone.normalized_point .= cone.point ./ sqrt(cone.dist)
#     normalized_dual_point = cone.normalized_dual_point .= cone.dual_point ./ sqrt(dual_dist)
#     cone.gamma = sqrt((1 + dot(normalized_point, normalized_dual_point)) / 2)
#
#     q = cone.q
#     v = cone.v
#     w = cone.w
#     w_old_dist = calc_dist(cone.w)
#     q_old_dist = calc_dist(cone.q)
#
#     q[1] = normalized_point[1] + normalized_dual_point[1]
#     @. @views q[2:end] = normalized_point[2:end] - normalized_dual_point[2:end]
#     q ./= 2
#     q ./= cone.gamma
#
#     w .= sqrt(w_old_dist) * sqrt(q_old_dist) * (2 * v * v' * q - J(q))
#
#     v = copy(w)
#     v[1] += 1
#     v ./= sqrt(2 * (w[1] + 1))
#
#
#
#
#     w2nw2n = sum(abs2, w[2:end])
#     cone.ww = abs2(w[1]) + w2nw2n
#     cone.b = 1 + 2 / (1 + w[1]) + w2nw2n / abs2(1 + w[1])
#     cone.c = 1 + w[1] + w2nw2n / (1 + w[1])
#
#     cone.scaling_updated = true
# end

function update_hess(cone::EpiNormEucl)
    @assert cone.grad_updated
    @assert cone.is_feas
    if cone.use_scaling
        if !cone.scaling_updated
            update_scaling(cone)
        end
        w = cone.w
        hess = cone.hess.data

        # dumb instantiation
        # Wbar = similar(cone.hess)
        # Wbar.data[1, 1] = w[1]
        # Wbar.data[1, 2:end] .= -w[2:end]
        # Wbar.data[2:end, 2:end] = w[2:end] * w[2:end]' / (w[1] + 1)
        # Wbar.data[2:end, 2:end] += I
        # cone.hess.data .= Wbar * Wbar * (cone.dual_dist / cone.dist) ^ (1 / 2)

        # pattern matched from inverse Hessian
        hess[1, 1] = cone.ww
        @. hess[1, 2:end] = -cone.c * w[2:end]
        @views mul!(hess[2:end, 2:end], w[2:end], w[2:end]', cone.b, false)
        hess[2:end, 2:end] += I
        hess .*= sqrt(cone.dual_dist / cone.dist)
    else
        mul!(cone.hess.data, cone.grad, cone.grad', 2, false)
        cone.hess += inv(cone.dist) * I
        cone.hess[1, 1] -= 2 / cone.dist
        cone.hess_updated = true
    end
    return cone.hess
end

function update_inv_hess(cone::EpiNormEucl)
    @assert cone.is_feas
    if cone.use_scaling
        if !cone.scaling_updated
            update_scaling(cone)
        end
        w = cone.w
        inv_hess = cone.inv_hess.data

        # dumb instantiation
        # Wbar = similar(cone.inv_hess)
        # Wbar.data[1, :] .= w
        # Wbar.data[2:end, 2:end] = w[2:end] * w[2:end]' / (w[1] + 1)
        # Wbar.data[2:end, 2:end] += I
        # cone.inv_hess.data .= Wbar * Wbar * (cone.dist / cone.dual_dist) ^ (1 / 2)

        # see https://github.com/embotech/ecos/blob/develop/src/kkt.c
        inv_hess[1, 1] = cone.ww
        @. inv_hess[1, 2:end] = cone.c * w[2:end]
        @views mul!(inv_hess[2:end, 2:end], w[2:end], w[2:end]', cone.b, false)
        inv_hess[2:end, 2:end] += I
        inv_hess .*= sqrt(cone.dist / cone.dual_dist)
    else
        mul!(cone.inv_hess.data, cone.point, cone.point', 2, false)
        cone.inv_hess += cone.dist * I
        cone.inv_hess[1, 1] -= 2 * cone.dist
        cone.inv_hess_updated = true
    end
    return cone.inv_hess
end

update_hess_prod(cone::EpiNormEucl) = nothing
update_inv_hess_prod(cone::EpiNormEucl) = nothing

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormEucl)
    @assert cone.is_feas
    if cone.use_scaling
        if !cone.scaling_updated
            update_scaling(cone)
        end
        w = cone.w
        prod[1, :] = cone.ww * arr[1, :] - cone.c * arr[2:end, :]' * w[2:end]
        for j in 1:size(prod, 2)
            @views pa = dot(cone.w[2:end], arr[2:end, j])
            @. prod[2:end, j] = pa * cone.w[2:end] * cone.b
        end
        @. prod[2:end, :] += arr[2:end, :]
        @. prod[2:end, :] -= cone.c * arr[1, :]' * w[2:end]
        prod .*= sqrt(cone.dual_dist / cone.dist)
    else
        p1 = cone.point[1]
        p2 = @view cone.point[2:end]
        @inbounds for j in 1:size(prod, 2)
            arr_1j = arr[1, j]
            arr_2j = @view arr[2:end, j]
            ga = 2 * (dot(p2, arr_2j) - p1 * arr_1j) / cone.dist
            prod[1, j] = -ga * p1 - arr_1j
            @. prod[2:end, j] = ga * p2 + arr_2j
        end
        @. prod ./= cone.dist
    end
    return prod
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormEucl)
    @assert cone.is_feas
    if cone.use_scaling
        if !cone.scaling_updated
            update_scaling(cone)
        end
        w = cone.w
        prod[1, :] = cone.ww * arr[1, :] + cone.c * arr[2:end, :]' * w[2:end]
        for j in 1:size(prod, 2)
            @views pa = dot(cone.w[2:end], arr[2:end, j])
            @. prod[2:end, j] = pa * cone.w[2:end] * cone.b
        end
        @. prod[2:end, :] += arr[2:end, :]
        @. prod[2:end, :] += cone.c * arr[1, :]' * w[2:end]
        prod .*= sqrt(cone.dist / cone.dual_dist)
    else
        @inbounds for j in 1:size(prod, 2)
            @views pa = 2 * dot(cone.point, arr[:, j])
            @. prod[:, j] = pa * cone.point
        end
        @. @views prod[1, :] -= cone.dist * arr[1, :]
        @. @views prod[2:end, :] += cone.dist * arr[2:end, :]
    end
    return prod
end

# multiplies arr by W, the squareroot of the scaling matrix
# TODO there is a faster way
# function scalmat_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormEucl)
#     if !cone.scaling_updated
#         update_scaling(cone)
#     end
#     w = cone.w
#
#     @views begin
#         mul!(prod[1, :], arr', w)
#         for j in 1:size(prod, 2)
#             pa = dot(w[2:end], arr[2:end, j])
#             @. prod[2:end, j] = w[2:end] * pa
#         end
#         @. prod[2:end, :] /= (w[1] + 1)
#         @. prod[2:end, :] += arr[2:end, :]
#         @. prod[2:end, :] += arr[1, :]' * w[2:end]
#         # @. prod[2:end, :] += w[2:end] * arr[1, :]'
#
#     end
#     prod .*= sqrt(sqrt(cone.dist / cone.dual_dist))
#
#     # Wbar = similar(cone.inv_hess)
#     # Wbar.data[1, :] .= w
#     # Wbar.data[2:end, 2:end] = w[2:end] * w[2:end]' / (w[1] + 1)
#     # Wbar.data[2:end, 2:end] += I
#     # W = Wbar * (cone.dist / cone.dual_dist) ^ (1 / 4)
#     # prod .= W * arr
#
#     return prod
# end

# TODO this oracle may not be necessary for matrices
function scalmat_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormEucl)
    if !cone.scaling_updated
        update_scaling(cone)
    end
    prod .= cone.v
    @inbounds @views for j in 1:size(arr, 2)
        prod[:, j] *= 2 * dot(cone.v, arr[:, j])
    end
    @views prod[1, :] .-= arr[1, :]
    @views prod[2:end, :] .+= arr[2:end, :]
    prod .*= sqrt(sqrt(cone.dist) / sqrt(cone.dual_dist))
    return prod
end
#
# function scalmat_ldiv!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormEucl)
#     if !cone.scaling_updated
#         update_scaling(cone)
#     end
#     w = cone.w
#
#     @views begin
#         prod[1, :] .= arr[1, :] * w[1]
#         mul!(prod[1, :], arr[2:end, :]', w[2:end], -1, true)
#         for j in 1:size(prod, 2)
#             pa = dot(w[2:end], arr[2:end, j])
#             @. prod[2:end, j] = w[2:end] * pa
#         end
#         @. prod[2:end, :] /= (w[1] + 1)
#         @. prod[2:end, :] += arr[2:end, :]
#         @. prod[2:end, :] -= arr[1, :]' * w[2:end]
#     end
#     prod .*= sqrt(sqrt(cone.dual_dist / cone.dist))
#     return prod
# end

# scaling is symmetric, trans kwarg ignored TODO factor as another function?
function scalmat_ldiv!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormEucl; trans::Bool = false)
    if !cone.scaling_updated
        update_scaling(cone)
    end
    prod .= cone.v
    @views prod[2:end, :] *= -1
    @inbounds @views for j in 1:size(arr, 2)
        prod[:, j] *= 2 * jdot(cone.v, arr[:, j])
    end

    @views prod[1, :] .-= arr[1, :]
    @views prod[2:end, :] .+= arr[2:end, :]
    prod .*= sqrt(sqrt(cone.dual_dist) / sqrt(cone.dist))
    return prod
end

# # TODO refactor into Cones.jl
# # returns W_inv \circ lambda \diamond correction
# function correction(cone::EpiNormEucl, s_sol::AbstractVector, z_sol::AbstractVector)
#     if !cone.scaling_updated
#         update_scaling(cone)
#     end
#     tmp_s = scalmat_ldiv!(similar(s_sol), s_sol, cone)
#     tmp_z = scalmat_prod!(similar(z_sol), z_sol, cone)
#
#     mehrotra_term = conic_prod!(similar(cone.point), tmp_s, tmp_z, cone)
#
#     C = scalvec_ldiv!(similar(cone.point), mehrotra_term, cone)
#     scalmat_ldiv!(cone.correction, C, cone)
#
#     return cone.correction
# end

# from MOSEK paper
# TODO cleanup
function correction(cone::EpiNormEucl, s_sol::AbstractVector, z_sol::AbstractVector)
    @assert cone.grad_updated
    corr = cone.correction
    point = cone.point

    jdot_p_s = jdot(point, s_sol)
    @. corr = jdot_p_s * z_sol
    dot_s_z = dot(s_sol, z_sol)
    dot_p_z = dot(point, z_sol)
    corr[1] += dot_s_z * point[1] - dot_p_z * s_sol[1]
    @. @views corr[2:end] += -dot_s_z * point[2:end] + dot_p_z * s_sol[2:end]
    corr ./= cone.dist

    return corr
end

function dist_to_bndry(::EpiNormEucl{T}, point::Vector{T}, dir::AbstractVector{T}) where {T}
    point_dir_dist = point[1] * dir[1] - sum(point[i] * dir[i] for i in 2:length(point))
    fact = (point_dir_dist + dir[1]) / (point[1] + 1)

    dist2n = zero(T)
    for i in 2:length(point)
        dist2n += abs2(dir[i] - fact * point[i])
    end
    return -point_dir_dist + sqrt(dist2n)
end

function step_max_dist(cone::EpiNormEucl{T}, s_sol::AbstractVector{T}, z_sol::AbstractVector{T}) where {T}
    primal_step_dist = sqrt(cone.dist) / dist_to_bndry(cone, cone.normalized_point, s_sol)
    dual_step_dist = sqrt(cone.dual_dist) / dist_to_bndry(cone, cone.normalized_dual_point, z_sol)

    # TODO refactor
    step_dist = T(Inf)
    if primal_step_dist > 0
        step_dist = min(step_dist, primal_step_dist)
    end
    if dual_step_dist > 0
        step_dist = min(step_dist, dual_step_dist)
    end
    return step_dist
end



# TODO cleanup below and implement this function (for scaling and without scaling)
# function hess_Uprod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Nonnegative)


# function hess_L_fact(cone::EpiNormEucl{T}) where {T} # remove T
#     dist = (abs2(cone.point[1]) - sum(abs2, cone.point[2:end])) # replace by cone.dist when figured out how to double things correctly
#     # dist = cone.dist
#     dim = cone.dim
#     L = zeros(T, dim, dim)
#     w = cone.point[2:dim]
#     dist_sqrt = sqrt(dist)
#     mul!(L[2:dim, 2:dim], w, w')
#     # hyp_aat!(L_main, w) # maybe does dispatch to the right thing
#     @. L[2:dim, 2:dim] /= (cone.point[1] + dist_sqrt)
#     # L_main += dist_sqrt * I # returns an array
#     L[2:dim, 2:dim] += dist_sqrt * I
#     L[2:dim, 2:dim] = w * w' / (cone.point[1] + dist_sqrt) + dist_sqrt * I
#     L[1, 2:dim] = L[2:dim, 1] = -cone.point[2:dim]
#     L[1, 1] = cone.point[1]
#     @. L /= dist
#     @. L *= sqrt(2)
#     return L
# end
#
# hess_U_fact(cone::EpiNormEucl) = hess_L_fact(cone)
#
# hess_U_prod!(prod, arr, cone::EpiNormEucl) = mul!(prod, hess_L_fact(cone), arr)
#
# function hess_L_div!(dividend, arr, cone::EpiNormEucl{T}) where {T} # remove T
#     dist = (abs2(cone.point[1]) - sum(abs2, cone.point[2:end])) # replace by cone.dist when figured out how to double things correctly
#     # dist = cone.dist
#     dim = cone.dim
#     L = zeros(T, dim, dim)
#     w = cone.point[2:dim]
#     dist_sqrt = sqrt(dist)
#     mul!(L[2:dim, 2:dim], w, w')
#     # hyp_aat!(L_main, w) # maybe does dispatch to the right thing
#     @. L[2:dim, 2:dim] /= (cone.point[1] + dist_sqrt)
#     # L_main += dist_sqrt * I # returns an array
#     L[2:dim, 2:dim] += dist_sqrt * I
#     L[2:dim, 2:dim] = w * w' / (cone.point[1] + dist_sqrt) + dist_sqrt * I
#     L[1, 2:dim] = L[2:dim, 1] = cone.point[2:dim]
#     L[1, 1] = cone.point[1]
#     @. L /= sqrt(2)
#     return mul!(dividend, L, arr)
# end
