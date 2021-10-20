#=
Symm predict and center stepper
=#

mutable struct SymmStepper{T <: Real} <: Stepper{T}
    shift_sched::Int
    searcher_options

    prev_alpha::T
    gamma::T
    rhs::Point{T}
    dir::Point{T}
    temp::Point{T}
    dir_cent::Point{T}
    dir_pred::Point{T}
    dir_predadj::Point{T}
    dir_temp::Vector{T}

    searcher::StepSearcher{T}
    pred_only::Bool

    function SymmStepper{T}(;
        shift_sched::Int = 0,
        searcher_options...
        ) where {T <: Real}
        stepper = new{T}()
        stepper.shift_sched = shift_sched
        stepper.searcher_options = searcher_options
        return stepper
    end
end

function load(stepper::SymmStepper{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model

    stepper.prev_alpha = one(T)
    stepper.rhs = Point(model)
    stepper.dir = Point(model)
    stepper.temp = Point(model)
    stepper.dir_cent = Point(model, ztsk_only = false)
    # stepper.dir_cent = Point(model, ztsk_only = true)
    # stepper.dir_pred = Point(model, ztsk_only = true)
    stepper.dir_pred = Point(model, ztsk_only = false)
    stepper.dir_predadj = Point(model, ztsk_only = false)
    # stepper.dir_predadj = Point(model, ztsk_only = true)
    stepper.dir_temp = zeros(T, length(stepper.rhs.vec))

    stepper.searcher = StepSearcher{T}(model; stepper.searcher_options...)
    stepper.pred_only = true

    stepper.gamma = 0

    return stepper
end

# TODO test reduction in residuals after a step
function step(stepper::SymmStepper{T}, solver::Solver{T}) where {T <: Real}
    point = solver.point
    model = solver.model
    rhs = stepper.rhs
    dir = stepper.dir
    dir_cent = stepper.dir_cent
    dir_pred = stepper.dir_pred
    dir_predadj = stepper.dir_predadj

    # update linear system solver factorization
    solver.time_upsys += @elapsed update_lhs(solver.syssolver, solver)

    # calculate centering direction and adjustment
    solver.time_uprhs += @elapsed update_rhs_cent(solver, rhs)
    solver.time_getdir += @elapsed get_directions(stepper, solver)
    copyto!(dir_cent.vec, dir.vec)

    # calculate affine/prediction direction and adjustment
    solver.time_uprhs += @elapsed update_rhs_pred(solver, rhs)
    solver.time_getdir += @elapsed get_directions(stepper, solver)
    copyto!(dir_pred.vec, dir.vec)
    solver.time_uprhs += @elapsed update_rhs_predadj(solver, rhs, dir)
    solver.time_getdir += @elapsed get_directions(stepper, solver)
    copyto!(dir_predadj.vec, dir.vec)

    @assert point.kap[] * dir.tau[] + point.tau[] * dir.kap[] ≈
        -dir_pred.tau[] * dir_pred.kap[]
    @assert point.kap[] * dir_predadj.tau[] + point.tau[] * dir_predadj.kap[] ≈
        -dir_pred.tau[] * dir_pred.kap[]
    # @assert point.z .* dir_predadj.s + point.s .* dir_predadj.z ≈
    #     -dir_pred.s .* dir_pred.z

    # @show dot(stepper.dir_pred.primal_views, stepper.dir_pred.dual_views) +
    #     stepper.dir_pred.tau[] * stepper.dir_pred.kap[]
    # @show dot(stepper.dir_predadj.primal_views, stepper.dir_predadj.dual_views) +
    #     stepper.dir_predadj.tau[] * stepper.dir_predadj.kap[]
    # @show dot(point.s, stepper.dir_predadj.z) +
    #     dot(point.z, stepper.dir_predadj.s) +
    #     stepper.dir_predadj.tau[] * point.kap[] +
    #     stepper.dir_predadj.kap[] * point.tau[]
    # ok
    # check =
    #     dot(point.s, stepper.dir_pred.z) +
    #     dot(point.z, stepper.dir_pred.s) +
    #     stepper.dir_pred.tau[] * point.kap[] +
    #     stepper.dir_pred.kap[] * point.tau[] +
    #     dot(point.s, point.z) + point.tau[] * point.kap[]
    # @show check

    # search with Symm directions and adjustments
    stepper.pred_only = true
    stepper.searcher.prox_bound = 1e-10 # TODO use
    solver.time_search += @elapsed alpha = search_alpha(point, model, stepper)
    stepper.gamma = (1 - alpha) * min(abs2(1 - alpha), 0.25)
    stepper.pred_only = false
    stepper.searcher.prox_bound = 0.1 # TODO use
    solver.time_search += @elapsed alpha = search_alpha(point, model, stepper)

    # gamma = stepper.gamma
    # s_comb = (1 - gamma) * dir_pred.s + gamma * dir_cent.s + dir_predadj.s
    # z_comb = (1 - gamma) * dir_pred.z + gamma * dir_cent.z + dir_predadj.z
    # t_comb = (1 - gamma) * dir_pred.tau[] + gamma * dir_cent.tau[] + dir_predadj.tau[]
    # k_comb = (1 - gamma) * dir_pred.kap[] + gamma * dir_cent.kap[] + dir_predadj.kap[]
    # @show dot(s_comb, z_comb) + t_comb * k_comb

    if iszero(alpha)
        # solver.verbose && println("trying centering")
        # stepper.gamma = 1
        # dir_predadj.vec .= 0
        # solver.time_search += @elapsed alpha =
        #     search_alpha(point, model, stepper)

        if iszero(alpha)
            @warn("cannot step in combined direction")
            solver.status = NumericalFailure
            stepper.prev_alpha = alpha
            return false
        end
    end

    # step
    update_stepper_points(alpha, point, stepper, false)
    stepper.prev_alpha = alpha

    return true
end

expect_improvement(stepper::SymmStepper) = true

function update_stepper_points(
    alpha::T,
    point::Point{T},
    stepper::SymmStepper{T},
    ztsk_only::Bool,
    ) where {T <: Real}
    gamma = stepper.gamma
    if ztsk_only
        cand = stepper.temp.ztsk
        copyto!(cand, point.ztsk)
        dir_cent = stepper.dir_cent.ztsk
        dir_pred = stepper.dir_pred.ztsk
    else
        cand = point.vec
        dir_cent = stepper.dir_cent.vec
        dir_pred = stepper.dir_pred.vec
    end

    if stepper.pred_only
        @. cand += alpha * dir_pred
    else
        dir_predadj = (ztsk_only ? stepper.dir_predadj.ztsk : stepper.dir_predadj.vec)
        @. cand += alpha * (dir_cent * gamma + (1 - gamma) * dir_pred + dir_predadj)
    end

    # if stepper.unadj_only
    #     # no adjustment
    #     if stepper.cent_only
    #         # centering
    #         @. cand += alpha * dir_cent
    #     else
    #         # Symm
    #         alpha_m1 = 1 - alpha
    #         @. cand += alpha * dir_pred + alpha_m1 * dir_cent
    #     end
    # else
    #     # adjustment
    #     dir_centadj = (ztsk_only ? stepper.dir_centadj.ztsk :
    #         stepper.dir_centadj.vec)
    #     alpha_sqr = abs2(alpha)
    #     if stepper.cent_only
    #         # centering
    #         @. cand += alpha * dir_cent + alpha_sqr * dir_centadj
    #     else
    #         # Symm
    #         dir_predadj = (ztsk_only ? stepper.dir_predadj.ztsk :
    #             stepper.dir_predadj.vec)
    #         alpha_m1 = 1 - alpha
    #         alpha_m1sqr = abs2(alpha_m1)
    #         @. cand += alpha * dir_pred + alpha_sqr * dir_predadj +
    #             alpha_m1 * dir_cent + alpha_m1sqr * dir_centadj
    #     end
    # end

    return
end

function check_cone_points(
    model::Models.Model{T},
    stepper::SymmStepper{T};
    ) where {T <: Real}
    searcher = stepper.searcher
    cand = stepper.temp
    szk = searcher.szk
    cones = model.cones
    # min_prox = searcher.min_prox
    use_max_prox = searcher.use_max_prox
    # proxsqr_bound = abs2(searcher.prox_bound)
    β = T(0.01)
    # β = sqrt(sqrt(eps(T)))

    taukap = cand.tau[] * cand.kap[]
    (min(cand.tau[], cand.kap[], taukap) < eps(T)) && return false

    for k in eachindex(cones)
        szk[k] = dot(cand.primal_views[k], cand.dual_views[k])
        (szk[k] < eps(T)) && return false
    end
    mu = (sum(szk) + taukap) / searcher.nup1
    (mu < eps(T)) && return false

    taukap_rel = taukap / mu
    # (taukap_rel < min_prox) && return false
    # taukap_proxsqr = abs2(taukap_rel - 1)
    # (taukap_proxsqr > proxsqr_bound) && return false
    (taukap_rel < β) && return false


    # for k in eachindex(cones)
    #     nu_k = Cones.get_nu(cones[k])
    #     sz_rel_k = szk[k] / (mu * nu_k)
    #     if (sz_rel_k < min_prox) || (nu_k * abs2(sz_rel_k - 1) > proxsqr_bound)
    #         return false
    #     end
    # end

    # order the cones by how long it takes to check proximity condition and
    # iterate in that order, to improve efficiency
    cone_order = searcher.cone_order
    sortperm!(cone_order, searcher.cone_times, initialized = true) # stochastic

    irtmu = inv(sqrt(mu))
    agg_proxsqr = taukap_rel # TODO agg related stuff is unneeded
    # aggfun = (use_max_prox ? max : +)
    aggfun = min

    for k in cone_order
        cone_k = cones[k]
        start_time = time()
        Cones.load_point(cone_k, cand.primal_views[k])
        Cones.load_dual_point(cone_k, cand.dual_views[k])
        Cones.reset_data(cone_k)

        in_prox_k = false
        if Cones.is_feas(cone_k) && Cones.is_dual_feas(cone_k) &&
            Cones.check_numerics(cone_k)
            # TODO come up with a proper way to do this
            if Cones.use_scal(cone_k)
                proxsqr_k = Cones.get_proxcompl(cone_k, irtmu, use_max_prox)
                agg_proxsqr = aggfun(agg_proxsqr, proxsqr_k)
                in_prox_k = (agg_proxsqr > β)
            else
                proxsqr_k = Cones.get_proxsqr(cone_k, irtmu, use_max_prox)
                in_prox_k = (proxsqr_k < T(0.99))
            end
        end
        searcher.cone_times[k] = time() - start_time
        in_prox_k || return false
    end

    # searcher.prox = sqrt(agg_proxsqr)
    searcher.prox = agg_proxsqr
    return true
end

function start_sched(stepper::SymmStepper, searcher::StepSearcher)
    (stepper.shift_sched <= 0) && return 1
    return max(1, searcher.prev_sched - stepper.shift_sched)
end

print_header_more(stepper::SymmStepper, solver::Solver) =
    @printf("%5s %9s", "gamma", "alpha")

function print_iteration_more(stepper::SymmStepper, solver::Solver)
    # if stepper.cent_only
    #     step = (stepper.unadj_only ? "cent" : "ce-a")
    # else
    #     step = (stepper.unadj_only ? "comb" : "co-a")
    # end
    @printf("%9.2e %9.2e", stepper.gamma, stepper.prev_alpha)
    return
end
