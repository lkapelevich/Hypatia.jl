#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

interpolation-based weighted-sum-of-squares (multivariate) polynomial cone parametrized by interpolation points ipwt

definition and dual barrier from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792

TODO can perform loop for calculating g and H in parallel
TODO maybe can avoid final factorization?
TODO scale the interior direction
=#

mutable struct WSOSPolyInterp <: PrimitiveCone
    usedual::Bool
    dim::Int
    ipwt::Vector{Matrix{Float64}}
    pnt::AbstractVector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    F
    tmp1::Vector{Matrix{Float64}}
    tmp2::Vector{Matrix{Float64}}
    tmp3::Matrix{Float64}
    scal::Float64

    function WSOSPolyInterp(dim::Int, ipwt::Vector{Matrix{Float64}}, isdual::Bool)
        for ipwtj in ipwt
            @assert size(ipwtj, 1) == dim
        end
        prmtv = new()
        prmtv.usedual = !isdual # using dual barrier
        prmtv.dim = dim
        prmtv.ipwt = ipwt
        prmtv.g = similar(ipwt[1], dim)
        prmtv.H = similar(ipwt[1], dim, dim)
        prmtv.H2 = similar(prmtv.H)
        prmtv.tmp1 = [similar(ipwt[1], size(ipwtj, 2), size(ipwtj, 2)) for ipwtj in ipwt]
        prmtv.tmp2 = [similar(ipwt[1], dim, size(ipwtj, 2)) for ipwtj in ipwt]
        prmtv.tmp3 = similar(ipwt[1], dim, dim)
        return prmtv
    end
end

WSOSPolyInterp(dim::Int, ipwt::Vector{Matrix{Float64}}) = WSOSPolyInterp(dim, ipwt, false)

dimension(prmtv::WSOSPolyInterp) = prmtv.dim
barrierpar_prmtv(prmtv::WSOSPolyInterp) = sum(size(ipwtj, 2) for ipwtj in prmtv.ipwt)
getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::WSOSPolyInterp) = (@. arr = 1.0; arr)
loadpnt_prmtv!(prmtv::WSOSPolyInterp, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::WSOSPolyInterp, scal::Float64)
    newpnt = copy(prmtv.pnt)
    # @show newpnt
    # prmtv.scal = norm(newpnt, Inf)
    prmtv.scal = prmtv.dim*scal
    newpnt ./= prmtv.scal
    # lmul!(prmtv.iscal, newpnt)
    # @show newpnt
    # @show prmtv.scal

    @. prmtv.g = 0.0
    @. prmtv.H = 0.0
    tmp3 = prmtv.tmp3

    for j in eachindex(prmtv.ipwt) # TODO can be done in parallel, but need multiple tmp3s
        ipwtj = prmtv.ipwt[j]
        tmp1j = prmtv.tmp1[j]
        tmp2j = prmtv.tmp2[j]

        # tmp1j = ipwtj'*Diagonal(pnt)*ipwtj
        mul!(tmp2j, Diagonal(newpnt), ipwtj)
        mul!(tmp1j, ipwtj', tmp2j)

        # @show norm(tmp1j)

        # pivoted cholesky, upper triangle solve
        F = cholesky!(Symmetric(tmp1j, :U), Val(true), check=false)
        if !isposdef(F)
            return false
        end
        tmp2j .= view(ipwtj, :, F.p)
        rdiv!(tmp2j, F.U)
        mul!(tmp3, tmp2j, tmp2j')

        # # bunch-kaufman-based
        # F = bunchkaufman!(Symmetric(tmp1j), true, check=false)
        # if !issuccess(F)
        #     return false
        # end
        # # ipwtj * (tmp1j^-1 * ipwtj')
        # tmp3 .= ipwtj * (F\ipwtj')
        # # @. tmp2j = ipwtj
        # # ldiv!(F, tmp2j')
        # # mul!(tmp3, ipwtj, tmp2j)

        # # advanced bunch-kaufman solve
        # # ipwtj * (tmp1j^-1 * ipwtj')
        # B = Array(ipwtj')
        # (Brows, Bcols) = size(B)
        # lsferr = Vector{Float64}(undef, Bcols)
        # lsberr = Vector{Float64}(undef, Bcols)
        # lswork = Vector{Float64}(undef, 3*Brows)
        # lsiwork = Vector{Float64}(undef, Brows)
        # lsAF = Matrix{Float64}(undef, Brows, Brows)
        # lsS = Vector{Float64}(undef, Brows)
        #
        # X = Matrix{Float64}(undef, Brows, Bcols)
        # A = copy(tmp1j)
        # # B = copy(ipwtj)
        # posdef = hypatia_posvx!(X, A, B, lsferr, lsberr, lswork, lsiwork, lsAF, lsS)
        # if !posdef
        #     return false
        # end
        # # tmp3 .= ipwtj * X
        # mul!(tmp3, ipwtj, X)
        # @show norm(M - tmp3)

        # @show norm(tmp2j)
        # @show norm(tmp3)

        for i in eachindex(prmtv.g)
            prmtv.g[i] -= tmp3[i,i]
        end
        @. prmtv.H += abs2(tmp3)
    end

    # @show norm(prmtv.g)
    # @show norm(prmtv.H)

    @. prmtv.H2 = prmtv.H
    # prmtv.F = cholesky!(Symmetric(prmtv.H2), Val(true), check=false)
    # @show isposdef(prmtv.F)
    # return isposdef(prmtv.F)
    # return factH(prmtv)
    prmtv.F = bunchkaufman!(Symmetric(prmtv.H2), true, check=false)
    # @show issuccess(prmtv.F)
    return issuccess(prmtv.F)
end

calcg_prmtv!(g::AbstractVector{Float64}, prmtv::WSOSPolyInterp) = (@. g = prmtv.g/prmtv.scal; g)
calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::WSOSPolyInterp) = (ldiv!(prod, prmtv.F, arr); @. prod = prod * prmtv.scal * prmtv.scal; prod)
calcHarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::WSOSPolyInterp) = (mul!(prod, prmtv.H, arr); @. prod = prod / prmtv.scal / prmtv.scal; prod)
