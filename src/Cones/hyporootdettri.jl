#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

hypograph of the root determinant of a (row-wise lower triangle) symmetric positive definite matrix
(u in R, W in S_n+) : u <= det(W)^(1/n)

SC barrier from correspondence with A. Nemirovski
-(5 / 3) ^ 2 * (log(det(W) ^ (1 / n) - u) + logdet(W))
=#

mutable struct HypoRootdetTri{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual::Bool
    dim::Int
    side::Int
    is_complex::Bool
    point::Vector{T}
    rt2::T

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_prod_updated::Bool
    inv_hess_prod_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache

    W::Matrix{R}
    work_mat::Matrix{R}
    fact_W
    Wi::Matrix{R}
    rootdet::T
    rootdetu::T
    frac::T
    # constants for Kronecker product and dot product components of the Hessian
    kron_const::T
    dot_const::T
    sc_const::T

    function HypoRootdetTri{T, R}(
        dim::Int,
        is_dual::Bool;
        hess_fact_cache = hessian_cache(T),
        ) where {R <: RealOrComplex{T}} where {T <: Real}
        @assert dim >= 2
        cone = new{T, R}()
        cone.use_dual = is_dual
        cone.dim = dim
        cone.rt2 = sqrt(T(2))
        if R <: Complex
            side = isqrt(dim - 1) # real lower triangle and imaginary under diagonal
            @assert side^2 == dim - 1
            cone.is_complex = true
        else
            side = round(Int, sqrt(0.25 + 2 * (dim - 1)) - 0.5)
            @assert side * (side + 1) == 2 * (dim - 1)
            cone.is_complex = false
        end
        cone.side = side
        cone.sc_const = T(25) / T(9)
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

HypoRootdetTri{T, R}(dim::Int) where {R <: RealOrComplex{T}} where {T <: Real} = HypoRootdetTri{T, R}(dim, false)

reset_data(cone::HypoRootdetTri) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.hess_prod_updated = cone.inv_hess_prod_updated = false)

function setup_data(cone::HypoRootdetTri{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.W = zeros(R, cone.side, cone.side)
    cone.work_mat = zeros(R, cone.side, cone.side)
    return
end

get_nu(cone::HypoRootdetTri) = (cone.side + 1) * cone.sc_const

function set_initial_point(arr::AbstractVector{T}, cone::HypoRootdetTri{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    arr .= 0
    side = cone.side
    const1 = sqrt(T(5side^2 + 2side + 1))
    const2 = arr[1] = -5 * sqrt((3side + 1 - const1) / T(side + 1)) / (3 * sqrt(T(2)))
    const3 = -const2 * (side + 1 + const1) / side / 2
    incr = (cone.is_complex ? 2 : 1)
    k = 2
    @inbounds for i in 1:side
        arr[k] = const3
        k += incr * i + 1
    end
    return arr
end

function update_feas(cone::HypoRootdetTri{T, R}) where {R <: RealOrComplex{T}} where {T <: Real}
    @assert !cone.feas_updated
    u = cone.point[1]

    @views svec_to_smat!(cone.W, cone.point[2:end], cone.rt2)
    cone.fact_W = cholesky!(Hermitian(cone.W, :U), check = false) # mutates W, which isn't used anywhere else
    if isposdef(cone.fact_W)
        cone.rootdet = det(cone.fact_W) ^ inv(T(cone.side))
        cone.rootdetu = cone.rootdet - u
        cone.is_feas = (cone.rootdetu > 0)
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::HypoRootdetTri)
    @assert cone.is_feas
    u = cone.point[1]

    cone.grad[1] = inv(cone.rootdetu)
    cone.Wi = inv(cone.fact_W) # TODO in-place
    @views smat_to_svec!(cone.grad[2:cone.dim], cone.Wi, cone.rt2)
    cone.frac = cone.rootdet / cone.side / cone.rootdetu
    cone.grad[1] *= cone.sc_const
    @. cone.grad[2:end] *= -cone.sc_const * (cone.frac + 1)

    cone.grad_updated = true
    return cone.grad
end

# TODO refactor the hessian element parts
function update_hess(cone::HypoRootdetTri)
    if !cone.hess_prod_updated
        update_hess_prod(cone)
    end
    Wi = cone.Wi
    H = cone.hess.data
    side = cone.side

    idx_incr = (cone.is_complex ? 2 : 1)
    for i in 1:side
        for j in 1:(i - 1)
            row_idx = (cone.is_complex ? (i - 1)^2 + 2j : 1 + div((i - 1) * i, 2) + j)
            col_idx = row_idx
            for k in i:side
                for l in (i == k ? j : 1):(k - 1)
                    terma = Wi[k, i] * Wi[j, l]
                    termb = Wi[l, i] * Wi[j, k]
                    Wiji = Wi[j, i]
                    Wijk = Wi[l, k]
                    term1 = (terma + termb) * cone.kron_const + Wiji * 2 * real(Wijk) * cone.dot_const

                    H[row_idx, col_idx] = real(term1)
                    if cone.is_complex
                        H[row_idx + 1, col_idx] = -imag(term1)
                        term2 = (terma - termb) * cone.kron_const - Wiji * 2im * imag(Wijk) * cone.dot_const
                        H[row_idx, col_idx + 1] = imag(term2)
                        H[row_idx + 1, col_idx + 1] = real(term2)
                    end

                    col_idx += idx_incr
                end

                l = k
                term = cone.rt2 * (Wi[i, k] * Wi[k, j] * cone.kron_const + Wi[i, j] * Wi[k, k] * cone.dot_const)
                H[row_idx, col_idx] = real(term)

                if cone.is_complex
                    H[row_idx + 1, col_idx] = imag(term)
                end

                col_idx += 1
            end
        end

        j = i
        row_idx = (cone.is_complex ? (i - 1)^2 + 2j : 1 + div((i - 1) * i, 2) + j)
        col_idx = row_idx
        for k in i:side
            for l in (i == k ? j : 1):(k - 1)
                term = cone.rt2 * (Wi[k, i] * Wi[j, l] * cone.kron_const + Wi[i, j] * Wi[k, l] * cone.dot_const)

                H[row_idx, col_idx] = real(term)
                if cone.is_complex
                    H[row_idx, col_idx + 1] = imag(term)
                end

                col_idx += idx_incr
            end

            l = k

            H[row_idx, col_idx] = abs2(Wi[k, i]) * cone.kron_const + Wi[i, i] * Wi[k, k] * cone.dot_const

            col_idx += 1
        end
    end
    @. H[2:end, 2:end] *= cone.sc_const

    cone.hess_updated = true
    return cone.hess
end

# # TODO maybe can refactor with spectral cone etc
# # TODO @inbounds
# function _hess_WW_element(H::Matrix{T}, r_idx::Int, c_idx::Int, term1::T, term2::T) where {T <: Real}
#     H[r_idx, c_idx] = term1 + term2
#     return nothing
# end
#
# function _hess_WW_element(H::Matrix{T}, r_idx::Int, c_idx::Int, term1::Complex{T}, term2::Complex{T}) where {T <: Real}
#     # @inbounds begin
#         H[r_idx, c_idx] = real(term1) + real(term2)
#         H[r_idx + 1, c_idx] = imag(term2) - imag(term1)
#         H[r_idx, c_idx + 1] = imag(term1) + imag(term2)
#         H[r_idx + 1, c_idx + 1] = real(term1) - real(term2)
#     # end
#     return nothing
# end
#
# function _hess_WW_element(H::Matrix{T}, r_idx::Int, c_idx::Int, term1::Complex{T}, term2::Complex{T}) where {T <: Real}
#     # @inbounds begin
#         H[r_idx, c_idx] = real(term1) + real(term2)
#         H[r_idx + 1, c_idx] = imag(term2) - imag(term1)
#         H[r_idx, c_idx + 1] = imag(term1) + imag(term2)
#         H[r_idx + 1, c_idx + 1] = real(term1) - real(term2)
#     # end
#     return nothing
# end

# updates first row of the Hessian
function update_hess_prod(cone::HypoRootdetTri)
    @assert cone.grad_updated

    frac = cone.frac # rootdet / rootdetu / side
    # update constants used in the Hessian
    cone.kron_const = frac + 1
    cone.dot_const = abs2(frac) - frac / cone.side
    # update first row in the Hessian
    rootdetu = cone.rootdetu
    Wi = cone.Wi
    hess = cone.hess.data
    hess[1, 1] = cone.grad[1] / rootdetu
    @views smat_to_svec!(hess[1, 2:cone.dim], Wi, cone.rt2)
    @. hess[1, 2:end] *= -frac / rootdetu * cone.sc_const

    cone.hess_prod_updated = true
    return
end

# function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::HypoRootdetTri)
#     if !cone.hess_prod_updated
#         update_hess_prod(cone)
#     end
#
#     @views mul!(prod[1, :]', cone.hess[1, :]', arr)
#     @inbounds for i in 1:size(arr, 2)
#         svec_to_smat!(cone.work_mat, view(arr, 2:cone.dim, i), cone.rt2)
#         dot_prod = dot(Hermitian(cone.work_mat, :U), Hermitian(cone.Wi, :U))
#         copytri!(cone.work_mat, 'U')
#         rdiv!(cone.work_mat, cone.fact_W)
#         ldiv!(cone.fact_W, cone.work_mat)
#         axpby!(dot_prod * cone.dot_const, cone.Wi, cone.kron_const, cone.work_mat)
#         @views smat_to_svec!(prod[2:cone.dim, i], cone.work_mat, cone.rt2)
#     end
#     @. @views prod[2:cone.dim, :] *= cone.sc_const
#     @views mul!(prod[2:cone.dim, :], cone.hess[2:cone.dim, 1], arr[1, :]', true, true)
#
#     return prod
# end
