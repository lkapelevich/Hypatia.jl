#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

find maximum volume hypercube with edges parallel to the axes inside a polyhedron
=#

include(joinpath(@__DIR__, "../common_native.jl"))

function maxvolume_native(
    ::Type{T},
    n::Int,
    use_hypogeomean::Bool, # use hypogeomean cone for geomean objective
    use_power::Bool, # use power cones for geomean objective
    use_epipersquare::Bool, # use epipersquare cones for geomean objective
    ) where {T <: Real}
    @assert use_hypogeomean + use_power + use_epipersquare == 1
    @assert n > 2
    poly_hrep = Matrix{T}(I, n, n)
    poly_hrep .+= T.(randn(n, n)) / n
    c = vcat(-1, zeros(T, n))
    A = hcat(zeros(T, n), poly_hrep)
    b = ones(T, n)

    # modified from https://github.com/JuliaOpt/MathOptInterface.jl/blob/master/src/Bridges/Constraint/geomean.jl
    function log_floor(n, i)
        if n <= 2 ^ i
            i
        else
            log_floor(n, i + 1)
        end
    end
    function log_floor(n::Integer)
        @assert n > zero(n)
        log_floor(n, zero(n))
    end

    if use_hypogeomean
        G = -Matrix{T}(I, n + 1, n + 1)
        h = zeros(T, n + 1)
        cones = CO.Cone{T}[CO.HypoGeomean{T}(fill(inv(T(n)), n))]
    elseif use_power
        cones = CO.Cone{T}[]
        # number of 3-dimensional power cones needed is n - 1, number of new variables is n - 2
        len_power = 3 * (n - 1)
        G_geo_orig = zeros(T, len_power, n)
        G_geo_newvars = zeros(T, len_power, n - 2)
        c = vcat(c, zeros(T, n - 2))
        A = hcat(A, zeros(T, n, n - 2))

        # first cone is a special case since two of the original variables participate in it
        G_geo_orig[1, 1] = -1
        G_geo_orig[2, 2] = -1
        G_geo_newvars[3, 1] = -1
        push!(cones, CO.Power{T}(fill(inv(T(2)), 2), 1))
        offset = 4
        # loop over new vars
        for i in 1:(n - 3)
            G_geo_newvars[offset + 2, i + 1] = -1
            G_geo_newvars[offset + 1, i] = -1
            G_geo_orig[offset, i + 2] = -1
            push!(cones, CO.Power{T}([inv(T(i + 2)), T(i + 1) / T(i + 2)], 1))
            offset += 3
        end
        # last row also special becuase hypograph variable is involved
        G_geo_orig[offset, n] = -1
        G_geo_newvars[offset + 1, n - 2] = -1
        G = [
            vcat(zeros(T, len_power - 1), -one(T))  G_geo_orig  G_geo_newvars
            ]
        push!(cones, CO.Power{T}([inv(T(n)), T(n - 1) / T(n)], 1))
        h = zeros(T, 3 * (n - 1))
    else
        @assert use_epipersquare == true
        # number of variables inside geometric mean is n
        # number of layers of variables
        num_layers = log_floor(n)
        # number of new variables = 1 + 2 + ... + 2^(l - 1) = 2^l - 1
        num_new_vars = 2 ^ num_layers - 1

        c = vcat(c, zeros(T, num_new_vars))
        A = hcat(A, zeros(T, n, num_new_vars))
        rtfact = sqrt(T(2) ^ num_layers)
        # excludes original hypograph variable, padded later
        G_rsoc = zeros(T, 3 * num_new_vars, n + num_new_vars)
        cones = CO.Cone{T}[]

        offset = offset_next = 0
        row = 1
        # loop over layers, layer 1 describes hypograph variable
        for i in 1:(num_layers - 1)
            num_lvars = 2 ^ (i - 1)
            offset_next = offset + num_lvars
            # loop over variables in each layer
            for j in 1:num_lvars
                G_rsoc[row, n + offset_next + 2j - 1] = -1
                G_rsoc[row + 1, n + offset_next + 2j] = -1
                G_rsoc[row + 2, n + offset + j] = -1
                push!(cones, CO.EpiPerSquare{T}(3))
                row += 3
            end
            offset = offset_next
        end

        for j in 1:(2 ^ (num_layers - 1))
            # in the last layer, we use the original variables
            if 2j - 1 > n
                # if we are beyond the number of variables in the actual geometric mean, we are adding the buffer variable
                G_rsoc[row, n + 1] = -inv(rtfact)
            else
                G_rsoc[row, 2j - 1] = -1
            end
            if 2j > n
                # if we are beyond the number of variables in the actual geometric mean, we are adding the buffer variable
                G_rsoc[row + 1, n + 1] = -inv(rtfact)
            else
                G_rsoc[row + 1, 2j] = -1
            end
            G_rsoc[row + 2, n + offset + j] = -1
            push!(cones, CO.EpiPerSquare{T}(3))
            row += 3
        end

        # account for original hypograph variable
        G = [
            zeros(T, 3 * num_new_vars)  G_rsoc;
            one(T)  zeros(T, 1, n)  -inv(rtfact)  zeros(T, 1, num_new_vars - 1);
            ]
        push!(cones, CO.Nonnegative{T}(1))
        h = zeros(T, 3 * num_new_vars + 1)
    end

    model = Models.Model{T}(c, A, b, G, h, cones)
    return (model, ())
end

function test_maxvolume_native(result, test_helpers, test_options)
    @test result.status == :Optimal
end

options = ()
maxvolume_native_fast = [
    ((Float64, 3, true, false, false), (), options),
    ((Float64, 3, false, true, false), (), options),
    ((Float64, 3, false, false, true), (), options),
    ((Float64, 12, true, false, false), (), options),
    ((Float64, 12, false, true, false), (), options),
    ((Float64, 12, false, false, true), (), options),
    ((Float64, 100, true, false, false), (), options),
    ((Float64, 100, false, true, false), (), options),
    ((Float64, 100, false, false, true), (), options),
    ((Float64, 1000, true, false, false), (), options),
    ]
maxvolume_native_slow = [
    ((Float64, 1000, false, true, false), (), options),
    ((Float64, 1000, false, false, true), (), options),
    ((Float64, 1500, true, false, false), (), options),
    ((Float64, 1500, false, true, false), (), options),
    ((Float64, 1500, false, false, true), (), options),
    ]

# @testset begin test_native_instance.(maxvolume_native, test_maxvolume_native, maxvolume_native_fast) end
;
