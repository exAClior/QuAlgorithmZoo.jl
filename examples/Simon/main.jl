# Simon's Algorithm 
## Setting
# Given a black box function f : {0, 1}^n -> X where X ⊆ {0, 1}^n, 
# such that ∃ s ∈ {0, 1}^n, such that f(x) = f(y) ⟺ x = y or x = y ⊕ s 
# Find s 
using Pkg;
Pkg.activate(dirname(@__FILE__));
using Yao, Random, SparseArrays, LinearAlgebra

function back_substitution(l::AbstractMatrix)
    n = size(l, 2)
    query_order = reverse([findfirst(isone, l[ii, :]) for ii in 1:n-1])
    ans1 = [ii ∈ query_order ? 0 : 1 for ii in 1:n]
    ans2 = zeros(Int, n)

    for (ii, jj) in enumerate(query_order)
        ans1[jj] = foldr(⊻, l[n-ii, :] .* ans1[:])
        ans2[jj] = foldr(⊻, l[n-ii, :] .* ans2[:])
    end
    if ans2 == zeros(Int, n)
        return ans1
    else
        return ans2
    end
end

function gaussian_elim_mod2(W::MT) where {T,MT<:AbstractMatrix{T}}
    rows, cols = size(W)
    W = copy(W)
    W = sortslices(W, dims=1, rev=true)
    for mm in 1:rows
        nz_pos = findfirst(!iszero, W[mm, :])
        for nn in mm+1:rows
            if iszero(W[nn, nz_pos])
                continue
            end
            W[nn, :] = W[nn, :] .⊻ W[mm, :]
        end
        W = sortslices(W, dims=1, rev=true)
    end
    return W
end

function my_rank(W::MT) where {T,MT<:AbstractMatrix{T}}
    elim_W = gaussian_elim_mod2(W)
    return sum([!all(iszero, elim_W[ii, :]) for ii in 1:size(elim_W, 1)])
end

function main()
    n = 5
    Random.seed!(123)
    for _ in 1:100
        target_s = rand(1:2^n-2)
        # target_s = 27
        # target_s = 11
        # target_s = 2
        # target_s = 25
        target_s_bits = Bool.(digits(target_s, base=2, pad=n))
        # create arbitrary 2->1 function f 
        mapped_vals = shuffle(0:2^n-1)
        for x in 0:2^n-1
            y = x ⊻ target_s
            mapped_vals[y+1] = mapped_vals[x+1]
        end

        Is = Int[]
        Js = Int[]

        for xx in 0:2^n-1
            for bb in 0:2^n-1
                push!(Is, xx + (bb ⊻ mapped_vals[xx+1]) << n + 1)
                push!(Js, xx + bb << n + 1)
            end
        end

        Uf_mtx = sparse(Is, Js, ones(ComplexF64, length(Is)), 2^(2 * n), 2^(2 * n))

        Uf = GeneralMatrixBlock(Uf_mtx)

        simon_circ = chain(2 * n, repeat(2 * n, H, 1:n), put(2 * n, (1:2*n) => Uf), repeat(2 * n, H, 1:n))
        wis = Set{Vector{Int}}()
        rank_of_span = 0
        while rank_of_span < n - 1
            init_state = zero_state(2 * n)
            measure_result = measure(apply(init_state, simon_circ), 1:n)
            wi = [Int(bit) for bit in measure_result[]]
            if wi != zeros(Int, n) && my_rank(vcat([wii' for wii in wis]..., wi')) > rank_of_span
                push!(wis, wi)
                rank_of_span = my_rank(vcat([wi' for wi in wis]...))
            end
        end
        W = vcat([wi' for wi in wis]...)


        elim_W = gaussian_elim_mod2(W)
        # implement mod2 back substitution

        # @show target_s
        # @show target_s_bits
        # display(W)
        # display(elim_W)
        ans_s = back_substitution(copy(elim_W))

        # @show ans_s
        # yeah! the answer matches!
        @assert mod.(W * ans_s, 2) == zeros(Int, n - 1)
        @assert ans_s == target_s_bits
    end
end

main()
