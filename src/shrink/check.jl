# A wrong derivative as a failure: `check` is `autodiff` followed by a comparison with central
# finite differences of the primal, and throws when they disagree.

# Thrown by `check`: the derivative called `where` disagrees with finite differences.
struct WrongDerivative <: Exception
    where::String
    detail::String
end
Base.showerror(io::IO, e::WrongDerivative) = print(io, "WrongDerivative: the ", e.where, " disagrees with finite differences, ", e.detail)

"""
    check(mode, f, ret, args...)

`autodiff(mode, f, ret, args...)`, then its derivative against central finite differences of
`f`: in forward mode the tangent of the return value and of every `Duplicated` argument, in
reverse mode the gradient along one direction.  Throws `WrongDerivative` when they
disagree, else returns what `autodiff` returned.  Only real floating-point numbers and arrays
are compared; a call with any other differentiable value, or with batches, is not checked.
"""
function check(mode::Mode, f, ret::Type{<:Annotation}, args::Vararg{Annotation})
    before = map(copied, args)                                   # f and autodiff may write into them
    result = autodiff(mode, f, ret, args...)
    all(checkable, args) && !all(Base.Fix2(isa, Const), args) && ret <: Union{Const, Active, Duplicated, DuplicatedNoNeed} || return result
    compare(mode, f isa Annotation ? f.val : f, ret, before, args, result)
    return result
end

copied(a::Const) = a
copied(a::Annotation) = deepcopy(a)

floats(x) = x isa AbstractFloat || x isa AbstractArray{<:AbstractFloat}
checkable(a::Const) = true
checkable(a::Active) = floats(a.val)
checkable(a::Duplicated) = floats(a.val) && floats(a.dval)
checkable(a::Annotation) = false

function compare(mode::ForwardMode, f, ret, before, after, result)
    h = fd_step(before)
    plus, minus = probe(f, before, map(seed, before), h), probe(f, before, map(seed, before), -h)
    for (k, a) in enumerate(after)
        a isa Duplicated || continue
        d = disagreement(a.dval, (plus[2][k] - minus[2][k]) / 2h)
        d === nothing || throw(WrongDerivative("tangent of argument $k", d))
    end
    (ret <: Const || !floats(result[1])) && return
    d = disagreement(result[1], (plus[1] - minus[1]) / 2h)
    d === nothing || throw(WrongDerivative("tangent of the return value", d))
    return
end

function compare(mode::ReverseMode, f, ret, before, after, result)
    u = map(direction, before)
    h = fd_step(before)
    plus, minus = probe(f, before, u, h), probe(f, before, u, -h)
    lhs, rhs = 0.0, ret <: Active ? (plus[1] - minus[1]) / 2h : 0.0
    for (k, a) in enumerate(after)
        a isa Active && (lhs += result[1][k] * u[k])
        a isa Duplicated || continue
        lhs += sum(a.dval .* u[k])                                        # the adjoint of the input
        rhs += sum(before[k].dval .* (plus[2][k] - minus[2][k]) ./ 2h)   # the seed against the output's directional derivative
    end
    d = disagreement(lhs, rhs)
    d === nothing || throw(WrongDerivative("gradient", "along one direction: $d"))
    return
end

seed(a::Duplicated) = a.dval
seed(a::Annotation) = nothing
direction(a::Active) = one(a.val)
direction(a::Duplicated) = a.val isa Number ? one(a.val) : reshape(eltype(a.val).(sin.(1:length(a.val))), size(a.val))
direction(a::Annotation) = nothing

perturbed(a::Const, u, h) = a.val
perturbed(a::Annotation, u, h) = a.val .+ h .* u
function probe(f, args, u, h)
    xs = map(perturbed, args, u, ntuple(Returns(h), length(args)))
    return f(xs...), xs
end

function fd_step(args)
    xs = [a.val for a in args if !(a isa Const)]
    T = promote_type(float.(eltype.(xs))...)
    return cbrt(eps(T)) * max(one(T), maximum([maximum(abs, x) for x in xs]))
end

# Where Enzyme's `a` and the finite differences `b` differ beyond round-off, or `nothing`.
function disagreement(a, b)
    err, i = findmax(abs.(a .- b))
    err <= sqrt(eps(float(eltype(b)))) * (1 + 100maximum(abs, b)) && return nothing
    return (length(b) == 1 ? "" : "largest at $(Tuple(i)): ") * "Enzyme $(a[i]), finite differences $(b[i])"
end
