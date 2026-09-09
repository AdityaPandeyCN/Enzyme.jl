# Making programs smaller.  Each pass proposes smaller programs along one axis, as `label => program`
# in the order to try them, coarse cuts first; the loop keeps the first the oracle accepts.

const Proposals = Vector{Pair{String, Program}}

replace_def(p::Program, i, d) = with(p; defs = [j == i ? d : p.defs[j] for j in eachindex(p.defs)])
replace_arg(c::Call, i, a) = with(c; args = [j == i ? a : c.args[j] for j in eachindex(c.args)])

"""
    chunks(items; singles = false)

Consecutive runs of `items` at every power-of-two size, largest first, each once:
`chunks([2, 3, 5])` is `[[2, 3], [5], [2], [3]]`; with `singles`, only the runs of one.
"""
function chunks(items::AbstractVector; singles::Bool = false)
    out = Vector{eltype(items)}[]
    singles && return [[x] for x in items]
    isempty(items) && return out
    for size in (2^k for k in floor(Int, log2(length(items))):-1:0), c in Iterators.partition(items, size)
        c = collect(c)
        c in out || push!(out, c)
    end
    return out
end

function bisect_order(n)
    out = Int[]
    queue = [(1, n)]
    while !isempty(queue)
        lo, hi = popfirst!(queue)
        lo > hi && continue
        mid = (lo + hi) ÷ 2
        push!(out, mid)
        push!(queue, (lo, mid - 1), (mid + 1, hi))
    end
    return out
end

function placeholder_arg(name::Symbol, v::Value, mode::Mode)
    kind = nameof(Enzyme.guess_activity(typeof(v.data), mode))
    kind in (:Active, :Duplicated) || return Arg(:Const, name, v, Value[])
    shadows = kind == :Duplicated ? [Value(v.data isa Number ? one(v.data) : Enzyme.make_zero(v.data))] : Value[]
    return Arg(kind, name, v, shadows)
end

# Keep only the first `k` statements of the entry function and return the `k`-th's value, `k` in bisection order.
function remove_suffix(p::Program)
    out = Proposals()
    i = entry(p)
    i === nothing && return out
    d = p.defs[i]
    body = stmts(d)
    for k in bisect_order(length(body) - 1)
        s = body[k]
        is_assignment(s) && length(assigned_names(s)) == 1 && haskey(p.values, s) || continue
        y, data = only(assigned_names(s)), only(p.values[s]).data
        p.call.ret === Active && Enzyme.guess_activity(typeof(data), Reverse) <: Const && continue
        ret = p.call.ret !== Active || data isa Real ? y : :(sum(abs2, $y))
        new = [body[1:k]; Expr(:return, ret)]
        new == body && continue
        push!(out, "$(fname(d)): keep statements 1:$k, return $ret" => replace_def(p, i, with_body(d, new)))
    end
    return out
end

"""
    placeholder_statements(p)

Delta debugging over the statements of every named function at any depth, the entry first.
A chunk of statements is replaced at once: an assignment with a recorded value becomes
`y = y_rec`, with `y_rec` a new input of the entry function carrying the value, so the code
after it still works; anything else is deleted.  In a function whose parameter list cannot
change, statements are only deleted.  A function's last statement is never touched.
"""
function placeholder_statements(p::Program; singles::Bool = false)
    out = Proposals()
    e = entry(p)
    order = [i for i in eachindex(p.defs) if is_fdef(p.defs[i]) && fname(p.defs[i]) !== nothing]
    e === nothing || (order = [e; filter(!=(e), order)])
    for i in order
        d = p.defs[i]
        nodes = filter(s -> s !== last(stmts(d)), all_statements(d.args[2]))
        for chunk in chunks(1:length(nodes); singles)
            which = length(chunk) == 1 ? "`$(one_line(string(nodes[only(chunk)]), 60))`" : "statements $(first(chunk)):$(last(chunk)) of $(length(nodes))"
            push!(out, "without $which in $(fname(d))" => replace_statements(p, i, nodes[chunk], i == e && !entry_is_called(p)))
        end
    end
    return out
end

function replace_statements(p::Program, i, targets, as_inputs::Bool)
    d = p.defs[i]
    params, args, taken = copy(fparams(d)), copy(p.call.args), bound_names(p)
    replacement = IdDict{Any, Any}()
    for s in targets
        vals = get(p.values, s, nothing)
        if !(as_inputs && is_assignment(s) && vals !== nothing)
            replacement[s] = nothing
            continue
        end
        inputs = map(zip(assigned_names(s), vals)) do (y, v)
            name = fresh(Symbol(y, :_rec), taken)
            push!(taken, name)
            push!(params, name)
            push!(args, placeholder_arg(name, v, p.call.mode))
            name
        end
        replacement[s] = Expr(:(=), s.args[1], length(inputs) == 1 ? only(inputs) : Expr(:tuple, inputs...))
    end
    body = map_statements(s -> haskey(replacement, s) ? replacement[s] : s, d.args[2])
    return with(replace_def(p, i, with_params(with_body(d, body), params)); call = with(p.call; args))
end

function eliminate_dead_code(p::Program)
    out = Proposals()
    for (i, d) in enumerate(p.defs)
        is_fdef(d) && fname(d) !== nothing || continue
        body = stmts(d)
        dead = [
            k for k in 1:(length(body) - 1) if is_assignment(body[k]) &&
                !any(y -> any(s -> mentions(y, s), body[(k + 1):end]), assigned_names(body[k]))
        ]
        isempty(dead) && continue
        push!(out, "$(fname(d)): remove dead statements $(join(dead, ","))" => replace_def(p, i, with_body(d, body[setdiff(1:length(body), dead)])))
    end
    return out
end

function remove_unused_args(p::Program)
    out = Proposals()
    i = entry(p)
    (i === nothing || entry_is_called(p)) && return out
    d = p.defs[i]
    ps = fparams(d)
    length(ps) == length(p.call.args) && all(x -> param_name(x) !== nothing && !Meta.isexpr(x, (:kw, :(...))), ps) || return out
    unused = [k for (k, x) in enumerate(ps) if !mentions(param_name(x), d.args[2])]
    for group in chunks(unused)
        keep = setdiff(1:length(ps), group)
        names = join((param_name(ps[k]) for k in group), ", ")
        push!(
            out, "$(fname(d)): drop unused argument$(length(group) > 1 ? "s" : "") $names" =>
                with(replace_def(p, i, with_params(d, ps[keep])); call = with(p.call; args = p.call.args[keep]))
        )
    end
    return out
end

function remove_defs(p::Program)
    out = Proposals()
    is_enzyme_import(d) = Meta.isexpr(d, (:using, :import)) && mentions(:Enzyme, d)
    removable = [i for i in eachindex(p.defs) if i != entry(p) && !is_enzyme_import(p.defs[i])]
    for chunk in chunks(removable)
        push!(out, "remove definition$(length(chunk) == 1 ? "" : "s") $(join(chunk, ","))" => with(p; defs = p.defs[setdiff(eachindex(p.defs), chunk)]))
    end
    return out
end

function simplify_call(p::Program)
    out = Proposals()
    m = p.call.mode
    propose(label, mode) = push!(out, label => with(p; call = with(p.call; mode)))
    EnzymeCore.runtime_activity(m) && propose("clear runtime activity", EnzymeCore.clear_runtime_activity(m))
    EnzymeCore.strong_zero(m) && propose("clear strong zero", EnzymeCore.clear_strong_zero(m))
    m == ReverseWithPrimal && propose("drop WithPrimal", Reverse)
    m == ForwardWithPrimal && propose("drop WithPrimal", Forward)
    p.call.ret === Const || push!(out, "return $(nameof(p.call.ret)) → Const" => with(p; call = with(p.call; ret = Const)))
    return out
end

function drop_activity(p::Program)
    out = Proposals()
    for (i, a) in enumerate(p.call.args)
        a.kind == :Const && continue
        push!(out, "with $(a.name) Const" => with(p; call = replace_arg(p.call, i, Arg(:Const, a.name, a.val, Value[]))))
    end
    return out
end

extents(n) = unique([1; [n - n ÷ 2^k for k in 1:floor(Int, log2(n))]])

"""
    shrink_arrays(p)

Smaller array arguments: dimensions sharing an extent are cut together first, then each alone.
"""
function shrink_arrays(p::Program)
    out = Proposals()
    dims = Pair{Tuple{Int, Int}, Int}[]                    # (argument, dimension) => extent
    for (i, a) in enumerate(p.call.args), (dim, n) in enumerate(a.val.data isa Array ? size(a.val.data) : ())
        n > 1 && push!(dims, (i, dim) => n)
    end
    groups = [[first(d) for d in dims if last(d) == n] for n in sort!(unique(last.(dims)); rev = true)]
    append!(groups, [[first(d)] for d in dims if count(x -> last(x) == last(d), dims) > 1])
    for group in groups
        n = size(p.call.args[group[1][1]].val.data, group[1][2])
        for k in extents(n)
            label = length(group) == 1 ? "arg $(p.call.args[group[1][1]].name) dim $(group[1][2]): $n → $k" : "dims with extent $n → $k"
            push!(out, label => cut_dims(p, group, k))
        end
    end
    return out
end

function cut_dims(p::Program, dims, k)
    args = copy(p.call.args)
    for (i, dim) in dims
        take(x) = x isa Array ? Value(x[ntuple(d -> d == dim ? (1:k) : Colon(), ndims(x))...]) : nothing
        a = args[i]
        args[i] = Arg(a.kind, a.name, take(a.val.data), [something(take(s.data), s) for s in a.shadows])
    end
    return with(p; call = with(p.call; args))
end

const PASSES = (simplify_call, drop_activity, remove_suffix, placeholder_statements, shrink_arrays)
const CLEANUPS = (remove_unused_args, eliminate_dead_code, remove_defs)

size_summary(p::Program) = (i = entry(p); "defs $(length(p.defs)), statements $(i === nothing ? "?" : length(all_statements(p.defs[i].args[2]))), args $(length(p.call.args))")

"""
    probes(p)

The program with each of runtime activity, strong zero and the inline ABI switched on where it
is not already; not reductions, their verdicts go in the repro's header.
"""
function probes(p::Program)
    m = p.call.mode
    variants = Pair{String, Mode}[]
    EnzymeCore.runtime_activity(m) || push!(variants, "set_runtime_activity" => EnzymeCore.set_runtime_activity(m))
    EnzymeCore.strong_zero(m) || push!(variants, "set_strong_zero" => EnzymeCore.set_strong_zero(m))
    abi(m) === Enzyme.InlineABI || push!(variants, "InlineABI" => EnzymeCore.set_abi(m, Enzyme.InlineABI))
    return [label => with(p; call = with(p.call; mode)) for (label, mode) in variants]
end

verdict(class::Class, target::Class) =
    class == target ? "same failure" : class == PASS ? "passes" : "different failure: $(class.kind)"

"""
    ingredients(p, o, target, known) -> Vector{String}

Why each statement and each active argument of the final program is there: what the program
does without it, from `known` where the last round already tried that.
"""
ingredients(p::Program, o::Oracle, target::Class, known::Dict{Any, Class}) = [
    "$label: $(verdict(get!(() -> o(q)[2], known, proposal_key(p, q)), target))"
        for (label, q) in [placeholder_statements(p; singles = true); drop_activity(p)]
]

# CLEANUPS then PASSES, each to a fixpoint, until nothing changes; every accepted program is checkpointed.
function minimize(p::Program, o::Oracle, target::Class, original::String)
    checkpoint(q) = write_repro(joinpath(o.dir, "checkpoints", lpad(o.count, 4, '0') * ".jl"), q, target, original)
    checkpoint(p)
    println("start: ", size_summary(p))
    known = Dict{Any, Class}()
    changed = true
    while changed
        p, _ = apply(CLEANUPS, p, o, target, known, checkpoint)
        p, changed = apply(PASSES, p, o, target, known, checkpoint)
    end
    return p, known
end

function apply(passes, p::Program, o::Oracle, target::Class, known::Dict{Any, Class}, checkpoint)
    changed = false
    for pass in passes
        while (next = first_accepted(pass(p), p, o, target, known)) !== nothing
            p = next
            changed = true
            checkpoint(p)
        end
    end
    return p, changed
end

"""
    proposal_key(p, q)

What proposal `q` changes in `p`, with the call it is made under.  A rejected proposal is not
tried again until the function it edits, the call's activity or the input values change: its
class is taken to be what it was.
"""
proposal_key(p::Program, q::Program) =
    (filter(!in(p.defs), q.defs), filter(!in(q.defs), p.defs), call_line(q.call, ""), [a.val.text for a in q.call.args])

function first_accepted(proposals::Proposals, p::Program, o::Oracle, target::Class, known::Dict{Any, Class})
    for (label, q) in proposals
        key = proposal_key(p, q)
        haskey(known, key) && continue
        id, class = o(q)
        known[key] = class
        accepted = class == target
        println("[", id, "] ", rpad(label, 48), " ", accepted ? "kept  → " * size_summary(q) : "different: $(class.kind)")
        flush(stdout)
        accepted && return q
    end
    return nothing
end
