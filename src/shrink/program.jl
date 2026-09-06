# What the reducer works on, how it is read from a script and how it is written back as source.

"""
    Value(data, text)

A captured value: `text` is source that rebuilds it; `data` is the number or `Array` behind
it, `nothing` for anything else.
"""
struct Value
    data::Union{Nothing, Number, Array}
    text::String
end
Value(x::Union{Number, Array}) = Value(x, repr(x))

"""
    value(x, m, file) -> Value or nothing

`x` as source in module `m`: a number or array as a literal, a global of `m` or a function of a
loaded package by name, anything else serialised to `file` and read back from there.
"""
function value(x, m::Module, file::String)
    x isa Number && return Value(x)
    x isa AbstractArray{<:Number} && return Value(x isa Array ? copy(x) : collect(x))
    name = global_name(x, m)
    name === nothing || return Value(nothing, string(name))
    mkpath(dirname(file))
    caught(serialize, file, x) === nothing || return nothing
    return Value(nothing, "deserialize($(repr(file)))")
end

struct Arg
    kind::Symbol
    name::Symbol
    val::Value
    shadows::Vector{Value}
end

"""
    Call(mode, f, ret, args)

The captured `autodiff(mode, f, ret, args...)`; `f` is a name, or the source of an anonymous
function until it is named.  A `checked` call is made through [`check`](@ref), the
derivative against finite differences.
"""
struct Call
    mode::Mode
    f::Union{Symbol, Expr}
    ret::Type
    args::Vector{Arg}
    checked::Bool
end
Call(mode, f, ret, args) = Call(mode, f, ret, args, false)

"""
    Program(defs, call, values)

The script's top-level `defs` before the call, the `call`, and the recorded `values` of
assignment statements, keyed by the statement `Expr` object itself: passes rebuild bodies from
the same objects, so a statement keeps its values through deletions and re-nesting.
"""
struct Program
    defs::Vector{Any}
    call::Call
    values::IdDict{Any, Vector{Value}}
end

with(p::Program; defs = p.defs, call = p.call) = Program(defs, call, p.values)
with(c::Call; mode = c.mode, f = c.f, ret = c.ret, args = c.args, checked = c.checked) = Call(mode, f, ret, args, checked)

"""
    script_expressions(file)
    script_definitions(file)

The top-level expressions of a script, literal `include`s inlined, `using A, B` split in two,
docstrings dropped: as parsed with line numbers, so that a function can be found in its source
again; or without them and with every function definition in `function f(x) ... end` form.
Both views hold the same expressions in the same order.
"""
script_expressions(file) = toplevel(file, false)
script_definitions(file) = toplevel(file, true)

function toplevel(file, strip::Bool, out = Any[])
    for e in Meta.parseall(read(file, String); filename = file).args
        add_toplevel!(out, e, file, strip)
    end
    return out
end

function add_toplevel!(out, e, file, strip)
    e isa LineNumberNode && return
    Meta.isexpr(e, :macrocall) && e.args[1] == GlobalRef(Core, Symbol("@doc")) && (e = e.args[end])
    if Meta.isexpr(e, (:toplevel, :block))                      # `a = 1; b = 2` on one line, or what a macro expanded to
        foreach(x -> add_toplevel!(out, x, file, strip), e.args)
    elseif Meta.isexpr(e, :call, 2) && e.args[1] == :include && e.args[2] isa String
        toplevel(joinpath(dirname(file), e.args[2]), strip, out)
    elseif Meta.isexpr(e, (:using, :import)) && length(e.args) > 1
        append!(out, [Expr(e.head, a) for a in e.args])
    else
        push!(out, strip ? normalize(strip_linenums(e)) : e)
    end
    return
end

is_signature(x) = Meta.isexpr(x, :call) || (Meta.isexpr(x, (:where, :(::))) && is_signature(x.args[1]))

# `Base.remove_linenums!` that also drops the line a macro call carries, which prints as a comment.
function strip_linenums(e)
    e isa Expr || return e
    Meta.isexpr(e, :macrocall) && (e.args[2] = nothing)
    Meta.isexpr(e, (:block, :quote)) && filter!(!Base.Fix2(isa, LineNumberNode), e.args)
    foreach(strip_linenums, e.args)
    return e
end

function normalize(e)
    (Meta.isexpr(e, :(=)) && is_signature(e.args[1])) || Meta.isexpr(e, :function, 2) || return e
    body = e.args[2]
    return Expr(:function, e.args[1], Meta.isexpr(body, :block) ? body : Expr(:block, body))
end

is_fdef(e) = Meta.isexpr(e, :function, 2)
call_part(sig) = sig.head == :call ? sig : call_part(sig.args[1])
fname(d) = (n = call_part(d.args[1]).args[1]; n isa Symbol ? n : nothing)
fparams(d) = call_part(d.args[1]).args[2:end]
stmts(d) = d.args[2].args
with_body(d, body::Expr) = Expr(:function, d.args[1], body)
with_body(d, body::Vector) = with_body(d, Expr(:block, body...))
function with_params(d, ps::Vector)
    rebuild(s) = s.head == :call ? Expr(:call, s.args[1], ps...) : Expr(s.head, rebuild(s.args[1]), s.args[2:end]...)
    return Expr(:function, rebuild(d.args[1]), d.args[2])
end

param_name(p) = p isa Symbol ? p : Meta.isexpr(p, (:(::), :kw, :(...))) ? param_name(p.args[1]) : nothing

is_assignment(s) = Meta.isexpr(s, :(=)) &&
    (s.args[1] isa Symbol || (Meta.isexpr(s.args[1], :tuple) && all(Base.Fix2(isa, Symbol), s.args[1].args)))
assigned_names(s) = s.args[1] isa Symbol ? [s.args[1]] : Symbol[s.args[1].args...]

mentions(name::Symbol, e) = e === name || (e isa Expr && any(a -> mentions(name, a), e.args))

function fresh(base::Symbol, taken)
    name, k = base, 1
    while name in taken
        name = Symbol(base, :_, k += 1)
    end
    return name
end

function bound_names(p::Program)
    names = Set{Symbol}(a.name for a in p.call.args)
    p.call.f isa Symbol && push!(names, p.call.f)
    for d in p.defs
        is_fdef(d) && fname(d) !== nothing && push!(names, fname(d))
    end
    return names
end

function map_blocks(f, s)
    s isa Expr || return s
    if s.head in (:for, :while, :let)
        return Expr(s.head, s.args[1], f(s.args[2]))
    elseif s.head in (:if, :elseif)
        return Expr(s.head, s.args[1], [Meta.isexpr(a, :block) ? f(a) : map_blocks(f, a) for a in s.args[2:end]]...)
    elseif s.head == :block
        return f(s)
    elseif s.head == :macrocall
        return Expr(:macrocall, s.args[1:(end - 1)]..., map_blocks(f, s.args[end]))
    end
    return s
end

function all_statements(body::Expr, out = Any[])
    for s in body.args
        push!(out, s)
        map_blocks(b -> (all_statements(b, out); b), s)
    end
    return out
end

"""
    map_statements(f, body)

Rebuild `body` with `f` applied to every statement at any depth; `f` returns a replacement,
`nothing` to delete the statement, or the statement itself to descend into it.
"""
function map_statements(f, body::Expr)
    out = Any[]
    for s in body.args
        r = f(s)
        r === nothing && continue
        r === s && (r = map_blocks(b -> map_statements(f, b), s))
        push!(out, r)
    end
    return Expr(:block, out...)
end

function entry(p::Program)
    p.call.f isa Symbol || return nothing
    found = findall(d -> is_fdef(d) && fname(d) == p.call.f, p.defs)
    return length(found) == 1 ? only(found) : nothing
end

function entry_is_called(p::Program)
    j = entry(p)
    j === nothing && return true
    return any(i -> mentions(p.call.f, i == j ? p.defs[i].args[2] : p.defs[i]), eachindex(p.defs))
end

# An anonymous or local function captured as source becomes a top-level definition, named `entry` if it had no name.
function name_the_function(p::Program)
    src = p.call.f
    Meta.isexpr(src, (:->, :function)) || return p
    if src.head == :function && is_signature(src.args[1])
        def = normalize(src)
        name = fname(def)
        name === nothing && return p
    else
        ps = Meta.isexpr(src.args[1], :tuple) ? src.args[1].args : Any[src.args[1]]
        captured = [a.name for a in p.call.args[(length(ps) + 1):end]]    # what the closure captured, made inputs by `describe`
        name = fresh(:entry, bound_names(p))
        def = normalize(Expr(:function, Expr(:call, name, ps..., captured...), src.args[2]))
    end
    return with(p; defs = [p.defs; def], call = with(p.call; f = name))
end

abi(::Mode{ABI}) where {ABI} = ABI

function mode_source(m::Mode, pre::String)
    base = EnzymeCore.set_abi(EnzymeCore.clear_strong_zero(EnzymeCore.clear_runtime_activity(m)), Enzyme.DefaultABI)
    name = base == Reverse ? "Reverse" : base == ReverseWithPrimal ? "ReverseWithPrimal" :
        base == Forward ? "Forward" : base == ForwardWithPrimal ? "ForwardWithPrimal" : nothing
    name === nothing && return repr(m)
    text = pre * name
    EnzymeCore.runtime_activity(m) && (text = "$(pre)set_runtime_activity($text)")
    EnzymeCore.strong_zero(m) && (text = "$(pre)set_strong_zero($text)")
    abi(m) === Enzyme.DefaultABI || (text = "$(pre)set_abi($text, $pre$(nameof(abi(m))))")
    return text
end

shadow_name(a::Arg, k) = length(a.shadows) == 1 ? Symbol(:d, a.name) : Symbol(:d, a.name, :_, k)

function call_line(c::Call, pre::String)
    function annotated(a::Arg)
        shadows = [string(shadow_name(a, k)) for k in eachindex(a.shadows)]
        isempty(shadows) && return "$pre$(a.kind)($(a.name))"
        length(shadows) == 1 && return "$pre$(a.kind)($(a.name), $(only(shadows)))"
        return "$pre$(a.kind)($(a.name), ($(join(shadows, ", ")),))"
    end
    parts = [mode_source(c.mode, pre), string(c.f), pre * string(nameof(c.ret)), annotated.(c.args)...]
    return (c.checked ? "Enzyme.Shrink.check" : pre * "autodiff") * "(" * join(parts, ", ") * ")"
end

primal_line(c::Call; copy = false) = "$(c.f)(" * join([copy ? "deepcopy($(a.name))" : string(a.name) for a in c.args], ", ") * ")"

# The definition as it was probably written: a lambda, or a short single-expression function, on one line.
function compact(e)
    e isa Expr || return e
    e = Expr(e.head, map(compact, e.args)...)
    if Meta.isexpr(e, :->, 2) && Meta.isexpr(e.args[2], :block, 1)
        e = Expr(:->, e.args[1], only(e.args[2].args))
    elseif Meta.isexpr(e, :function, 2) && Meta.isexpr(e.args[2], :block, 1)
        body = only(e.args[2].args)
        Meta.isexpr(body, :return, 1) && (body = body.args[1])
        line = Expr(:(=), e.args[1], body)
        Meta.isexpr(body, (:for, :while, :if, :let, :try)) || length(string(line)) > 92 || (e = line)
    end
    return e
end

"""
    source(p) -> (setup, call)

The definitions and argument values, and the `autodiff` line; Enzyme's names are bare when the
script does `using Enzyme`, else qualified.
"""
function source(p::Program)
    pre = :(using Enzyme) in p.defs ? "" : "Enzyme."
    io = IOBuffer()
    isempty(pre) || :(import Enzyme) in p.defs || println(io, "import Enzyme")
    any(a -> any(startswith("deserialize("), [a.val.text; [s.text for s in a.shadows]]), p.call.args) && println(io, "using Serialization")
    for d in p.defs
        println(io, compact(d), "\n")
    end
    for a in p.call.args
        a.val.text == string(a.name) || println(io, a.name, " = ", a.val.text)   # a function argument needs no assignment
        for (k, s) in enumerate(a.shadows)
            println(io, shadow_name(a, k), " = ", s.data isa Array && all(iszero, s.data) ? "zero($(a.name))" : s.text)
        end
    end
    return String(take!(io)), call_line(p.call, pre)
end

one_line(text, n = 200) = (t = join(strip.(split(text, '\n')), " "); length(t) > n ? first(t, n) * "…" : t)

function write_repro(path::String, p::Program, class, original::String; notes = String[])
    setup, call = source(p)
    open(path, "w") do io
        println(io, "# Reduced by Enzyme.shrink, Julia $VERSION, Enzyme $(pkgversion(Enzyme))")
        println(io, "# Failure: ", replace(string(class), '\n' => "\n#          "))
        println(io, "# Reduced from: ", one_line(original))
        foreach(note -> println(io, "# ", note), notes)
        println(io)
        print(io, setup)
        println(io, call)
    end
    return
end
