# Catching the failing call.  Enzyme's two core `autodiff` methods call `describe_call` when
# `CAPTURING[]` is set, before doing anything else; it describes the call and makes it once
# itself, on copies, with the derivative checked.  The script's top-level expressions are run
# one at a time; when one throws after a call was described, that call is the failure.

const CAPTURING = Ref(false)

mutable struct Session
    script::Module                       # the module the script runs in; calls are described by its globals
    index::Int                           # the top-level expression being run
    call::Union{Nothing, Call, String}   # the last call described in that expression, or why it could not be
    pending::String                      # in a child process: file the description is written to before the call runs
end
const SESSION = Session(Main, 0, nothing, "")

@noinline function describe_call(@nospecialize(mode::Mode), @nospecialize(f::Annotation), @nospecialize(ret::Type), @nospecialize(args::Vararg{Annotation}))
    SESSION.call = describe(mode, f, ret, args, SESSION.script)
    isempty(SESSION.pending) || serialize_atomically(SESSION.pending, (SESSION.index, SESSION.call))
    CAPTURING[] = false                                          # the check's own call must not come back here
    try
        check(mode, f, ret, map(copied, args)...)
    finally
        CAPTURING[] = true
    end
    return nothing
end

"""
    describe(mode, f, ret, args, m) -> Call or String

Describe `autodiff(mode, f, ret, args...)` in terms of the script module `m`: the function
and each argument are named after the globals of `m` bound to them (else `a1`, `a2`, ...)
and the values are copied, since `autodiff` accumulates into the shadows.  Return the reason
as a string when the call cannot be written as source.
"""
function describe(@nospecialize(mode::Mode), @nospecialize(f::Annotation), @nospecialize(ret::Type), @nospecialize(args::Tuple), m::Module)
    f isa Const || return "only a bare or `Const` function is reduced, not $(nameof(typeof(f)))"
    name = global_name(f.val, m)
    name === nothing && (name = local_source(f.val))
    name === nothing && return "cannot name the differentiated function, of type $(typeof(f.val))"
    taken = Set{Symbol}(name isa Symbol ? [name] : Symbol[])
    described = Arg[]
    for (i, a) in enumerate(args)
        n = fresh(something(global_name(a.val, m), Symbol(:a, i)), taken)
        push!(taken, n)
        val = value(a.val, m)
        val === nothing && return "argument $n of type $(typeof(a.val)) cannot be written as source"
        shadows = shadow_values(a, m)
        shadows === nothing && return "a shadow of argument $n cannot be written as source"
        push!(described, Arg(nameof(typeof(a)), n, val, shadows))
    end
    return Call(EnzymeCore.clear_err_if_func_written(mode), name, ret, described)   # set again when `f` is passed bare
end

shadow_values(a::Union{Const, Active}, m) = Value[]
shadow_values(a::Union{Duplicated, DuplicatedNoNeed}, m) = shadow_values((a.dval,), m)
shadow_values(a::Union{BatchDuplicated, BatchDuplicatedNoNeed}, m) = shadow_values(a.dval, m)
shadow_values(a::Annotation, m) = nothing
function shadow_values(dvals::Tuple, m)
    vals = [value(d, m) for d in dvals]
    return any(isnothing, vals) ? nothing : Value[vals...]
end

function global_name(x, m::Module)
    for n in names(m; all = true)
        Base.isidentifier(n) && isdefined(m, n) && !isconst(m, n) && getglobal(m, n) === x && return n
    end
    x isa Function || return nothing
    n = nameof(x)
    Base.isidentifier(n) || return nothing
    isdefined(m, n) && getglobal(m, n) === x && return n
    return parentmodule(x) === Enzyme ? :(Enzyme.$n) : nothing
end

function local_source(f)
    f isa Function && length(methods(f)) == 1 || return nothing
    method = only(methods(f))
    file = string(method.file)
    isfile(file) || return nothing
    found = find_function(Meta.parseall(read(file, String); filename = file), Int(method.line))
    return found === nothing ? nothing : strip_linenums(found)
end

function find_function(e, line)
    e isa Expr || return nothing
    if Meta.isexpr(e, (:->, :function), 2) && e.args[2] isa Expr
        first_node = findfirst(Base.Fix2(isa, LineNumberNode), e.args[2].args)
        first_node !== nothing && e.args[2].args[first_node].line == line && return e
    end
    for a in e.args
        found = find_function(a, line)
        found === nothing || return found
    end
    return nothing
end

"""
    run_script(file; pending = "") -> (index, call, exception) or nothing

Run the script's top-level expressions one at a time, with capture on, up to the first that
throws: its index, the last call described in it (a `Call`, why there is none, or `nothing`)
and the exception.  With `pending`, each description is written there before its call runs.
"""
function run_script(file::String; pending::String = "")
    exprs = script_expressions(file)
    m = Module(:Script)
    Core.eval(m, :(eval(x) = Core.eval($m, x)))
    Core.eval(m, :(include(x) = Base.include($m, x)))
    SESSION.script, SESSION.pending = m, pending
    CAPTURING[] = true
    result = cd(dirname(file)) do
        for (i, e) in enumerate(exprs)
            SESSION.index, SESSION.call = i, nothing
            isempty(pending) || rm(pending; force = true)
            err = caught(Core.eval, m, e)
            err === nothing || return (i, SESSION.call, err)
        end
        nothing
    end
    CAPTURING[] = false
    SESSION.script, SESSION.call, SESSION.pending = Main, nothing, ""
    return result
end

function write_typed_ir(path::String, err::Exception)
    err = unwrap(err)
    mi = hasfield(typeof(err), :mi) ? err.mi : nothing
    mi isa Core.MethodInstance && hasmethod(InteractiveUtils.code_typed, Tuple{typeof(err)}) || return
    failure = caught() do
        ir = InteractiveUtils.code_typed(err)
        open(path, "w") do io
            Compiler.pretty_print_mi(mi, io)
            println(io, "\n")
            show(io, MIME"text/plain"(), ir)
        end
    end
    failure === nothing || @warn "the typed IR of the failing method could not be written" exception = failure
    return
end

function capture_here(base::String, file::String, dir::String)
    captured = run_script(file; pending = base * ".pending")
    captured === nothing && return nothing
    index, call, err = captured
    write_typed_ir(joinpath(dir, "typed_ir.txt"), err)
    return index, call, classify(err)
end

"""
    capture_failure(file, o) -> (index, call, class) or nothing

The script's first failure, caught in this process or in the oracle's child (a call the child
died in is read back from its pending file); the failing method's typed IR goes to `typed_ir.txt`.
"""
function capture_failure(file::String, o::Oracle)
    base = joinpath(o.dir, "capture")
    result = perform(o, base, :capture_here, file, o.dir)
    result isa Class || return result
    isfile(base * ".pending") || error("the script $(result.kind == :crash ? "crashed" : "timed out") outside any autodiff call; see $base.log")
    index, call = deserialize(base * ".pending")
    return index, call, result
end
