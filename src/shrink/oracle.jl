# Judging programs: an outcome is a `Class`, the oracle runs a candidate and returns its class, in a
# fresh module of this process or in a child process, and the primal is run once to record its values.

"""
    Class(kind, key, detail)

Identity of a failure: `kind` is the exception type (or `:pass`, `:setup`, `:crash`, `:timeout`),
`key` says where, `detail` is for display; two outcomes are the same failure when `kind` and
`key` agree.  Enzyme's own exceptions are keyed by the method they blame, which survives the
program around it being cut down; other exceptions by the first line of their message with
numbers normalised; a wrong derivative by which derivative disagreed; a crash by its
assertion or internal error message, and a plain signal death by nothing at all, memory
corruption ending in whichever signal comes first.
"""
struct Class
    kind::Symbol
    key::String
    detail::String
end
Base.:(==)(a::Class, b::Class) = a.kind == b.kind && a.key == b.key
Base.hash(c::Class, h::UInt) = hash((c.kind, c.key), h)
Base.show(io::IO, c::Class) = print(io, c.kind, isempty(c.detail) ? "" : ": " * c.detail)
const PASS = Class(:pass, "", "")

normalize_numbers(s) = replace(s, r"%\d+" => "%N", r"0x[0-9a-fA-F]+" => "0xX", r"\d+" => "N")
unwrap(err) = err isa LoadError ? unwrap(err.error) : err

function classify(@nospecialize(err::Exception))
    err = unwrap(err)
    kind = nameof(typeof(err))
    msg = hasfield(typeof(err), :msg) && err.msg isa AbstractString ? err.msg : Base.invokelatest(sprint, showerror, err)
    lines = filter(!isempty, strip.(split(msg, '\n')))
    line = String(chopprefix(get(lines, 1, ""), "$kind: "))
    err isa WrongDerivative && return Class(kind, err.where, line)
    err isa Union{Compiler.EnzymeError, Compiler.CustomRuleError} || return Class(kind, normalize_numbers(line), line)
    mi = hasfield(typeof(err), :mi) ? err.mi : nothing
    if mi isa Core.MethodInstance && mi.def isa Method
        method = string(mi.def.name)
        startswith(method, "#") && (method = normalize_numbers(method))       # only compiler-generated names carry counters
        return Class(kind, "in " * method, "in $method: $line")
    end
    return Class(kind, err isa Compiler.EnzymeInternalError ? normalize_numbers(line) : "", line)
end

function crash_class(log::AbstractString)
    lines = split(log, '\n')
    i = findlast(l -> occursin(r"Assertion|LLVM ERROR|EnzymeInternalError", l), lines)
    key = i === nothing ? "" : normalize_numbers(lines[i])
    i === nothing && (i = findlast(l -> occursin(r"signal \(\d+\)|signal \d+ \(", l), lines))
    detail = i === nothing ? "process died" : replace(String(strip(lines[i])), r"^\[\d+\] " => "")
    return Class(:crash, key, detail)
end

# The exception `f(args...)` throws, or `nothing`.
function caught(f, args...)
    try
        f(args...)
    catch err
        return err
    end
    return nothing
end

outcome(f, args...) = (err = caught(f, args...); err === nothing ? PASS : classify(err))

logged(f, logfile::String) = open(io -> redirect_stdio(f; stdout = io, stderr = io), logfile, "w")

"""
    evaluate(base, setup, call, primal, expected) -> Class

The class of `call` after `setup` in a fresh module; `:setup` when that fails, or when `primal`
neither passes nor ends in `expected`.
"""
function evaluate(base::String, setup::String, call::String, primal, expected)
    m, file = Module(Symbol(:Candidate_, basename(base))), basename(base) * ".jl"
    c = outcome(Base.include_string, m, setup, file)
    c == PASS || return Class(:setup, "", c.detail)
    if primal !== nothing
        c = outcome(Base.include_string, m, primal, file)
        c == PASS || c == expected || return Class(:setup, "", "the primal: $c")
    end
    touch(base * ".primal")
    return outcome(Base.include_string, m, call, file)
end

# A Julia process with Enzyme loaded that runs jobs until it dies; `log` collects its own output.
struct Child
    proc::Base.Process
    log::String
end

"""
    Oracle(dir; isolate = false, timeout = 600)

Runs candidate programs and remembers their classes; every candidate is written to `dir` as
`<n>.jl` with its output in `<n>.log`.  With `isolate`, candidates run in a child process kept
warm between them, and once a failure has killed a child a spare is always loading in the
background.  A candidate is killed after `timeout` seconds, a cut often leaving a loop with no
exit; its setup and primal, plain Julia, get three times the longest such run seen.
"""
mutable struct Oracle
    const dir::String
    const isolate::Bool
    timeout::Float64
    primal::Union{Nothing, Class}            # how the original's primal ends as a plain program; a candidate's may do that or pass
    const seen::Dict{Tuple{String, String}, Tuple{String, Class}}
    count::Int
    child::Union{Nothing, Child}
    spare::Union{Nothing, Child}
    children::Int
    primal_time::Float64                     # the longest a job took to get past its setup and primal
end
Oracle(dir; isolate = false, timeout = 600.0) = Oracle(dir, isolate, timeout, nothing, Dict{Tuple{String, String}, Tuple{String, Class}}(), 0, nothing, nothing, 0, 0.0)

function spawn_child(o::Oracle)
    project = Base.active_project()
    flags = ["--threads=$(Threads.nthreads())"; project === nothing ? String[] : ["--project=$project"]]
    log = joinpath(o.dir, "child_$(o.children += 1).log")
    proc = open(pipeline(`$(Base.julia_cmd()) $flags -e "using Enzyme; Enzyme.Shrink.serve()"`; stderr = log), "r+")
    return Child(proc, log)
end

function Base.close(o::Oracle)
    for c in (o.child, o.spare)
        c === nothing || close(c.proc)
    end
    o.child = o.spare = nothing
    return
end

"""
    run_job(o, base, job, args...)

Run `job` with `args` in the oracle's child: the job goes through `base.in`, the result comes
back through `base.out`.  A child that dies, or reports a fatal signal and then hangs, is
replaced and the job is a `:crash` classified from the logs; one still running after
`o.timeout` seconds, or an `:evaluate` that has not reached `base.primal` in its budget, is
killed and the job a `:timeout`.
"""
function run_job(o::Oracle, base::String, job::Symbol, args...)
    serialize(base * ".in", (job, args))
    if o.child === nothing || !process_running(o.child.proc)
        died = o.child !== nothing
        o.child = @something(o.spare, spawn_child(o))
        o.spare = died ? spawn_child(o) : nothing
        readline(o.child.proc)                                       # the child says when Enzyme is loaded
    end
    child = o.child
    println(child.proc, base)
    flush(child.proc)
    out, start = base * ".out", time()
    budget = job == :evaluate ? min(o.timeout, max(10.0, 3o.primal_time)) : o.timeout
    log = base * ".log"
    if await(child, log, start + budget, base * ".primal", out)
        job in (:evaluate, :record_run) && (o.primal_time = max(o.primal_time, time() - start))
        await(child, log, time() + o.timeout, out)
    end
    isfile(out) && return deserialize(out)
    crashed = !process_running(child.proc) || dying(log)
    process_running(child.proc) && (kill(child.proc, Base.SIGKILL); wait(child.proc))
    crashed && return crash_class(read(log, String) * read(child.log, String))
    return Class(:timeout, "", "still running after $(round(Int, time() - start))s")
end

# Whether the process writing `log` has reported a fatal signal; it may then hang in its own crash handler.
dying(log::String) = isfile(log) && occursin(r"signal \(\d+\)|signal \d+ \(", read(log, String))

function await(child::Child, log::String, deadline::Float64, files::String...)
    while !any(isfile, files) && process_running(child.proc) && time() < deadline
        dying(log) && (deadline = min(deadline, time() + 5))
        sleep(0.05)
    end
    return any(isfile, files)
end

perform_here(base::String, job::Symbol, args) = logged(() -> getfield(Shrink, job)(base, args...), base * ".log")

# Run `job` (`:evaluate`, `:record_run` or `:capture_here`) with `args`, output in `base.log`: here, or in the child when isolating.
perform(o::Oracle, base::String, job::Symbol, args...) = o.isolate ? run_job(o, base, job, args...) : perform_here(base, job, args)

function serialize_atomically(path::String, x)                  # calls may come from several threads
    tmp = "$path.$(Threads.threadid()).tmp"
    serialize(tmp, x)
    mv(tmp, path; force = true)
    return
end

# A child process: say when Enzyme is loaded, then for each base path on stdin run the job in `base.in`, result to `base.out`.
function serve()
    println("ready")
    flush(stdout)
    for base in eachline(stdin)
        job, args = deserialize(base * ".in")
        serialize_atomically(base * ".out", perform_here(base, job, args))
    end
    return
end

# Run `p` as `dir/<id>.jl` and classify it, as `(id, class)`; a program with source seen before is answered from memory.
function (o::Oracle)(p::Program)
    key = source(p)
    haskey(o.seen, key) && return o.seen[key]
    setup, call = key
    id = lpad(o.count += 1, 4, '0')
    base = joinpath(o.dir, id)
    write(base * ".jl", setup, call, "\n")
    primal = o.primal === nothing ? nothing : primal_line(p.call; copy = true)
    return o.seen[key] = (id, perform(o, base, :evaluate, setup, call, primal, o.primal))
end

const RECORDS = Dict{String, Any}()

# Record the first value seen for `key` (a loop records its first iteration), arrays copied so later mutation does not reach them; return `x`.
function record!(key::String, @nospecialize(x))
    if !haskey(RECORDS, key) && !(x isa AbstractArray && length(x) > 100_000)
        RECORDS[key] = x isa AbstractArray{<:Number} ? copy(x) : x
    end
    return x
end
EnzymeRules.inactive(::typeof(record!), args...) = nothing

function instrumented(p::Program)
    keys = IdDict{Any, String}()
    defs = map(enumerate(p.defs)) do (j, d)
        is_fdef(d) && fname(d) !== nothing || return d
        n = 0
        body = map_statements(d.args[2]) do s
            is_assignment(s) || return s
            key = keys[s] = "$(fname(d))@$j#$(n += 1)"
            records = [:(Enzyme.Shrink.record!($("$key.$y"), $y)) for y in assigned_names(s)]
            Expr(:block, s, records..., s.args[1])           # still evaluates to the assigned value
        end
        with_body(d, body)
    end
    return defs, keys
end

function record_run(base::String, setup::String, primal::String)
    empty!(RECORDS)
    m = Module(:Record)
    Base.include_string(m, setup, "record.jl")
    class = outcome(Base.include_string, m, primal, "record.jl")
    recorded = Dict{String, Value}()
    for (key, x) in RECORDS
        v = Base.invokelatest(value, x, m)
        v === nothing || (recorded[key] = v)
    end
    empty!(RECORDS)
    return class, recorded
end

"""
    record_values(p, o) -> (program, class)

Run the primal once with every assignment instrumented and return `p` with the recorded
values attached, and the class of that run: a candidate's primal must pass or end the same way.
"""
function record_values(p::Program, o::Oracle)
    defs, keys = instrumented(p)
    setup, _ = source(with(p; defs))
    primal = primal_line(p.call)
    base = joinpath(o.dir, "record")
    write(base * ".jl", setup, primal, "\n")
    result = perform(o, base, :record_run, setup, primal)
    class, recorded = result isa Class ? (result, Dict{String, Value}()) : result
    values = IdDict{Any, Vector{Value}}()
    for (s, key) in keys
        vals = [get(recorded, "$key.$y", nothing) for y in assigned_names(s)]
        any(isnothing, vals) || (values[s] = Value[vals...])
    end
    class == PASS || @warn "the primal itself fails before any differentiation; a candidate's primal may do the same: $class"
    return Program(p.defs, p.call, values), class
end
