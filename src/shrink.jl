"""
    Enzyme.Shrink

Reduce a failing `autodiff` call to a minimal reproducer; see [`Enzyme.shrink`](@ref).

The script's definitions and the captured call form a [`Program`](@ref). The primal is run
once to record every intermediate value, then each pass proposes smaller programs and the
[`Oracle`](@ref) keeps one when the call still fails with the original [`Class`](@ref) and
the primal still runs.
"""
module Shrink

import InteractiveUtils
using Serialization
using ..Enzyme
using ..Enzyme: Compiler, EnzymeCore

include("shrink/program.jl")
include("shrink/check.jl")
include("shrink/oracle.jl")
include("shrink/capture.jl")
include("shrink/passes.jl")

function Enzyme.shrink(file::AbstractString; isolate::Bool = false, timeout::Union{Nothing, Real} = nothing, workers::Int = 1)
    file = abspath(file)
    isfile(file) || throw(ArgumentError("no such file: $file"))
    dir = mkpath(joinpath(dirname(file), "shrink_" * Libc.strftime("%Y%m%d-%H%M%S", time())))
    mkpath(joinpath(dir, "checkpoints"))
    o = Oracle(dir; isolate, timeout = something(timeout, 600), workers)
    try
        return reduce_script(file, dir, o, timeout)
    finally
        close(o)
    end
end

function reduce_script(file::String, dir::String, o::Oracle, timeout)
    captured = capture_failure(file, o)
    if captured === nothing
        println("no autodiff call failed; nothing to reduce")
        return nothing
    end
    index, call, class = captured
    call === nothing && error("the script failed outside any autodiff call, at top-level expression $index: $class")
    call isa Call || error("the failing call at top-level expression $index cannot be reduced: $call\n  $class")
    original = call_line(call, "")
    class.kind == :WrongDerivative && (call = with(call; checked = true))
    println("captured at top-level expression $index: ", one_line(original), "\n  ", class)

    p = name_the_function(Program(script_definitions(file)[1:(index - 1)], call, IdDict{Any, Vector{Value}}()))
    elapsed = @elapsed (_, target) = o(p)
    target == PASS && error("the captured call does not fail when run in isolation; the failure depends on something outside the script's top-level definitions")
    (target.kind == :setup || target.kind != class.kind) && error("the call fails differently in isolation: $target")
    timeout === nothing && (o.timeout = target.kind == :timeout ? 60.0 : max(60.0, 5elapsed))
    println("target: ", target, "\n  candidates get $(round(Int, o.timeout)) s")
    write_repro(joinpath(dir, "original.jl"), p, target, original)
    p, o.primal = record_values(p, o)
    o.primal == target && error("the primal itself fails this way before any differentiation; nothing for Enzyme to explain: $target")
    p, known = minimize(p, o, target, original)
    variants = probes(p)
    verdicts = o([q for (_, q) in variants])
    notes = [["with $label: $(verdict(class, target))" for ((label, _), (_, class)) in zip(variants, verdicts)]; ingredients(p, o, target, known)]
    foreach(println, notes)

    repro = joinpath(dir, "repro.jl")
    write_repro(repro, p, target, original; notes)
    println("\nrepro written to ", repro)
    return repro
end

end
