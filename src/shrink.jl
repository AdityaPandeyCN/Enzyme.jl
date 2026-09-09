"""
    Enzyme.Shrink

Reduce a failing `autodiff` call to a minimal reproducer; the machinery behind [`Enzyme.shrink`](@ref).

    script.jl ── run with capture on ──▶ Call                shrink/capture.jl
    Call ── autodiff, finite differences ▶ WrongDerivative    shrink/check.jl
    Call + the script's definitions ──▶ Program             shrink/program.jl
    Program ── run the primal once ───▶ Program with values  shrink/oracle.jl
    Program ── the oracle ────────────▶ Class                shrink/oracle.jl
    Program ── a pass ────────────────▶ smaller Programs     shrink/passes.jl
    smallest Program ─────────────────▶ repro.jl             shrink/program.jl

This is the FX minifier's approach (truncate the suffix, then delta-debug with intermediates
turned into inputs) applied to Julia source.  FX works on a graph where every intermediate is
a node whose value is known; Julia source is not, so the primal is run once with every
assignment instrumented to record its value, and statements play the part of nodes.

A smaller program is kept when it is the same failure in a valid program: the differentiated
call ends in the original's [`Class`](@ref), and the plain run of the program either passes or
ends as the original's plain run did, so that a cut which merely breaks the program is never
mistaken for one that exposes the bug.
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

function Enzyme.shrink(file::AbstractString; isolate::Bool = false, timeout::Union{Nothing, Real} = nothing)
    file = abspath(file)
    isfile(file) || throw(ArgumentError("no such file: $file"))
    dir = mkpath(joinpath(dirname(file), "shrink_" * Libc.strftime("%Y%m%d-%H%M%S", time())))
    mkpath(joinpath(dir, "checkpoints"))
    o = Oracle(dir; isolate, timeout = something(timeout, 600))
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
    call isa Call || error("the failing call at top-level expression $index cannot be reduced: $call")
    original = call_line(call, "")
    class.kind == :WrongDerivative && (call = with(call; checked = true))
    println("captured at top-level expression $index: ", one_line(original), "\n  ", class)

    p = name_the_function(Program(script_definitions(file)[1:(index - 1)], call, IdDict{Any, Vector{Value}}()))
    elapsed = @elapsed (_, target) = o(p)
    target == PASS && error("the captured call does not fail when run in isolation; the failure depends on something outside the script's top-level definitions")
    (target.kind == :setup || target.kind != class.kind) && error("the call fails differently in isolation: $target")
    println("target: ", target)
    timeout === nothing && (o.timeout = target.kind == :timeout ? 60.0 : max(60.0, 5elapsed))
    write_repro(joinpath(dir, "original.jl"), p, target, original)
    p, o.primal = record_values(p, o)
    o.primal == target && error("the primal itself fails this way before any differentiation; nothing for Enzyme to explain: $target")
    p, known = minimize(p, o, target, original)
    notes = [["with $label: $(verdict(o(q)[2], target))" for (label, q) in probes(p)]; ingredients(p, o, target, known)]
    foreach(println, notes)

    repro = joinpath(dir, "repro.jl")
    write_repro(repro, p, target, original; notes)
    println("\nrepro written to ", repro)
    return repro
end

end
