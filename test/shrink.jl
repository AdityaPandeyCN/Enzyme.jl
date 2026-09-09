using Enzyme, Test
using Enzyme.Shrink: Shrink, CAPTURING, SESSION, PASS, classify, Value, Arg, Call, Program, Oracle,
    script_definitions, run_script, capture_failure, name_the_function, call_line, source, entry, stmts, fparams,
    fname, is_fdef, is_assignment, all_statements, remove_suffix, placeholder_statements, remove_unused_args,
    remove_defs, drop_activity, shrink_arrays, simplify_call, probes, check, WrongDerivative, Class
using Enzyme: EnzymeRules

cf(x, p) = sum(x) * p
cbad(x) = x[7]
cx = [1.0, 2.0]
cp = 3.0

twice(x) = 2x
EnzymeRules.forward(config, ::Const{typeof(twice)}, ::Type{<:DuplicatedNoNeed}, x::Duplicated) = 3x.dval   # wrong on purpose

@testset "Failure classes" begin
    c = classify(BoundsError([1.0], 3))
    @test c.kind == :BoundsError && c.key == "attempt to access N-element Vector{FloatN} at index [N]"
    @test c == classify(BoundsError([1.0, 2.0], 7))
    @test classify(ErrorException("a")) != classify(ErrorException("b"))
    @test classify(LoadError("f.jl", 1, BoundsError([1.0], 3))).kind == :BoundsError

    ta(mi, msg = "Illegal updateAnalysis prev:{[-1]:Pointer} new: {[-1]:Integer}\n val: %5 = load") =
        Enzyme.Compiler.IllegalTypeAnalysisException(msg, mi, mi === nothing ? nothing : UInt(1), "", nothing, nothing)
    c = classify(ta(Enzyme.Compiler.my_methodinstance(nothing, typeof(sin), Tuple{Float64})))
    @test c.kind == :IllegalTypeAnalysisException && c.key == "in sin"
    @test c == classify(ta(Enzyme.Compiler.my_methodinstance(nothing, typeof(sin), Tuple{Float32}), "another message"))
    @test c != classify(ta(Enzyme.Compiler.my_methodinstance(nothing, typeof(cos), Tuple{Float64})))
    @test classify(ta(nothing)).key == ""
    ie(msg) = Enzyme.Compiler.EnzymeInternalError(msg, nothing, nothing, nothing, nothing)
    @test classify(ie("Illegal replace ficticious phi for: %12 = phi\n<ir>")) == classify(ie("Illegal replace ficticious phi for: %13 = phi\n<other>"))
end

@testset "Describing a call" begin
    m = @__MODULE__
    dx = [5.0, 5.0]
    c = Shrink.describe(set_err_if_func_written(Reverse), Const(cf), Active, (Duplicated(cx, dx), Const(cp)), m)
    @test c.f == :cf && c.mode == Reverse && [a.name for a in c.args] == [:cx, :cp]
    @test c.args[1].shadows[1].data == [5.0, 5.0] && c.args[1].shadows[1].data !== dx
    @test call_line(c, "") == "autodiff(Reverse, cf, Active, Duplicated(cx, dcx), Const(cp))"
    @test call_line(c, "Enzyme.") == "Enzyme.autodiff(Enzyme.Reverse, cf, Enzyme.Active, Enzyme.Duplicated(cx, dcx), Enzyme.Const(cp))"
    c = Shrink.describe(set_runtime_activity(Forward), Const(cf), Duplicated, (BatchDuplicated([5.0], ([0.0], [1.0])), Const(cx)), m)
    @test call_line(c, "") == "autodiff(set_runtime_activity(Forward), cf, Duplicated, BatchDuplicated(a1, (da1_1, da1_2,)), Const(cx))"
    @test Shrink.describe(Reverse, Duplicated(cf, cf), Active, (Const(cp),), m) isa String

    SESSION.script = m
    CAPTURING[] = true
    try
        @test autodiff(Reverse, cf, Active, Duplicated(cx, zero(cx)), Const(cp)) === ((nothing, nothing),)
        @test call_line(SESSION.call, "") == "autodiff(Reverse, cf, Active, Duplicated(cx, dcx), Const(cp))"
        @test_throws BoundsError Enzyme.gradient(Reverse, cbad, cx)
        @test call_line(SESSION.call, "") == "autodiff(Reverse, cbad, Active, Duplicated(cx, dcx))"
        @test_throws BoundsError Enzyme.hvp(cbad, cx, [1.0, 0.0])
        @test SESSION.call.f == :gradient! && SESSION.call.args[3].val.text == "cbad"
        SESSION.script = Enzyme.Shrink
        @test_throws BoundsError autodiff(Reverse, cbad, Active, Duplicated(cx, zero(cx)))
        @test SESSION.call isa String && occursin("cannot name the differentiated function", SESSION.call)
    finally
        CAPTURING[] = false
        SESSION.script = Main
        SESSION.call = nothing
    end
end

@testset "Running a script" begin
    dir = mktempdir()
    script = joinpath(dir, "s.jl")
    write(script, "using Enzyme\ny = [1.0, 2.0]\nautodiff(Reverse, x -> x[3] * y[1], Active, Duplicated(y, zero(y)))\n")
    index, call, err = run_script(script)
    @test index == 3 && classify(err).kind == :BoundsError && [a.name for a in call.args] == [:y]
    @test call.f == Base.remove_linenums!(:(x -> x[3] * y[1]))
    p = name_the_function(Program(script_definitions(script)[1:2], call, IdDict{Any, Vector{Value}}()))
    @test call_line(p.call, "") == "autodiff(Reverse, entry, Active, Duplicated(y, dy))" && fparams(p.defs[end]) == [:x]
    @test !CAPTURING[] && SESSION.script === Main
    write(script, "using Enzyme\nautodiff(Reverse, sum, Active, Duplicated([1.0], [0.0]))\n")
    @test run_script(script) === nothing
    write(script, "using Enzyme\nerror(\"before any call\")\n")
    @test run_script(script)[2] === nothing
end

@testset "Passes" begin
    script = joinpath(mktempdir(), "s.jl")
    write(script, "using Enzyme\ng(x) = x .* 2\nfunction f(x, p)\n    a = g(x)\n    b = sum(a) * p\n    c = b + 1\n    return c\nend\n")
    x = [1.0, 2.0, 3.0, 4.0]
    args = [Arg(:Duplicated, :x, Value(x), [Value(zero(x))]), Arg(:Const, :p, Value(3.0), Value[])]
    p = Program(script_definitions(script), Call(ReverseWithPrimal, :f, Active, args), IdDict{Any, Vector{Value}}())
    for (i, s) in enumerate(stmts(p.defs[entry(p)]))
        is_assignment(s) && (p.values[s] = [Value(i == 1 ? x .* 2 : Float64(i))])
    end
    body(q) = stmts(q.defs[entry(q)])

    s = remove_suffix(p)
    @test body(s[1].second) == [:(a = g(x)), :(b = sum(a) * p), :(return b)]
    @test body(s[2].second) == [:(a = g(x)), :(return sum(abs2, a))]
    q = placeholder_statements(p)[1].second
    @test body(q)[1:2] == [:(a = a_rec), :(b = b_rec)] && fparams(q.defs[entry(q)]) == [:x, :p, :a_rec, :b_rec]
    @test [a.kind for a in q.call.args] == [:Duplicated, :Const, :Duplicated, :Active]
    u = remove_unused_args(q)[1].second
    @test fparams(u.defs[entry(u)]) == [:a_rec, :b_rec] && [a.name for a in u.call.args] == [:a_rec, :b_rec]
    @test remove_defs(p)[1].second.defs == [:(using Enzyme), p.defs[3]]
    @test [q.call.mode for (_, q) in simplify_call(p)] == [Reverse, ReverseWithPrimal]
    @test simplify_call(p)[end].second.call.ret === Const
    @test drop_activity(p)[1].second.call.args[1].kind == :Const
    a = shrink_arrays(p)[1].second.call.args[1]
    @test a.val.data == [1.0] && a.shadows[1].data == [0.0]
    setup, call = source(u)
    @test occursin("a_rec = [2.0, 4.0, 6.0, 8.0]\nda_rec = zero(a_rec)\nb_rec = 2.0\n", setup)
    @test call == "autodiff(ReverseWithPrimal, f, Active, Duplicated(a_rec, da_rec), Active(b_rec))"
    @test first.(probes(p)) == ["set_runtime_activity", "set_strong_zero", "InlineABI"]
    @test call_line(last(probes(p))[2].call, "") == "autodiff(set_abi(ReverseWithPrimal, InlineABI), f, Active, Duplicated(x, dx), Const(p))"
end

@testset "Wrong derivatives" begin
    @test check(Forward, sin, Duplicated, Duplicated(1.0, 1.0)) == (cos(1.0),)
    @test check(Reverse, cf, Active, Duplicated(cx, zero(cx)), Active(cp)) == ((nothing, 3.0),)
    err = (@test_throws WrongDerivative check(Forward, twice, Duplicated, Duplicated(1.0, 1.0))).value
    @test err isa WrongDerivative && classify(err) == Class(:WrongDerivative, "tangent of the return value", "")
    dir = mktempdir()
    script = joinpath(dir, "wrong.jl")
    write(
        script, "using Enzyme\nusing Enzyme: EnzymeRules\ntwice(x) = 2x\n" *
            "EnzymeRules.forward(config, ::Const{typeof(twice)}, ::Type{<:DuplicatedNoNeed}, x::Duplicated) = 3x.dval\n" *
            "autodiff(Forward, twice, Duplicated, Duplicated(1.0, 1.0))\n"
    )
    index, call, err = run_script(script)
    @test index == 5 && call.f == :twice && err isa WrongDerivative
    @test startswith(call_line(Shrink.with(call; checked = true), ""), "Enzyme.Shrink.check(Forward, twice, Duplicated, Duplicated(a1, da1))")
end

@testset "Isolation" begin
    dir = mktempdir()
    call = Call(Reverse, :f, Active, [Arg(:Active, :x, Value(1.0), Value[])])
    program(defs...) = Program(Any[:(using Enzyme), :(f(x) = x), defs...], call, IdDict{Any, Vector{Value}}())
    o = Oracle(dir; isolate = true, timeout = 2)
    try
        @test o(program(:(ccall(:abort, Cvoid, ()))))[2].kind == :crash
        @test o(program(:(sleep(60))))[2].kind == :timeout
        script = joinpath(dir, "abort.jl")
        write(script, "using Enzyme\nboom(x) = (Enzyme.within_autodiff() && ccall(:abort, Cvoid, ()); x[1])\nx = [1.0]\nautodiff(Reverse, boom, Active, Duplicated(x, zero(x)))\n")
        o.timeout = 600.0
        index, call, class = capture_failure(script, o)
        @test index == 4 && call.f == :boom && class.kind == :crash
    finally
        close(o)
    end
end

const E2E = """
using Enzyme
scale(a) = a .* 2
function pick(x, y, w)
    a = scale(x)
    b = sum(a)
    c = w .* w
    z = x[1] > 0 ? x : y
    s = sum(z) * w[1]
    return s + b + c[1]
end
x = [1.0, 2.0, 3.0, 4.0]; dx = zero(x)
y = [4.0, 3.0, 2.0, 1.0]
w = [1.0 2.0; 3.0 4.0]; dw = zero(w)
unused = rand(10)
autodiff(ReverseWithPrimal, pick, Active, Duplicated(x, dx), Const(y), Duplicated(w, dw))
"""

@testset "End to end" begin
    dir = mktempdir()
    script = joinpath(dir, "script.jl")
    write(script, E2E)
    repro = Enzyme.shrink(script)
    @test repro isa String && isfile(repro) && isfile(joinpath(dirname(repro), "original.jl"))
    d = only(filter(d -> is_fdef(d) && fname(d) == :pick, script_definitions(repro)))
    @test length(all_statements(d.args[2])) <= 2 && length(fparams(d)) <= 2
    @test occursin("pick", read(joinpath(dirname(repro), "typed_ir.txt"), String))
    @test occursin("# with set_runtime_activity: passes", read(repro, String))
    err = (@test_throws LoadError Base.include(Module(), repro)).value
    @test err isa LoadError && err.error isa Enzyme.Compiler.EnzymeRuntimeActivityError
    write(joinpath(dir, "ok.jl"), "using Enzyme\nautodiff(Reverse, sum, Active, Duplicated([1.0], [0.0]))\n")
    @test Enzyme.shrink(joinpath(dir, "ok.jl")) === nothing
end
