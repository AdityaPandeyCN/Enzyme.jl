# Every kind of failure the reducer handles, one small script each, run to completion through
# `Enzyme.shrink` in a subprocess.  Slow (an hour); runs only with SHRINK_BASELINE=1.

using Enzyme, Test

const CASES = [
    (
        "runtime_activity", "EnzymeRuntimeActivityError", false, 600, raw"""
        using Enzyme
        scale(a) = a .* 2
        function f(x, y, w)
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
        autodiff(ReverseWithPrimal, f, Active, Duplicated(x, dx), Const(y), Duplicated(w, dw))
        """,
    ),
    (
        "mutability", "EnzymeMutabilityException", false, 600, raw"""
        using Enzyme
        struct RWClos
            x::Vector{Float64}
        end
        function (c::RWClos)(y)
            c.x[1] *= y
            return y
        end
        c = RWClos([4.0])
        autodiff(Reverse, c, Active(3.0))
        """,
    ),
    (
        "nonscalar_return", "EnzymeNonScalarReturnException", false, 600, raw"""
        using Enzyme
        array_square(x) = 2 .* x
        function g(x)
            a = array_square(x)
            b = a .+ 1
            return b
        end
        Enzyme.gradient(Reverse, g, [2.0, 3.0])
        """,
    ),
    (
        "error_exception_complex", "ErrorException", false, 600, raw"""
        using Enzyme
        mul2(z) = 2 * z
        function f(z, p)
            a = mul2(z)
            b = a * p
            return b
        end
        autodiff(Reverse, f, Active, Active(1.0 + 1.0im), Const(2.0))
        """,
    ),
    (
        "wrong_reverse_rule", "WrongDerivative", false, 900, raw"""
        using Enzyme
        using Enzyme: EnzymeRules
        sq(x) = x * x
        function EnzymeRules.augmented_primal(config, ::Const{typeof(sq)}, ::Type{<:Active}, x::Active)
            return EnzymeRules.AugmentedReturn(EnzymeRules.needs_primal(config) ? x.val * x.val : nothing, nothing, nothing)
        end
        EnzymeRules.reverse(config, ::Const{typeof(sq)}, dret::Active, tape, x::Active) = (3 * x.val * dret.val,)
        function f(x, y)
            a = sq(x[1])
            b = a + sum(y)
            return b
        end
        x = [2.0, 1.0]; dx = zero(x)
        y = [1.0, 2.0]; dy = zero(y)
        autodiff(Reverse, f, Active, Duplicated(x, dx), Duplicated(y, dy))
        """,
    ),
    (
        "rule_forward_return_error", "ForwardRuleReturnError", false, 600, raw"""
        using Enzyme
        using Enzyme: EnzymeRules
        function f_kw(out)
            out[1] *= 2
            nothing
        end
        EnzymeRules.forward(config, ::Const{typeof(f_kw)}, ::Type{<:Const}, x::Duplicated) = (f_kw(x.val); 2)
        x = [2.7]; dx = [3.1]
        autodiff(Forward, f_kw, Duplicated(x, dx))
        """,
    ),
    (
        "rule_reverse_return_error", "ReverseRuleReturnError", false, 600, raw"""
        using Enzyme
        using Enzyme: EnzymeRules
        function g_kw(out)
            out[1] *= 2
            nothing
        end
        EnzymeRules.augmented_primal(config, ::Const{typeof(g_kw)}, ::Type{<:Const}, x::Duplicated) = (g_kw(x.val); EnzymeRules.AugmentedReturn(nothing, nothing, nothing))
        EnzymeRules.reverse(config, ::Const{typeof(g_kw)}, ::Type{<:Const}, tape, x::Duplicated) = (g_kw(x.dval); ())
        x = [2.7]; dx = [3.1]
        autodiff(Reverse, g_kw, Duplicated(x, dx))
        """,
    ),
    (
        "kw_nonconst", "NonConstantKeywordArgException", false, 600, raw"""
        using Enzyme
        using Enzyme: EnzymeRules
        f_kw4(x; y = 2.0) = x * y
        EnzymeRules.forward(config, ::Const{typeof(f_kw4)}, ::Type{<:DuplicatedNoNeed}, x::Duplicated; y) = 1000 * y + 2 * x.val * x.dval
        g4(x, y) = f_kw4(x; y)
        autodiff(Forward, g4, Duplicated(2.0, 1.0), Duplicated(42.0, 1.0))
        """,
    ),
    (
        "closure_local", "BoundsError", false, 600, raw"""
        using Enzyme
        function run(k)
            x = [1.0, 2.0]
            autodiff(Reverse, v -> v[k] * 2, Active, Duplicated(x, zero(x)))
        end
        run(5)
        """,
    ),
    (
        "mutation_arg", "EnzymeRuntimeActivityError", false, 600, raw"""
        using Enzyme
        function f(x, y)
            x[1] = x[1] * 2
            z = x[1] > 0 ? x : y
            x[2] += 1
            return sum(z)
        end
        x = [1.0, 2.0]; dx = zero(x)
        y = [3.0, 4.0]
        autodiff(Reverse, f, Active, Duplicated(x, dx), Const(y))
        """,
    ),
    (
        "illegal_type_analysis", "IllegalTypeAnalysisException", false, 600, raw"""
        using Enzyme
        bump(v) = reinterpret(Float64, reinterpret(UInt64, v) .+ 1)
        function f(x, p)
            v = [x, 2x]
            w = bump(v)
            s = w[1] + w[2]
            return s * p
        end
        autodiff(Reverse, f, Active, Active(1.5), Const(2.0))
        """,
    ),
    (
        "no_derivative_runnable", "EnzymeNoDerivativeError", false, 600, raw"""
        using Enzyme
        function f(x)
            v = [x]
            p = Ptr{Int64}(pointer(v))
            i = unsafe_load(p)
            unsafe_store!(p, i + 1)
            return v[1]
        end
        autodiff(Reverse, f, Active, Active(1.0))
        """,
    ),
    (
        "wrong_forward_rule_fixed", "WrongDerivative", false, 900, raw"""
        using Enzyme
        using Enzyme: EnzymeRules
        twice(x) = 2x
        EnzymeRules.forward(config, ::Const{typeof(twice)}, ::Type{<:Duplicated}, x::Duplicated) = Duplicated(2 * x.val, 3 * x.dval)
        EnzymeRules.forward(config, ::Const{typeof(twice)}, ::Type{<:DuplicatedNoNeed}, x::Duplicated) = 3 * x.dval
        function f(x, p)
            a = twice(x)
            b = a + p
            c = sin(b)
            return c
        end
        autodiff(Forward, f, Duplicated, Duplicated(1.0, 1.0), Const(0.5))
        """,
    ),
    (
        "crash_abort", "crash", true, 1200, raw"""
        using Enzyme
        function boom(x, p)
            a = x .* p
            b = sum(a)
            Enzyme.within_autodiff() && ccall(:abort, Cvoid, ())
            return b * 2
        end
        x = [1.0, 2.0]; dx = zero(x)
        autodiff(Reverse, boom, Active, Duplicated(x, dx), Active(3.0))
        """,
    ),
    (
        "hang_isolate_t30", "timeout", true, 1500, raw"""
        using Enzyme
        function spin(x, p)
            a = x .* p
            b = sum(a)
            while Enzyme.within_autodiff()
                x[1] = x[1] + 0.0
            end
            return b * 2
        end
        x = [1.0, 2.0]; dx = zero(x)
        autodiff(Reverse, spin, Active, Duplicated(x, dx), Active(3.0))
        """,
    ),
    (
        "no_failure", "no autodiff call failed", false, 300, raw"""
        using Enzyme
        f(x) = sum(x .* 2)
        x = [1.0, 2.0]
        autodiff(Reverse, f, Active, Duplicated(x, zero(x)))
        autodiff(Forward, f, Duplicated, Duplicated(x, ones(2)))
        """,
    ),
    (
        "error_outside", "failed outside any autodiff call", false, 300, raw"""
        using Enzyme
        f(x) = x[1] * 2
        x = [1.0]
        error("boom before any call")
        autodiff(Reverse, f, Active, Duplicated(x, zero(x)))
        """,
    ),
]

function reduce_case(src, isolate, cap)
    dir = mktempdir()
    script = joinpath(dir, "script.jl")
    write(script, src)
    log = joinpath(dir, "run.log")
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) -e "using Enzyme; Enzyme.shrink(\"$script\"; isolate = $isolate, timeout = $(isolate ? 30 : nothing))"`
    proc = run(pipeline(cmd; stdout = log, stderr = log); wait = false)
    deadline = time() + cap
    while process_running(proc) && time() < deadline
        sleep(1)
    end
    process_running(proc) && kill(proc)
    text = read(log, String)
    m = match(r"captured at top-level expression \d+: .*\n  (\w+)", text)
    m === nothing || return String(m[1]), isfile(joinpath(dir, only(filter(startswith("shrink_"), readdir(dir))), "repro.jl"))
    m = match(r"ERROR: (.*)|(no autodiff call failed)", text)
    return m === nothing ? "no result after $(cap)s" : String(something(m[1], m[2])), false
end

if get(ENV, "SHRINK_BASELINE", "") == "1"
    @testset "$name" for (name, expected, isolate, cap, src) in CASES
        t = @elapsed (got, repro) = reduce_case(src, isolate, cap)
        println(rpad(name, 28), rpad(got, 40), repro ? "repro " : "       ", round(Int, t), "s")
        @test occursin(expected, got)
    end
end
