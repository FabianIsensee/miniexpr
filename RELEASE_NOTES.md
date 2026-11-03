Release notes for miniexpr
==========================

Changes from 0.2.0 to 0.2.1
===========================

* Mixed signed/unsigned integers promote as in NumPy: int16 + uint8 is int16
  and int64 + uint32 is int64 (they were widened one step further, and int64
  with a narrower unsigned type went to float64).

* Expression operator precedence follows Python. Behaviour change: `-x ** 2` is
  now `-(x ** 2)` (was `(-x) ** 2`), `~x ** 2` is `~(x ** 2)`, `a ** b ** c` is
  `a ** (b ** c)` (was `(a ** b) ** c`), and in plain expressions `a < b < c` is
  `(a < b) and (b < c)` (was `(a < b) < c`; DSL kernels already chained). The
  `ME_POW_FROM_RIGHT` source option is removed. Write `(-x) ** 2` to keep the old
  meaning. Function arguments accept `not`/`and`/`or` (`where(a > 0 and b > 0, x, y)`).

* DSL JIT kernels group `not` and `&`/`^`/`|` around comparisons as Python and
  the interpreter do. Generated C used C's precedence, so `x & 1 == 0` computed
  `x & (1 == 0)` and `not a == b` computed `(not a) == b`. Comparison chains
  accept `~` operands (`0 < ~x < 5` was rejected).

* Behaviour change: number literals in expressions are NumPy's weak scalars
  (NEP 50). They take the dtype of the array they meet unless their kind is
  higher, independently of the output dtype: `int32 < 2.5` compares in float64
  (was: with 2), `int8 < 300` is true, `float32 > 0.1` compares with
  float32(0.1), and under `ME_AUTO` `int32 + 0.5` is float64 (was int32) and
  `bool + 1` is int64. An integer literal outside the integer dtype it meets
  (`uint8 + 300`, rejected by NumPy 2) widens the computation (uint16) instead
  of wrapping.

* Behaviour change: true division and `atan2`, `copysign`, `hypot`,
  `logaddexp`, `nextafter` of integer or bool operands compute in float64, as
  in NumPy (int / int was computed in integer arithmetic, and bool / bool could
  raise SIGFPE).

* Behaviour change: in expressions, bool + bool is logical or and bool * bool
  is logical and, as for NumPy bool arrays (they were added as int8, so
  `(p + q) * 1.5` gave 3.0). DSL kernels keep Python's numeric bools.

* Expression trees are retyped after parsing and optimization, so mixed-dtype
  expressions no longer convert values between buffers of the wrong dtype
  (wrong results and heap overflows such as int16 `(i + i) * abs(i)`), and a
  non-bool `where()` condition selects by its truth.

* Behavior change: the `%` operator and `remainder()` now follow NumPy
  (`np.remainder`, floor modulo) in expressions and full-profile DSL kernels,
  for every engine: the result takes the sign of the divisor
  (`-7 % 3 == 2`, `7 % -3 == -2`, `-7.0 % 3.0 == 2.0`). Previously `%` was C
  `fmod` and `remainder()` was C99 `remainder()`. Integer operands are computed
  exactly in their own width; `x % 0` is 0 and `MIN % -1` is 0, with no trap.
  Floating `x % 0` is NaN and a zero result takes the divisor's sign. `fmod()`
  is unchanged (C `fmod`, as `np.fmod`). Menudet kernels already used these rules.

* Menudet JIT emits same-width typed integer bitwise operations directly instead
  of calling scalar interpreter helpers. Scalar integer/Boolean local declarations no
  longer use volatile rounding barriers, reducing stack traffic in hash kernels.
  Floating-point local barriers and checked-operation error semantics are unchanged.

* Portable interpretation batches complex64/128 addition, subtraction,
  multiplication, negation and conversions to complex over bounded vectors.
  Uniform captures are supported; complex division, other unsupported operations
  and partially masked expressions retain scalar evaluation. Scalar numerical
  semantics and aggregate IEEE status are preserved; no complex JIT is added.
  Input-free complex kernels with an immutable final-index return override
  handle the endpoint once per unmasked block, avoiding per-element condition
  interpretation while preserving skipped-lane errors and IEEE status.

* Windows (clang-cl) now supports complex64 and complex128 expressions and
  portable interpretation, using ABI-safe UCRT complex math wrappers.
  Complex DSL JIT kernels remain unsupported on all platforms.
  - API change: `ME_COMPILE_ERR_COMPLEX_UNSUPPORTED` has been removed; its
    numeric value (-4) remains unused. Consumers referencing the enum name
    must remove that reference when updating miniexpr.

* Flexible math libraries and floating-point modes for Menudet kernels.
  - New source pragmas `me:math=auto|system|sleef|accelerate` and
    `me:ulp=u10|u35`; `me:fp=strict|contract|fast` now reaches portable kernels.
  - Resolution precedence: source pragma, artifact declaration,
    `ME_SIMD_MATH_BACKEND`/`ME_DSL_FP_MODE`, built-in default.
  - `auto` no longer prefers SLEEF; it follows the system-math policy.
  - Explicit unavailable SLEEF/Accelerate requests fail during preparation.
  - Accelerated math uses a kernel-local immutable callback table and bounded
    vector interpretation; artifacts declare the `flexible-math` capability.
* New varlen output for string expressions: `me_eval_varlen()` writes the Arrow
  layout (int64 offsets + a tight byte blob) instead of fixed-width slots.
  - `ME_STRING` yields UTF-8 (`large_string`); `ME_BYTES` copies bytes verbatim
    (`large_binary`). DSL kernels work unchanged.
  - Size the data buffer with the new `me_varlen_data_bound()`.
  - The conservative compile-time width bound is then spent on scratch only and
    never on the result: 9.7x smaller output for a 12% pack overhead on
    `bench/benchmark_varlen_output.c`. See `doc/strings.md`.

* macOS SIMD math now prefers SLEEF by default when it is available at build time.
  - New CMake option: `MINIEXPR_USE_ACCELERATE=ON|OFF` (enabled by default on macOS, off elsewhere).
  - In runtime `auto` mode, macOS prefers SLEEF first, then falls back to Accelerate when enabled, then to scalar kernels.
  - Added `ME_SIMD_MATH_BACKEND=auto|sleef|accelerate|scalar` for backend forcing during debugging and benchmarking.
  - Updated SIMD math benchmarks to report backend-appropriate columns instead of implying fake `U10/U35` distinctions for Accelerate/scalar runs.

Changes from 0.1.0 to 0.2.0
===========================

* New DSL kernel support (major feature).
  - miniexpr now supports Python-like multi-statement kernels with assignments,
    temporaries, `if/elif/else`, `for`/`while`, `break`/`continue`, `print`,
    reductions, and ND index symbols.
  - For DSL language details, see:
    - `doc/dsl-syntax.md` (canonical syntax/behavior reference)
    - `doc/dsl-usage.md` (practical usage guide and examples)

* Runtime JIT for DSL kernels has been added and significantly expanded.
  - New DSL parser + IR + C code generation pipeline.
  - Runtime JIT compiler policy controls (`ME_JIT_DEFAULT`, `ME_JIT_ON`, `ME_JIT_OFF`),
    available both at compile-time (`me_compile_nd_jit`) and eval-time
    (`me_eval_params.jit_mode`).
  - JIT backend/compiler controls via DSL pragmas (`# me:fp=...`, `# me:compiler=...`)
    and environment/toolchain integration (`CC`, `CFLAGS`).
  - Improved runtime behavior for Linux/macOS/Windows/wasm32 paths, including cache
    handling and wasm helper registration APIs.

* New public diagnostics helper:
  - `me_get_last_error_message()` to retrieve thread-local human-readable failure details.

* Extended string support:
  - `ME_STRING` data type and string operations/predicates are now documented and covered.
  - See `doc/strings.md`.

* Quality, compatibility, and correctness improvements.
  - Multiple fixes in dtype inference/casting and interpreter/JIT parity.
  - ND/padded block and index-variable behavior improvements.
  - Broader test/benchmark coverage (DSL syntax, JIT IR/codegen/runtime cache, side-module
    glue smoke tests, dtype mismatch cases, string ops, and more).
  - CI/build updates and platform robustness improvements (especially Windows and wasm32).


Changes from 0.0.0 to 0.1.0
===========================

* Initial public beta release of miniexpr.
* Vectorized expression evaluation across multiple data types.
* Thread-safe evaluation and chunked/ND processing helpers.
* Optional SLEEF-accelerated SIMD math kernels (CMake toggle).
* Examples, tests, and benchmarks covering core usage patterns.
