/*
 * Mixed-dtype expressions whose operands need conversion nodes.
 *
 * The evaluator computes each subtree in one dtype and converts at conversion
 * nodes.  These cases used to get a conversion built from a provisional
 * parse-time dtype (calls start as float64, comparisons as bool) and wrote a
 * value of one dtype into a buffer of another: garbage values, and heap
 * overflows for 1-, 2- and 4-byte buffers.  Expected values follow NumPy.
 */

#include "../src/miniexpr.h"
#include "minctest.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Not a multiple of any SIMD width, and large enough for an overflowing
 * conversion to corrupt the heap. */
#define N 3001

static int failures = 0;

typedef double (*ref_fn)(const double *v);

typedef struct {
    const char *expr;
    int nvars;
    const char *names[3];
    me_dtype dtypes[3];
    me_dtype out;
    ref_fn ref;
    /* Inferred dtype depends on literal typing (bool ** 2 is int8 in NumPy). */
    bool explicit_only;
} conv_case;

static size_t dtype_bytes(me_dtype dt) {
    switch (dt) {
    case ME_BOOL: case ME_INT8: case ME_UINT8: return 1;
    case ME_INT16: case ME_UINT16: return 2;
    case ME_INT32: case ME_UINT32: case ME_FLOAT32: return 4;
    default: return 8;
    }
}

static void store(void *buf, me_dtype dt, int i, double v) {
    switch (dt) {
    case ME_BOOL: ((bool *)buf)[i] = v != 0; break;
    case ME_INT8: ((int8_t *)buf)[i] = (int8_t)v; break;
    case ME_INT16: ((int16_t *)buf)[i] = (int16_t)v; break;
    case ME_INT32: ((int32_t *)buf)[i] = (int32_t)v; break;
    case ME_INT64: ((int64_t *)buf)[i] = (int64_t)v; break;
    case ME_UINT8: ((uint8_t *)buf)[i] = (uint8_t)v; break;
    case ME_UINT16: ((uint16_t *)buf)[i] = (uint16_t)v; break;
    case ME_UINT32: ((uint32_t *)buf)[i] = (uint32_t)v; break;
    case ME_UINT64: ((uint64_t *)buf)[i] = (uint64_t)v; break;
    case ME_FLOAT32: ((float *)buf)[i] = (float)v; break;
    default: ((double *)buf)[i] = v; break;
    }
}

static double load(const void *buf, me_dtype dt, int i) {
    switch (dt) {
    case ME_BOOL: return ((const bool *)buf)[i];
    case ME_INT8: return ((const int8_t *)buf)[i];
    case ME_INT16: return ((const int16_t *)buf)[i];
    case ME_INT32: return ((const int32_t *)buf)[i];
    case ME_INT64: return (double)((const int64_t *)buf)[i];
    case ME_UINT8: return ((const uint8_t *)buf)[i];
    case ME_UINT16: return ((const uint16_t *)buf)[i];
    case ME_UINT32: return ((const uint32_t *)buf)[i];
    case ME_UINT64: return (double)((const uint64_t *)buf)[i];
    case ME_FLOAT32: return ((const float *)buf)[i];
    default: return ((const double *)buf)[i];
    }
}

/* Small values (no overflow in any dtype used below), negative for signed
 * and floating operands, alternating for bools. */
static double input_value(me_dtype dt, int var, int i) {
    int k = (i + 3 * var) % 9;
    switch (dt) {
    case ME_BOOL: return (i + var) % 3 != 0;
    case ME_UINT8: case ME_UINT16: case ME_UINT32: case ME_UINT64: return k;
    case ME_FLOAT32: case ME_FLOAT64: return 0.75 * k - 3.25;
    default: return k - 3;
    }
}

static double r_x2_abs(const double *v) { return (v[0] + v[0]) * fabs(v[0]); }
static double r_f2_gt(const double *v) { return (double)(float)(v[0] + v[0]) > v[1]; }
static double r_lt_add(const double *v) { return (v[0] < v[1]) + v[1]; }
static double r_lt_mul(const double *v) { return (v[0] < v[1]) * v[1]; }
static double r_u32_neg(const double *v) {
    uint32_t b = (uint32_t)v[0];
    return (double)(uint32_t)((uint32_t)(b * 3u) * (uint32_t)(0u - b));
}
static double r_f32_mix(const double *v) {
    float f = (float)v[0];
    return (float)((float)(f + f) - (float)(f * -1.5f));
}
static double r_where_f32(const double *v) { return v[0]; }
static double r_where_mix(const double *v) {
    float w = v[0] > 0 ? (float)v[1] : (float)v[2];
    return (float)(w + (float)(v[0] < v[1]));
}
static double r_lt_eq(const double *v) { return (v[0] < v[1]) == (v[2] != 0); }
static double r_neg_lt(const double *v) { return (double)(int16_t)(-v[0] + ((float)v[1] < v[2])); }
static double r_abs_u8(const double *v) { return (double)(int16_t)(v[0] * fabs(v[1])); }
static double r_sq_bool(const double *v) { return v[0] + v[1] * v[1]; }
/* A non-bool where() condition selects by its truth, not by its value
 * converted to the result dtype (0.5 must not become an int 0). */
static double r_where_truth(const double *v) { return v[0] != 0 ? v[1] : v[2]; }
static double r_where_shifted(const double *v) { return v[0] + 0.5 != 0 ? v[1] : v[2]; }
/* A constant condition too, whatever dtype the parser gave it (a bool
 * output typed 0.5 as bool, and the int64 where() then read it as 0). */
static double r_where_const_gt(const double *v) { return v[0] > 0; }
static double r_where_const_truth(const double *v) { return v[0] != 0; }

static const conv_case cases[] = {
    {"(x + x) * abs(x)", 1, {"x"}, {ME_INT64}, ME_INT64, r_x2_abs, false},
    {"(x + x) * abs(x)", 1, {"x"}, {ME_INT16}, ME_INT16, r_x2_abs, false},
    {"(x + x) * abs(x)", 1, {"x"}, {ME_INT8}, ME_INT8, r_x2_abs, false},
    {"f + f > g", 2, {"f", "g"}, {ME_FLOAT32, ME_FLOAT64}, ME_BOOL, r_f2_gt, false},
    {"(a < c) + c", 2, {"a", "c"}, {ME_INT8, ME_UINT8}, ME_UINT8, r_lt_add, false},
    {"(a < c) * c", 2, {"a", "c"}, {ME_INT8, ME_UINT8}, ME_UINT8, r_lt_mul, false},
    {"(b * 3) * (-b)", 1, {"b"}, {ME_UINT32}, ME_UINT32, r_u32_neg, false},
    {"(f + f) - (f * -1.5)", 1, {"f"}, {ME_FLOAT32}, ME_FLOAT32, r_f32_mix, false},
    {"where((f + f) > -1, f, f)", 1, {"f"}, {ME_FLOAT32}, ME_FLOAT32, r_where_f32, false},
    {"where(a > 0, b, c) + (a < b)", 3, {"a", "b", "c"}, {ME_INT8, ME_INT16, ME_FLOAT32}, ME_FLOAT32, r_where_mix, false},
    {"(a < c) == b", 3, {"a", "c", "b"}, {ME_INT8, ME_UINT8, ME_BOOL}, ME_BOOL, r_lt_eq, false},
    {"(-a) + (b < c)", 3, {"a", "b", "c"}, {ME_INT16, ME_FLOAT32, ME_FLOAT64}, ME_INT16, r_neg_lt, false},
    {"c * abs(a)", 2, {"c", "a"}, {ME_UINT8, ME_INT8}, ME_INT16, r_abs_u8, false},
    {"b + a ** 2", 2, {"b", "a"}, {ME_UINT32, ME_BOOL}, ME_INT64, r_sq_bool, true},
    {"where(c, a, b)", 3, {"c", "a", "b"}, {ME_FLOAT64, ME_INT8, ME_INT16}, ME_INT16, r_where_truth, false},
    {"where(c + 0.5, a, b)", 3, {"c", "a", "b"}, {ME_FLOAT32, ME_INT8, ME_INT8}, ME_INT8, r_where_shifted, false},
    {"where(0.5, a, 2) > 0", 1, {"a"}, {ME_INT64}, ME_BOOL, r_where_const_gt, false},
    {"where(0.25, a, 2)", 1, {"a"}, {ME_INT64}, ME_BOOL, r_where_const_truth, true},
};

static void run_case(const conv_case *c, me_dtype out_dtype) {
    void *inputs[3] = {NULL, NULL, NULL};
    me_variable vars[3];
    memset(vars, 0, sizeof(vars));
    const char *expr = c->expr;
    for (int v = 0; v < c->nvars; v++) {
        inputs[v] = malloc(N * dtype_bytes(c->dtypes[v]));
        for (int i = 0; i < N; i++) store(inputs[v], c->dtypes[v], i, input_value(c->dtypes[v], v, i));
        vars[v].name = c->names[v];
        vars[v].dtype = c->dtypes[v];
    }

    int err = 0;
    me_expr *e = NULL;
    if (me_compile(expr, vars, c->nvars, out_dtype, &err, &e) != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compile '%s'\n", expr);
        failures++;
        goto done;
    }
    /* The output buffer follows the compiled dtype, so that a wrongly
     * inferred dtype is reported, not overflowed. */
    const me_dtype got_dtype = me_get_dtype(e);
    if (got_dtype != c->out) {
        printf("  FAIL: '%s' (%s output) has dtype %d, expected %d\n", expr,
               out_dtype == ME_AUTO ? "inferred" : "explicit", got_dtype, c->out);
        failures++;
    }
    void *out = calloc(N, dtype_bytes(got_dtype));
    const void *ptrs[3] = {inputs[0], inputs[1], inputs[2]};
    ME_EVAL_CHECK(e, ptrs, c->nvars, out, N);
    for (int i = 0; i < N; i++) {
        double v[3];
        for (int k = 0; k < c->nvars; k++) v[k] = load(inputs[k], c->dtypes[k], i);
        double want = c->ref(v);
        double got = load(out, got_dtype, i);
        double tol = c->out == ME_FLOAT32 ? 1e-6 * fabs(want) : 0.0;
        if (fabs(got - want) > tol) {
            printf("  FAIL: '%s' (%s output) [%d] = %.17g, expected %.17g\n", expr,
                   out_dtype == ME_AUTO ? "inferred" : "explicit", i, got, want);
            failures++;
            break;
        }
    }
    free(out);
    me_free(e);
done:
    for (int v = 0; v < c->nvars; v++) free(inputs[v]);
}

/* In DSL kernels bools are Python numbers: (x > 1) + (x < 3) is 2 for an
 * integer output too, not a 2 stored in a bool. */
static void check_dsl_bool_sum(void) {
    static const me_dtype outs[] = {ME_INT8, ME_INT64, ME_FLOAT64};
    const float x[3] = {1.5f, 0.5f, 4.0f};
    const double want[3] = {2, 1, 1};
    me_variable vars[] = {{"x", ME_FLOAT32}};
    for (int k = 0; k < 3; k++) {
        int64_t shape[] = {3};
        int32_t grid[] = {3};
        int err = 0;
        me_expr *e = NULL;
        if (me_compile_nd_jit("def k(x):\n    return (x > 1) + (x < 3)\n", vars, 1, outs[k], 1, shape, grid, grid,
                              ME_JIT_OFF, &err, &e) != ME_COMPILE_SUCCESS) {
            printf("  FAIL: compile DSL bool sum (output %d)\n", outs[k]);
            failures++;
            continue;
        }
        double out[3] = {0};
        const void *ptrs[] = {x};
        ME_EVAL_CHECK(e, ptrs, 1, out, 3);
        for (int i = 0; i < 3; i++) {
            const double got = load(out, outs[k], i);
            if (got != want[i]) {
                printf("  FAIL: DSL (x > 1) + (x < 3) (output %d) [%d] = %g, expected %g\n", outs[k], i, got,
                       want[i]);
                failures++;
                break;
            }
        }
        me_free(e);
    }
}

/* A reduction used as an operand is broadcast to every item, also when the
 * typing pass puts a conversion around it (max(x) * 1.5 converts an int32
 * max to float64) and as a comparison operand. */
static double red_x[N];
static double red_max, red_min, red_sum;
static double r_max_f(int i) { (void)i; return red_max * 1.5; }
static double r_sum_half(int i) { (void)i; return red_sum * 0.5; }
static double r_any_add(int i) { return 1.0 + red_x[i]; }
static double r_where_min(int i) { return red_x[i] > red_min; }
static double r_sum_gt(int i) { return red_sum > red_x[i]; }
static double r_x_minus_max(int i) { return red_x[i] - red_max * 1.5; }
static double r_x_plus_sum(int i) { return red_x[i] + red_sum; }
static double r_x_times_any(int i) { return red_x[i] * (red_sum != 0); }

static void check_reductions_in_expressions(void) {
    static int32_t x[N];
    red_max = -1e300;
    red_min = 1e300;
    red_sum = 0;
    for (int i = 0; i < N; i++) {
        x[i] = (int32_t)(i % 11) - 5;
        red_x[i] = x[i];
        red_max = fmax(red_max, x[i]);
        red_min = fmin(red_min, x[i]);
        red_sum += x[i];
    }
    static const struct {
        const char *src;
        double (*ref)(int);
    } rcases[] = {
        {"max(x) * 1.5", r_max_f},
        {"sum(x) * 0.5", r_sum_half},
        {"any(x > 3) + x", r_any_add},
        {"where(x > min(x), 1, 0)", r_where_min},
        {"sum(x) > x", r_sum_gt},
        {"def k(x):\n    m = max(x) * 1.5\n    return x - m\n", r_x_minus_max},
        /* DSL casts evaluate their argument as an operand too. */
        {"def k(x):\n    return x + int(sum(x))\n", r_x_plus_sum},
        {"def k(x):\n    return x * bool(sum(x))\n", r_x_times_any},
    };
    static const me_dtype outs[] = {ME_FLOAT64, ME_AUTO};
    me_variable vars[] = {{"x", ME_INT32}};
    const void *ptrs[] = {x};
    static double out[N];
    for (size_t k = 0; k < sizeof(rcases) / sizeof(rcases[0]); k++) {
        for (int o = 0; o < 2; o++) {
            int64_t shape[] = {N};
            int32_t grid[] = {N};
            int err = 0;
            me_expr *e = NULL;
            if (me_compile_nd_jit(rcases[k].src, vars, 1, outs[o], 1, shape, grid, grid, ME_JIT_OFF, &err, &e) !=
                ME_COMPILE_SUCCESS) {
                printf("  FAIL: compile '%s' (output %d)\n", rcases[k].src, outs[o]);
                failures++;
                continue;
            }
            memset(out, 0xAB, sizeof(out));
            ME_EVAL_CHECK(e, ptrs, 1, out, N);
            const me_dtype dt = me_get_dtype(e);
            for (int i = 0; i < N; i++) {
                const double got = load(out, dt, i), want = rcases[k].ref(i);
                if (got != want) {
                    printf("  FAIL: '%s' (output %d) [%d] = %g, expected %g\n", rcases[k].src, outs[o], i, got, want);
                    failures++;
                    break;
                }
            }
            me_free(e);
        }
    }
}

/* A caller's function registered with ME_AUTO keeps the parse-time dtype
 * (the header: ME_AUTO = use output dtype); it is not typed by its operands. */
static double user_half(double x) { return x / 2; }

static void check_user_function(void) {
    const int32_t a[4] = {1, 2, 3, 4};
    const double b[4] = {0.25, 0.5, 0.75, 1.0};
    me_variable vars[] = {{"a", ME_INT32}, {"b", ME_FLOAT64}, {"half", ME_AUTO, (const void *)user_half, ME_FUNCTION1 | ME_FLAG_PURE}};
    int err = 0;
    me_expr *e = NULL;
    if (me_compile("half(a) + b", vars, 3, ME_FLOAT64, &err, &e) != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compile 'half(a) + b'\n");
        failures++;
        return;
    }
    double out[4];
    const void *ptrs[] = {a, b};
    ME_EVAL_CHECK(e, ptrs, 2, out, 4);
    for (int i = 0; i < 4; i++) {
        if (out[i] != a[i] / 2.0 + b[i]) {
            printf("  FAIL: 'half(a) + b' [%d] = %g, expected %g\n", i, out[i], a[i] / 2.0 + b[i]);
            failures++;
            break;
        }
    }
    me_free(e);
}

int main(void) {
    printf("Test: mixed-dtype expressions with conversion nodes\n");
    check_user_function();
    check_dsl_bool_sum();
    check_reductions_in_expressions();
    for (size_t k = 0; k < sizeof(cases) / sizeof(cases[0]); k++) {
        run_case(&cases[k], cases[k].out);
        if (!cases[k].explicit_only) run_case(&cases[k], ME_AUTO);
    }
    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("All conversion-node tests passed\n");
    return 0;
}
