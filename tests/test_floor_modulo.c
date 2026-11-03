/* `%` and remainder() follow NumPy's floor modulo (the result takes the sign of
 * the divisor) for every real dtype and engine; fmod() stays C fmod. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/miniexpr.h"
#include "../src/functions-simd.h"

/* NumPy npy_remainder: fmod, then move a nonzero result to the divisor's side
 * and give a zero result the divisor's sign. */
static double ref_mod_f64(double a, double b) {
    double r = fmod(a, b);
    if (r != 0.0) {
        if ((signbit(r) != 0) != (signbit(b) != 0)) r += b;
    }
    else {
        r = copysign(0.0, b);
    }
    return r;
}

static float ref_mod_f32(float a, float b) {
    float r = fmodf(a, b);
    if (r != 0.0f) {
        if ((signbit(r) != 0) != (signbit(b) != 0)) r += b;
    }
    else {
        r = copysignf(0.0f, b);
    }
    return r;
}

/* NumPy integer remainder: 0 for a zero divisor, MIN % -1 is 0 (no trap). */
static int64_t ref_mod_i64(int64_t a, int64_t b) {
    if (b == 0 || b == -1) return 0;
    int64_t r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) r += b;
    return r;
}

static uint64_t ref_mod_u64(uint64_t a, uint64_t b) {
    return b ? a % b : 0;
}

typedef enum { KIND_SIGNED, KIND_UNSIGNED, KIND_FLOAT } value_kind;

typedef struct {
    me_dtype dtype;
    const char *name;
    size_t size;
    value_kind kind;
    int bits;
} type_info;

static const type_info types[] = {
    {ME_INT8, "int8", 1, KIND_SIGNED, 8},
    {ME_INT16, "int16", 2, KIND_SIGNED, 16},
    {ME_INT32, "int32", 4, KIND_SIGNED, 32},
    {ME_INT64, "int64", 8, KIND_SIGNED, 64},
    {ME_UINT8, "uint8", 1, KIND_UNSIGNED, 8},
    {ME_UINT16, "uint16", 2, KIND_UNSIGNED, 16},
    {ME_UINT32, "uint32", 4, KIND_UNSIGNED, 32},
    {ME_UINT64, "uint64", 8, KIND_UNSIGNED, 64},
    {ME_FLOAT32, "float32", 4, KIND_FLOAT, 32},
    {ME_FLOAT64, "float64", 8, KIND_FLOAT, 64},
};

/* Each sample is repeated so SIMD bodies and scalar tails both see it. */
#define NSAMPLES 26
#define NREPEAT 5
#define NITEMS (NSAMPLES * NREPEAT + 3)

typedef struct {
    int64_t s[NITEMS];
    uint64_t u[NITEMS];
    double f[NITEMS];
} values;

static void fill_samples(const type_info *t, values *a, values *b) {
    if (t->kind == KIND_SIGNED) {
        int64_t mx = t->bits == 64 ? INT64_MAX : (int64_t)((UINT64_C(1) << (t->bits - 1)) - 1);
        int64_t mn = -mx - 1;
        int64_t sa[NSAMPLES] = {-7, 7, -7, 7, 0, 0, 5, -5, mn, mn, mx, mn, -1, mn, 100, -100,
                                13, -13, mx, mx, mn, mn, 1, -1, 6, -6};
        int64_t sb[NSAMPLES] = {3, -3, -3, 3, 3, -3, 0, 0, -1, 1, -1, mx, mn, 3, 7, 7,
                                -5, -5, 10, -10, 10, -10, mn, mx, 3, 3};
        for (int i = 0; i < NITEMS; i++) {
            a->s[i] = sa[i % NSAMPLES];
            b->s[i] = sb[i % NSAMPLES];
        }
    }
    else if (t->kind == KIND_UNSIGNED) {
        uint64_t mx = t->bits == 64 ? UINT64_MAX : (UINT64_C(1) << t->bits) - 1;
        uint64_t ua[NSAMPLES] = {7, 0, 5, mx, mx, mx, 1, 100, 13, 6, 0, mx - 1, 7, 7, 3, 9,
                                 mx, 2, 200, 250, 1, 0, 8, 64, 65, 255};
        uint64_t ub[NSAMPLES] = {3, 3, 0, 7, mx - 1, 10, mx, 7, 5, 3, 0, mx, 1, 7, 9, 3,
                                 mx, 2, 3, 7, 1, 5, 3, 7, 8, 16};
        for (int i = 0; i < NITEMS; i++) {
            a->u[i] = ua[i % NSAMPLES];
            b->u[i] = ub[i % NSAMPLES];
        }
    }
    else {
        double fa[NSAMPLES] = {-7.0, 7.0, -7.0, 7.0, 0.0, -0.0, 0.0, -0.0, 5.0, -5.0, 5.0, -5.0,
                               INFINITY, -INFINITY, NAN, 3.0, 3.0, -1e-20, 1e-20, 6.0, -6.0, 5.0,
                               5.5, -5.5, 1e30, -0.25};
        double fb[NSAMPLES] = {3.0, -3.0, -3.0, 3.0, 3.0, 3.0, -3.0, -3.0, INFINITY, INFINITY,
                               -INFINITY, -INFINITY, 3.0, 3.0, 3.0, NAN, 0.0, 1.0, -1.0, 3.0, 3.0,
                               -0.0, 2.0, 2.0, -7.0, 0.5};
        for (int i = 0; i < NITEMS; i++) {
            a->f[i] = fa[i % NSAMPLES];
            b->f[i] = fb[i % NSAMPLES];
        }
    }
}

static void store(const type_info *t, void *buf, int i, const values *v) {
    char *p = (char *)buf + (size_t)i * t->size;
    switch (t->dtype) {
    case ME_INT8: *(int8_t *)p = (int8_t)v->s[i]; break;
    case ME_INT16: *(int16_t *)p = (int16_t)v->s[i]; break;
    case ME_INT32: *(int32_t *)p = (int32_t)v->s[i]; break;
    case ME_INT64: *(int64_t *)p = v->s[i]; break;
    case ME_UINT8: *(uint8_t *)p = (uint8_t)v->u[i]; break;
    case ME_UINT16: *(uint16_t *)p = (uint16_t)v->u[i]; break;
    case ME_UINT32: *(uint32_t *)p = (uint32_t)v->u[i]; break;
    case ME_UINT64: *(uint64_t *)p = v->u[i]; break;
    case ME_FLOAT32: *(float *)p = (float)v->f[i]; break;
    case ME_FLOAT64: *(double *)p = v->f[i]; break;
    default: break;
    }
}

/* Compare one output element against the reference of (x, y) in dtype t. */
static bool check_element(const type_info *t, const void *buf, int i, const values *x, int xi,
                          const values *y, int yi, char *msg, size_t cap) {
    const char *p = (const char *)buf + (size_t)i * t->size;
    if (t->kind == KIND_SIGNED) {
        int64_t got = t->dtype == ME_INT8 ? *(const int8_t *)p : t->dtype == ME_INT16 ? *(const int16_t *)p :
                      t->dtype == ME_INT32 ? *(const int32_t *)p : *(const int64_t *)p;
        int64_t want = ref_mod_i64(x->s[xi], y->s[yi]);
        if (got == want) return true;
        snprintf(msg, cap, "%lld %% %lld: got %lld, want %lld", (long long)x->s[xi], (long long)y->s[yi],
                 (long long)got, (long long)want);
        return false;
    }
    if (t->kind == KIND_UNSIGNED) {
        uint64_t got = t->dtype == ME_UINT8 ? *(const uint8_t *)p : t->dtype == ME_UINT16 ? *(const uint16_t *)p :
                       t->dtype == ME_UINT32 ? *(const uint32_t *)p : *(const uint64_t *)p;
        uint64_t want = ref_mod_u64(x->u[xi], y->u[yi]);
        if (got == want) return true;
        snprintf(msg, cap, "%llu %% %llu: got %llu, want %llu", (unsigned long long)x->u[xi],
                 (unsigned long long)y->u[yi], (unsigned long long)got, (unsigned long long)want);
        return false;
    }
    double got, want;
    if (t->dtype == ME_FLOAT32) {
        got = *(const float *)p;
        want = ref_mod_f32((float)x->f[xi], (float)y->f[yi]);
    }
    else {
        got = *(const double *)p;
        want = ref_mod_f64(x->f[xi], y->f[yi]);
    }
    if ((isnan(got) && isnan(want)) || (got == want && signbit(got) == signbit(want))) return true;
    snprintf(msg, cap, "%g %% %g: got %g, want %g", x->f[xi], y->f[yi], got, want);
    return false;
}

typedef enum {
    MODE_INTERP,       /* me_compile + me_eval */
    MODE_INTERP_NOSIMD,
    MODE_ND_JIT_ON,    /* plain expression auto-lifted into a DSL kernel */
    MODE_DSL_JIT_OFF,  /* DSL kernel, interpreter */
    MODE_DSL_JIT_ON,   /* DSL kernel, JIT where the backend accepts it */
    MODE_COUNT
} eval_mode;

static const char *mode_names[] = {"interp", "interp-nosimd", "nd-jit-on", "dsl-jit-off", "dsl-jit-on"};

/* Operand shapes. A constant operand uses the scalar in `cvals`. */
typedef enum { FORM_ARRAY_ARRAY, FORM_ARRAY_CONST, FORM_CONST_ARRAY } operand_form;

/* Negative literals are currently typed float64 (a separate literal-typing
 * issue), so an integer node with one is computed in double. Skip the samples
 * where that, not the modulo rule, decides: x % 0 and operands beyond 2**53. */
static bool literal_typing_sample(const type_info *t, int64_t constant, const values *x, const values *y,
                                  int i) {
    if (constant >= 0 || t->kind != KIND_SIGNED) return false;
    const int64_t limit = INT64_C(1) << 53;
    return y->s[i] == 0 || x->s[i] > limit || x->s[i] < -limit || y->s[i] > limit || y->s[i] < -limit;
}

static int run_case(const type_info *t, const char *expression, operand_form form, int64_t constant,
                    eval_mode mode, const values *a, const values *b) {
    void *abuf = malloc(NITEMS * t->size), *bbuf = malloc(NITEMS * t->size), *out = calloc(NITEMS, t->size);
    if (!abuf || !bbuf || !out) {
        free(abuf);
        free(bbuf);
        free(out);
        return 1;
    }
    for (int i = 0; i < NITEMS; i++) {
        store(t, abuf, i, a);
        store(t, bbuf, i, b);
    }
    /* The constant, as the operand it replaces. */
    values cvals;
    for (int i = 0; i < NITEMS; i++) {
        cvals.s[i] = constant;
        cvals.u[i] = (uint64_t)constant;
        cvals.f[i] = (double)constant;
    }

    me_variable vars[2] = {{"a", t->dtype}, {"b", t->dtype}};
    const void *inputs[2] = {abuf, bbuf};
    int nvars = 2;
    if (form == FORM_ARRAY_CONST) {
        nvars = 1;
    }
    else if (form == FORM_CONST_ARRAY) {
        vars[0] = vars[1];
        inputs[0] = bbuf;
        nvars = 1;
    }

    char source[256];
    const char *text = expression;
    if (mode == MODE_DSL_JIT_OFF || mode == MODE_DSL_JIT_ON) {
        snprintf(source, sizeof(source), "def k(%s):\n    return %s\n", nvars == 2 ? "a, b" : vars[0].name,
                 expression);
        text = source;
    }

    me_expr *expr = NULL;
    int error = 0, rc;
    if (mode == MODE_INTERP || mode == MODE_INTERP_NOSIMD) {
        rc = me_compile(text, vars, nvars, t->dtype, &error, &expr);
    }
    else {
        int64_t shape[] = {NITEMS};
        int32_t grid[] = {NITEMS};
        int jit = mode == MODE_DSL_JIT_OFF ? ME_JIT_OFF : ME_JIT_ON;
        rc = me_compile_nd_jit(text, vars, nvars, t->dtype, 1, shape, grid, grid, jit, &error, &expr);
    }
    int failures = 0;
    if (rc != ME_COMPILE_SUCCESS || !expr) {
        printf("  FAIL %s %s [%s]: compile rc=%d at %d\n", t->name, expression, mode_names[mode], rc, error);
        failures = 1;
        goto done;
    }
    me_eval_params params = ME_EVAL_PARAMS_DEFAULTS;
    params.disable_simd = mode == MODE_INTERP_NOSIMD;
    params.jit_mode = mode == MODE_DSL_JIT_OFF ? ME_JIT_OFF : ME_JIT_DEFAULT;
    rc = me_eval(expr, inputs, nvars, out, NITEMS, &params);
    if (rc != ME_EVAL_SUCCESS) {
        printf("  FAIL %s %s [%s]: eval rc=%d\n", t->name, expression, mode_names[mode], rc);
        failures = 1;
        goto done;
    }
    const values *x = form == FORM_CONST_ARRAY ? &cvals : a;
    const values *y = form == FORM_ARRAY_ARRAY ? b : form == FORM_CONST_ARRAY ? b : &cvals;
    for (int i = 0; i < NITEMS; i++) {
        char msg[192];
        if (literal_typing_sample(t, constant, x, y, i)) continue;
        if (!check_element(t, out, i, x, i, y, i, msg, sizeof(msg))) {
            printf("  FAIL %s %s [%s%s] at %d: %s\n", t->name, expression, mode_names[mode],
                   me_expr_has_jit_kernel(expr) ? ", jit" : "", i, msg);
            failures = 1;
            break;
        }
    }

done:
    me_free(expr);
    free(abuf);
    free(bbuf);
    free(out);
    return failures;
}

static void set_env(const char *name, const char *value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

static int test_dtype_matrix(int nmodes) {
    int failures = 0;
    for (size_t ti = 0; ti < sizeof(types) / sizeof(types[0]); ti++) {
        const type_info *t = &types[ti];
        values a, b;
        fill_samples(t, &a, &b);
        bool is_signed = t->kind != KIND_UNSIGNED;
        for (int mode = 0; mode < nmodes; mode++) {
            failures += run_case(t, "a % b", FORM_ARRAY_ARRAY, 0, mode, &a, &b);
            failures += run_case(t, "remainder(a, b)", FORM_ARRAY_ARRAY, 0, mode, &a, &b);
            /* An intermediate node, not only the root. */
            failures += run_case(t, "(a % b) * 1", FORM_ARRAY_ARRAY, 0, mode, &a, &b);
            failures += run_case(t, "a % 3", FORM_ARRAY_CONST, 3, mode, &a, &b);
            failures += run_case(t, "a % 0", FORM_ARRAY_CONST, 0, mode, &a, &b);
            failures += run_case(t, "remainder(a, 7)", FORM_ARRAY_CONST, 7, mode, &a, &b);
            failures += run_case(t, "7 % b", FORM_CONST_ARRAY, 7, mode, &a, &b);
            if (is_signed) {
                failures += run_case(t, "a % -3", FORM_ARRAY_CONST, -3, mode, &a, &b);
                failures += run_case(t, "-7 % b", FORM_CONST_ARRAY, -7, mode, &a, &b);
                failures += run_case(t, "remainder(-7, b)", FORM_CONST_ARRAY, -7, mode, &a, &b);
            }
        }
    }
    return failures;
}

/* fmod() keeps C fmod semantics (np.fmod): the sign of the dividend. */
static int test_fmod_unchanged(void) {
    int failures = 0;
    double a64[] = {-7.0, 7.0, -7.0, 7.0, -0.0, 5.0, -5.0};
    double b64[] = {3.0, -3.0, -3.0, 3.0, 3.0, INFINITY, INFINITY};
    float a32[7], b32[7];
    for (int i = 0; i < 7; i++) {
        a32[i] = (float)a64[i];
        b32[i] = (float)b64[i];
    }
    for (int dt = 0; dt < 2; dt++) {
        me_variable vars[] = {{"a", dt ? ME_FLOAT32 : ME_FLOAT64}, {"b", dt ? ME_FLOAT32 : ME_FLOAT64}};
        const void *inputs[] = {dt ? (const void *)a32 : a64, dt ? (const void *)b32 : b64};
        double out64[7];
        float out32[7];
        me_expr *expr = NULL;
        int error = 0;
        if (me_compile("fmod(a, b)", vars, 2, vars[0].dtype, &error, &expr) != ME_COMPILE_SUCCESS ||
            me_eval(expr, inputs, 2, dt ? (void *)out32 : out64, 7, NULL) != ME_EVAL_SUCCESS) {
            printf("  FAIL fmod(a, b) %s: compile/eval\n", dt ? "float32" : "float64");
            me_free(expr);
            failures++;
            continue;
        }
        for (int i = 0; i < 7; i++) {
            double got = dt ? out32[i] : out64[i];
            double want = dt ? fmodf(a32[i], b32[i]) : fmod(a64[i], b64[i]);
            if (got != want || signbit(got) != signbit(want)) {
                printf("  FAIL fmod(%g, %g) %s: got %g, want %g\n", a64[i], b64[i],
                       dt ? "float32" : "float64", got, want);
                failures++;
                break;
            }
        }
        me_free(expr);
    }
    return failures;
}

/* Mixed operands: the integer operand is promoted, then floor modulo applies. */
static int test_mixed_int_float(void) {
    int32_t a[] = {-7, 7, -7, 7, 0, -1};
    double b[] = {3.0, -3.0, -3.0, 3.0, -2.0, 0.5};
    double out[6];
    me_variable vars[] = {{"a", ME_INT32}, {"b", ME_FLOAT64}};
    const void *inputs[] = {a, b};
    int failures = 0;
    const char *expressions[] = {"a % b", "remainder(a, b)"};
    for (int e = 0; e < 2; e++) {
        me_expr *expr = NULL;
        int error = 0;
        if (me_compile(expressions[e], vars, 2, ME_FLOAT64, &error, &expr) != ME_COMPILE_SUCCESS ||
            me_eval(expr, inputs, 2, out, 6, NULL) != ME_EVAL_SUCCESS) {
            printf("  FAIL mixed %s: compile/eval\n", expressions[e]);
            me_free(expr);
            failures++;
            continue;
        }
        for (int i = 0; i < 6; i++) {
            double want = ref_mod_f64((double)a[i], b[i]);
            if (out[i] != want || signbit(out[i]) != signbit(want)) {
                printf("  FAIL mixed %s: %d %% %g got %g, want %g\n", expressions[e], a[i], b[i], out[i], want);
                failures++;
                break;
            }
        }
        me_free(expr);
    }
    return failures;
}

/* bool % bool is integral modulo of 0/1 (NumPy: int8, always 0). */
static int test_bool_operands(void) {
    bool a[] = {true, false, true, false};
    bool b[] = {true, true, false, false};
    me_variable vars[] = {{"a", ME_BOOL}, {"b", ME_BOOL}};
    const void *inputs[] = {a, b};
    int failures = 0;
    const char *expressions[] = {"a % b", "remainder(a, b)"};
    for (int e = 0; e < 2; e++) {
        int8_t out[4] = {9, 9, 9, 9};
        me_expr *expr = NULL;
        int error = 0;
        if (me_compile(expressions[e], vars, 2, ME_INT8, &error, &expr) != ME_COMPILE_SUCCESS ||
            me_eval(expr, inputs, 2, out, 4, NULL) != ME_EVAL_SUCCESS) {
            printf("  FAIL bool %s: compile/eval\n", expressions[e]);
            me_free(expr);
            failures++;
            continue;
        }
        for (int i = 0; i < 4; i++) {
            if (out[i] != 0) {
                printf("  FAIL bool %s at %d: got %d, want 0\n", expressions[e], i, out[i]);
                failures++;
                break;
            }
        }
        me_free(expr);
    }
    return failures;
}

/* Folded constants and DSL locals use the same rule. */
static int test_constants_and_compound(void) {
    int failures = 0;
    struct { const char *text; double want; } folded[] = {
        {"-7 % 3", 2.0}, {"7 % -3", -2.0}, {"-7.5 % 2", 0.5}, {"remainder(-7, 3)", 2.0},
        {"remainder(7, -3)", -2.0}, {"fmod(-7, 3)", -1.0},
    };
    for (size_t i = 0; i < sizeof(folded) / sizeof(folded[0]); i++) {
        double x = 0.0, out = 0.0;
        me_variable vars[] = {{"x", ME_FLOAT64}};
        const void *inputs[] = {&x};
        char text[64];
        snprintf(text, sizeof(text), "x + (%s)", folded[i].text);
        me_expr *expr = NULL;
        int error = 0;
        if (me_compile(text, vars, 1, ME_FLOAT64, &error, &expr) != ME_COMPILE_SUCCESS ||
            me_eval(expr, inputs, 1, &out, 1, NULL) != ME_EVAL_SUCCESS || out != folded[i].want) {
            printf("  FAIL constant %s: got %g, want %g\n", folded[i].text, out, folded[i].want);
            failures++;
        }
        me_free(expr);
    }

    const char *source = "def k(a, b):\n    r = a % b\n    s = remainder(r + b, b)\n    return s\n";
    int32_t a[] = {-7, 7, -7, 7, 5, INT32_MIN};
    int32_t b[] = {3, -3, -3, 3, 0, -1};
    int32_t out[6];
    me_variable vars[] = {{"a", ME_INT32}, {"b", ME_INT32}};
    const void *inputs[] = {a, b};
    for (int jit = 0; jit < 2; jit++) {
        int64_t shape[] = {6};
        int32_t grid[] = {6};
        me_expr *expr = NULL;
        int error = 0;
        if (me_compile_nd_jit(source, vars, 2, ME_INT32, 1, shape, grid, grid, jit ? ME_JIT_ON : ME_JIT_OFF,
                              &error, &expr) != ME_COMPILE_SUCCESS ||
            me_eval(expr, inputs, 2, out, 6, NULL) != ME_EVAL_SUCCESS) {
            printf("  FAIL DSL locals (jit=%d): compile/eval\n", jit);
            me_free(expr);
            failures++;
            continue;
        }
        for (int i = 0; i < 6; i++) {
            int64_t r = ref_mod_i64(a[i], b[i]);
            int64_t want = ref_mod_i64((int32_t)(r + b[i]), b[i]);
            if (out[i] != want) {
                printf("  FAIL DSL locals (jit=%d): %d %% %d got %d, want %lld\n", jit, a[i], b[i], out[i],
                       (long long)want);
                failures++;
                break;
            }
        }
        me_free(expr);
    }
    return failures;
}

int main(void) {
    int failures = 0;
    printf("=== floor modulo: dtype x operand x engine matrix ===\n");
    failures += test_dtype_matrix(MODE_COUNT);
    /* The SLEEF backend has its own fused floor-modulo kernels. */
    printf("=== floor modulo: interpreter with the SLEEF backend ===\n");
    set_env("ME_SIMD_MATH_BACKEND", "sleef");
    me_simd_reset_for_tests();
    failures += test_dtype_matrix(MODE_INTERP_NOSIMD + 1);
    set_env("ME_SIMD_MATH_BACKEND", NULL);
    me_simd_reset_for_tests();
    printf("=== fmod keeps C semantics ===\n");
    failures += test_fmod_unchanged();
    printf("=== mixed int/float operands ===\n");
    failures += test_mixed_int_float();
    printf("=== bool operands ===\n");
    failures += test_bool_operands();
    printf("=== folded constants and DSL locals ===\n");
    failures += test_constants_and_compound();
    if (failures) {
        printf("FAILED: %d case(s)\n", failures);
        return 1;
    }
    printf("All floor modulo tests passed\n");
    return 0;
}
