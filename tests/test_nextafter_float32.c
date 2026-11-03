/*
 * nextafter() of float32 operands steps by a float32 ulp, as np.nextafter
 * does.  With a constant operand (nextafter(a, 2)) the interpreter used the
 * generic double loop: it stepped by a float64 ulp, and rounding the result
 * back to float32 returned the input unchanged.
 */

#include "../src/miniexpr.h"
#include "minctest.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define N 37

static int failures = 0;

typedef enum { MODE_INTERP, MODE_INTERP_NOSIMD, MODE_ND_JIT_ON, MODE_DSL_JIT_OFF, MODE_DSL_JIT_ON, MODE_COUNT } eval_mode;
static const char *mode_names[] = {"interp", "interp-nosimd", "nd-jit-on", "dsl-jit-off", "dsl-jit-on"};

typedef struct {
    const char *expr;
    int nvars;  /* uses a, or a and b */
    double (*ref)(double a, double b, bool f32);
} na_case;

static double step(double x, double toward, bool f32) {
    return f32 ? (double)nextafterf((float)x, (float)toward) : nextafter(x, toward);
}
static double r_ab(double a, double b, bool f32) { return step(a, b, f32); }
static double r_a2(double a, double b, bool f32) { (void)b; return step(a, 2, f32); }
static double r_a_tenth(double a, double b, bool f32) { (void)b; return step(a, f32 ? (double)0.1f : 0.1, f32); }
static double r_half_b(double a, double b, bool f32) { (void)a; return step(0.5, b, f32); }
static double r_sum_b(double a, double b, bool f32) {
    double s = f32 ? (double)((float)a + (float)b) : a + b;
    return step(s, b, f32);
}
static double r_ulp(double a, double b, bool f32) {
    (void)b;
    double s = step(a, 2, f32);
    return f32 ? (double)((float)s - (float)a) : s - a;
}

static const na_case cases[] = {
    {"nextafter(a, b)", 2, r_ab},
    {"nextafter(a, 2)", 1, r_a2},
    {"nextafter(a, 0.1)", 1, r_a_tenth},
    {"nextafter(0.5, b)", 2, r_half_b},
    {"nextafter(a + b, b)", 2, r_sum_b},
    {"nextafter(a, 2) - a", 1, r_ulp},
};

static void run(const na_case *c, me_dtype dt, eval_mode mode) {
    const bool f32 = dt == ME_FLOAT32;
    float af[N], bf[N], of[N];
    double ad[N], bd[N], od[N];
    for (int i = 0; i < N; i++) {
        ad[i] = (i % 2 ? -1.0 : 1.0) * (0.25 * i + 0.1);
        bd[i] = (i % 3) - 1.0 + (i == 5 ? INFINITY : 0.0);
        if (i == 7) ad[i] = 0.0;
        if (i == 8) ad[i] = -0.0;
        if (i == 9) ad[i] = NAN;
        af[i] = (float)ad[i];
        bf[i] = (float)bd[i];
        if (f32) {
            ad[i] = af[i];
            bd[i] = bf[i];
        }
    }
    /* "nextafter(0.5, b)" uses only b. */
    const bool only_b = strncmp(c->expr, "nextafter(0.5", 13) == 0;
    me_variable vars[2] = {{"a", dt}, {"b", dt}};
    const void *ptrs[2] = {f32 ? (const void *)af : (const void *)ad, f32 ? (const void *)bf : (const void *)bd};
    int nvars = c->nvars;
    if (only_b) {
        vars[0] = vars[1];
        ptrs[0] = ptrs[1];
        nvars = 1;
    }
    char source[256];
    const char *text = c->expr;
    if (mode == MODE_DSL_JIT_OFF || mode == MODE_DSL_JIT_ON) {
        snprintf(source, sizeof(source), "def k(%s):\n    return %s\n", nvars == 2 ? "a, b" : vars[0].name, c->expr);
        text = source;
    }
    me_expr *e = NULL;
    int err = 0, rc;
    if (mode == MODE_INTERP || mode == MODE_INTERP_NOSIMD) {
        rc = me_compile(text, vars, nvars, dt, &err, &e);
    }
    else {
        int64_t shape[] = {N};
        int32_t grid[] = {N};
        rc = me_compile_nd_jit(text, vars, nvars, dt, 1, shape, grid, grid,
                               mode == MODE_DSL_JIT_OFF ? ME_JIT_OFF : ME_JIT_ON, &err, &e);
    }
    if (rc != ME_COMPILE_SUCCESS || !e) {
        printf("  FAIL: compile '%s' [%s] rc=%d\n", c->expr, mode_names[mode], rc);
        failures++;
        return;
    }
    me_eval_params params = ME_EVAL_PARAMS_DEFAULTS;
    params.disable_simd = mode == MODE_INTERP_NOSIMD;
    params.jit_mode = mode == MODE_DSL_JIT_OFF ? ME_JIT_OFF : ME_JIT_DEFAULT;
    if (me_eval(e, ptrs, nvars, f32 ? (void *)of : (void *)od, N, &params) != ME_EVAL_SUCCESS) {
        printf("  FAIL: eval '%s' [%s]\n", c->expr, mode_names[mode]);
        failures++;
        me_free(e);
        return;
    }
    for (int i = 0; i < N; i++) {
        const double want = c->ref(ad[i], bd[i], f32);
        const double got = f32 ? (double)of[i] : od[i];
        if (!(got == want || (isnan(got) && isnan(want)))) {
            printf("  FAIL: '%s' (%s) [%s%s] a=%.9g b=%.9g: got %.9g, expected %.9g\n", c->expr,
                   f32 ? "float32" : "float64", mode_names[mode], me_expr_has_jit_kernel(e) ? ", jit" : "",
                   ad[i], bd[i], got, want);
            failures++;
            break;
        }
    }
    me_free(e);
}

int main(void) {
    printf("Test: nextafter() steps by an ulp of the operands' dtype\n");
    static const me_dtype dtypes[] = {ME_FLOAT32, ME_FLOAT64};
    for (int d = 0; d < 2; d++) {
        for (size_t k = 0; k < sizeof(cases) / sizeof(cases[0]); k++) {
            for (int mode = 0; mode < MODE_COUNT; mode++) run(&cases[k], dtypes[d], (eval_mode)mode);
        }
    }
    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("All nextafter tests passed\n");
    return 0;
}
