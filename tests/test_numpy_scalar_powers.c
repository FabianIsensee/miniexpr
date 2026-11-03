/*
 * NumPy computes float_array ** 0.5 as np.sqrt() and float_array ** -1 as
 * np.reciprocal() (complex_array ** 0.5 as np.sqrt() too).  miniexpr called
 * pow(), which differs: (-inf) ** 0.5 is inf (NumPy: NaN), (-0.0) ** 0.5 is
 * 0.0 (NumPy: -0.0), and pow() is not always correctly rounded.  Integer
 * bases keep pow(), as in NumPy.
 */

#include "../src/miniexpr.h"
#include "minctest.h"
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define N 2048

static int failures = 0;

static bool same_bits_f64(double a, double b) {
    if (isnan(a) && isnan(b)) return true;
    return memcmp(&a, &b, sizeof(a)) == 0;
}

static bool same_bits_f32(float a, float b) {
    if (isnan(a) && isnan(b)) return true;
    return memcmp(&a, &b, sizeof(a)) == 0;
}

static double sample(int i) {
    static const double special[] = {-0.0, 0.0, -INFINITY, INFINITY, NAN, -1.0, 2.0, 1e-310};
    if (i < 8) return special[i];
    /* Spread over many binades. */
    return ldexp(1.0 + (i * 0.618033988749895 - floor(i * 0.618033988749895)), i % 120 - 60);
}

static me_expr *compile1(const char *expr, me_dtype dt) {
    me_variable vars[] = {{"x", dt}};
    int err = 0;
    me_expr *e = NULL;
    if (me_compile(expr, vars, 1, dt, &err, &e) != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compile '%s' (dtype %d)\n", expr, dt);
        failures++;
        return NULL;
    }
    return e;
}

static void check_real(const char *expr, bool reciprocal) {
    static double xd[N], od[N];
    static float xf[N], of[N];
    for (int i = 0; i < N; i++) {
        xd[i] = sample(i);
        xf[i] = (float)xd[i];
    }
    for (int f32 = 0; f32 < 2; f32++) {
        me_expr *e = compile1(expr, f32 ? ME_FLOAT32 : ME_FLOAT64);
        if (!e) continue;
        const void *ptrs[] = {f32 ? (const void *)xf : (const void *)xd};
        ME_EVAL_CHECK(e, ptrs, 1, f32 ? (void *)of : (void *)od, N);
        me_free(e);
        for (int i = 0; i < N; i++) {
            bool ok;
            /* (x + 0): -0.0 + 0 is +0.0. */
            const bool plus0 = strstr(expr, "+ 0") != NULL;
            const float bf = plus0 ? xf[i] + 0.0f : xf[i];
            const double bd = plus0 ? xd[i] + 0.0 : xd[i];
            if (f32) ok = same_bits_f32(of[i], reciprocal ? 1.0f / bf : sqrtf(bf));
            else ok = same_bits_f64(od[i], reciprocal ? 1.0 / bd : sqrt(bd));
            if (!ok) {
                printf("  FAIL: '%s' (%s) x=%.17g: got %.17g\n", expr, f32 ? "float32" : "float64",
                       f32 ? (double)xf[i] : xd[i], f32 ? (double)of[i] : od[i]);
                failures++;
                break;
            }
        }
    }
}

static void check_complex(const char *expr) {
    static double _Complex xd[N], od[N];
    for (int i = 0; i < N; i++) xd[i] = sample(i + 8) - 0.5 * sample(N - 1 - i) * I;
    me_expr *e = compile1(expr, ME_COMPLEX128);
    if (!e) return;
    const void *ptrs[] = {xd};
    ME_EVAL_CHECK(e, ptrs, 1, od, N);
    me_free(e);
    for (int i = 0; i < N; i++) {
        const double _Complex want = csqrt(xd[i]);
        if (!same_bits_f64(creal(od[i]), creal(want)) || !same_bits_f64(cimag(od[i]), cimag(want))) {
            printf("  FAIL: '%s' [%d]: got (%.17g,%.17g), expected (%.17g,%.17g)\n", expr, i, creal(od[i]),
                   cimag(od[i]), creal(want), cimag(want));
            failures++;
            return;
        }
    }
}

/* An integer base keeps pow(), as NumPy's power loop does. */
static void check_integer_base(void) {
    static int32_t xi[N];
    static double od[N];
    for (int i = 0; i < N; i++) xi[i] = i * 7919;
    me_variable vars[] = {{"x", ME_INT32}};
    int err = 0;
    me_expr *e = NULL;
    if (me_compile("x ** 0.5", vars, 1, ME_FLOAT64, &err, &e) != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compile int32 x ** 0.5\n");
        failures++;
        return;
    }
    const void *ptrs[] = {xi};
    ME_EVAL_CHECK(e, ptrs, 1, od, N);
    me_free(e);
    for (int i = 0; i < N; i++) {
        if (!same_bits_f64(od[i], pow((double)xi[i], 0.5))) {
            printf("  FAIL: int32 x ** 0.5 x=%d: got %.17g, expected pow() %.17g\n", xi[i], od[i],
                   pow((double)xi[i], 0.5));
            failures++;
            return;
        }
    }
}

int main(void) {
    printf("Test: float ** 0.5 and ** -1 as NumPy's sqrt() and reciprocal()\n");
    check_real("x ** 0.5", false);
    check_real("(x + 0) ** 0.5", false);
    check_real("x ** -1", true);
    check_real("x ** (1 - 2)", true);
    check_complex("x ** 0.5");
    check_integer_base();
    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("All scalar-power tests passed\n");
    return 0;
}
