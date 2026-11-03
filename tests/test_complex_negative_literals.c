/*
 * Negative literals and negated operands next to complex operands
 * (c * -2, c + -1, c * -c, c ** -2).  The parser wraps negate(...) in a
 * conversion node to the complex dtype; the complex check rejected
 * conversion nodes, so these failed to compile, and optimize() folded a
 * conversion of a constant to complex into NaN.
 */

#include "../src/miniexpr.h"
#include "minctest.h"
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define N 5

static int failures = 0;

typedef double _Complex (*ref_fn)(double _Complex c);

static double _Complex r_mul_m2(double _Complex c) { return c * -2.0; }
static double _Complex r_add_m1(double _Complex c) { return c + -1.0; }
static double _Complex r_mul_neg(double _Complex c) { return c * -c; }
static double _Complex r_m2_mul(double _Complex c) { return -2.0 * c; }
static double _Complex r_div_m2(double _Complex c) { return c / -2.0; }
static double _Complex r_sub_m15(double _Complex c) { return c - -1.5; }
static double _Complex r_inc_mul_neg(double _Complex c) { return (c + 1.0) * -c; }
static double _Complex r_neg_neg(double _Complex c) { return -c * -c; }
static double _Complex r_inv(double _Complex c) { return 1.0 / c; }

static const struct {
    const char *expr;
    ref_fn ref;
} cases[] = {
    {"c * -2", r_mul_m2},
    {"c + -1", r_add_m1},
    {"c * -c", r_mul_neg},
    {"-2 * c", r_m2_mul},
    {"c / -2", r_div_m2},
    {"c - -1.5", r_sub_m15},
    {"(c + 1) * -c", r_inc_mul_neg},
    {"-c * -c", r_neg_neg},
    {"c ** -1", r_inv},
};

static void run(const char *expr, ref_fn ref, me_dtype dt, bool disable_simd) {
    const double _Complex in[N] = {1.0 + 1.0 * I, 4.0, -2.5 + 0.5 * I, 1.0 * I, 0.25 - 3.0 * I};
    float _Complex inf[N], outf[N];
    double _Complex outd[N];
    for (int i = 0; i < N; i++) inf[i] = (float _Complex)in[i];
    me_variable vars[] = {{"c", dt}};
    const void *ptrs[] = {dt == ME_COMPLEX64 ? (const void *)inf : (const void *)in};
    int err = 0;
    me_expr *e = NULL;
    if (me_compile(expr, vars, 1, dt, &err, &e) != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compile '%s' (dtype %d)\n", expr, dt);
        failures++;
        return;
    }
    me_eval_params params = ME_EVAL_PARAMS_DEFAULTS;
    params.disable_simd = disable_simd;
    if (me_eval(e, ptrs, 1, dt == ME_COMPLEX64 ? (void *)outf : (void *)outd, N, &params) != ME_EVAL_SUCCESS) {
        printf("  FAIL: eval '%s'\n", expr);
        failures++;
        me_free(e);
        return;
    }
    me_free(e);
    const double rtol = dt == ME_COMPLEX64 ? 1e-6 : 1e-15;
    for (int i = 0; i < N; i++) {
        const double _Complex want = ref(dt == ME_COMPLEX64 ? (double _Complex)inf[i] : in[i]);
        const double _Complex got = dt == ME_COMPLEX64 ? (double _Complex)outf[i] : outd[i];
        if (cabs(got - want) > rtol * cabs(want)) {
            printf("  FAIL: '%s' (dtype %d) [%d]: got (%.17g,%.17g), expected (%.17g,%.17g)\n", expr, dt, i,
                   creal(got), cimag(got), creal(want), cimag(want));
            failures++;
            return;
        }
    }
}

int main(void) {
    printf("Test: negative literals in complex expressions\n");
    static const me_dtype dtypes[] = {ME_COMPLEX64, ME_COMPLEX128};
    for (int d = 0; d < 2; d++) {
        for (int simd = 0; simd < 2; simd++) {
            for (size_t k = 0; k < sizeof(cases) / sizeof(cases[0]); k++) {
                run(cases[k].expr, cases[k].ref, dtypes[d], simd);
            }
        }
    }
    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("All complex negative-literal tests passed\n");
    return 0;
}
