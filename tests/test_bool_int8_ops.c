/*
 * NumPy has no bool loop for power, square, fmod and remainder: with bool
 * operands they compute in int8 (bool ** bool, np.square(bool), bool % bool),
 * and NumPy computes bool_array ** 2 as np.square(), also int8, while
 * bool ** 3 takes the int64 of a weak 3.  miniexpr gave these bool, and
 * optimize() turned bool ** 2 and bool ** 3 into bool products.
 */

#include "../src/miniexpr.h"
#include "minctest.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define N 8

static int failures = 0;

static bool p[N], q[N];
static float f[N];
static int8_t j[N];

static int64_t load(const void *buf, me_dtype dt, int i) {
    switch (dt) {
    case ME_BOOL: return ((const bool *)buf)[i];
    case ME_INT8: return ((const int8_t *)buf)[i];
    case ME_INT64: return ((const int64_t *)buf)[i];
    case ME_FLOAT64: return (int64_t)((const double *)buf)[i];
    default: return -999;
    }
}

static int64_t r_p(int i) { return p[i]; }
static int64_t r_p_pow_q(int i) { return p[i] || !q[i]; }
static int64_t r_zero(int i) { (void)i; return 0; }
static int64_t r_f_gt(int i) { return f[i] > 0.5f; }
static int64_t r_j_ge(int i) { return j[i] >= 1; }

/* names: the variables src uses; expected: the dtype NumPy gives; ref: the values. */
static void check(const char *src, const char *names, me_dtype expected, int64_t (*ref)(int)) {
    me_variable all[] = {{"p", ME_BOOL}, {"q", ME_BOOL}, {"f", ME_FLOAT32}, {"j", ME_INT8}};
    const void *data[] = {p, q, f, j};
    me_variable vars[4];
    const void *ptrs[4];
    int nvars = 0;
    for (int k = 0; k < 4; k++) {
        if (strchr(names, all[k].name[0])) {
            vars[nvars] = all[k];
            ptrs[nvars++] = data[k];
        }
    }
    for (int explicit_out = 0; explicit_out < 2; explicit_out++) {
        int err = 0;
        me_expr *e = NULL;
        if (me_compile(src, vars, nvars, explicit_out ? expected : ME_AUTO, &err, &e) != ME_COMPILE_SUCCESS) {
            printf("  FAIL: compile '%s'\n", src);
            failures++;
            return;
        }
        const me_dtype got_dtype = me_get_dtype(e);
        if (got_dtype != expected) {
            printf("  FAIL: '%s' (%s output) has dtype %d, expected %d\n", src, explicit_out ? "explicit" : "inferred",
                   got_dtype, expected);
            failures++;
            me_free(e);
            continue;
        }
        int64_t out[N];
        ME_EVAL_CHECK(e, ptrs, nvars, out, N);
        for (int i = 0; i < N; i++) {
            if (load(out, got_dtype, i) != ref(i)) {
                printf("  FAIL: '%s' [%d] = %lld, expected %lld\n", src, i, (long long)load(out, got_dtype, i),
                       (long long)ref(i));
                failures++;
                break;
            }
        }
        me_free(e);
    }
}

int main(void) {
    printf("Test: bool operands of power, square, fmod and remainder compute in int8\n");
    for (int i = 0; i < N; i++) {
        p[i] = i % 3 != 0;
        q[i] = i % 2 != 0;
        f[i] = 0.15f * (float)i;
        j[i] = (int8_t)(i % 4 - 1);
    }
    check("p ** 2", "p", ME_INT8, r_p);
    check("p ** 3", "p", ME_INT64, r_p);
    check("p ** 4", "p", ME_INT64, r_p);
    check("p ** q", "pq", ME_INT8, r_p_pow_q);
    check("p ** 2.0", "p", ME_FLOAT64, r_p);
    check("square(p)", "p", ME_INT8, r_p);
    check("(f > 0.5) ** 2", "f", ME_INT8, r_f_gt);
    check("(f > 0.5) ** 3", "f", ME_INT64, r_f_gt);
    check("square(f > 0.5)", "f", ME_INT8, r_f_gt);
    /* A comparison that computes in int8, the dtype of its parent. */
    check("square(j >= 1)", "j", ME_INT8, r_j_ge);
    check("(j >= 1) ** 2", "j", ME_INT8, r_j_ge);
    /* A divisor of True only: x % 0 is an integer-division edge case of its own. */
    for (int i = 0; i < N; i++) q[i] = true;
    check("p % q", "pq", ME_INT8, r_zero);
    check("fmod(p, q)", "pq", ME_INT8, r_zero);
    check("remainder(p, q)", "pq", ME_INT8, r_zero);
    /* Unchanged: bool loops exist for these. */
    check("p * p", "p", ME_BOOL, r_p);
    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("All bool int8 tests passed\n");
    return 0;
}
