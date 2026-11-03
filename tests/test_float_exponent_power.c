/*
 * An integer (or bool) base with a float exponent computes in float64, as
 * NumPy does (np.int32(2000) ** 3.0 is 8e9).  optimize() rewrites x ** 2.0 and
 * x ** 3.0 to x * x and x * x * x, and the products used to keep the integer
 * dtype: int32 2000 ** 3.0 wrapped to -589934592, also with a float64 output.
 */

#include "../src/miniexpr.h"
#include "minctest.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define N 6

static int failures = 0;

static void store(void *buf, me_dtype dt, int i, double v) {
    switch (dt) {
    case ME_BOOL: ((bool *)buf)[i] = v != 0; break;
    case ME_INT8: ((int8_t *)buf)[i] = (int8_t)v; break;
    case ME_INT16: ((int16_t *)buf)[i] = (int16_t)v; break;
    case ME_INT32: ((int32_t *)buf)[i] = (int32_t)v; break;
    case ME_INT64: ((int64_t *)buf)[i] = (int64_t)v; break;
    case ME_UINT8: ((uint8_t *)buf)[i] = (uint8_t)v; break;
    case ME_UINT32: ((uint32_t *)buf)[i] = (uint32_t)v; break;
    case ME_FLOAT32: ((float *)buf)[i] = (float)v; break;
    default: ((double *)buf)[i] = v; break;
    }
}

/* Values that overflow their own dtype when squared or cubed. */
static double input(me_dtype dt, int i) {
    static const double big[N] = {100, -3, 46341, 2000, 0, 1};
    switch (dt) {
    case ME_BOOL: return i % 2;
    case ME_INT8: return (double)(int8_t)(i ? -100 + 37 * i : 100);
    case ME_INT16: return i % 2 ? -300.0 * i : 1000.0 + i;
    case ME_UINT8: return 200 - 30 * i;
    case ME_UINT32: return 70000 + i;
    case ME_FLOAT32: return 1.5 * i - 2;
    default: return big[i];
    }
}

static double r_sq(double x) { return x * x; }
static double r_cube(double x) { return x * x * x; }
static double r_inc_sq(double x) { return (x + 1) * (x + 1); }

/* strong_exponent: the exponent is a folded constant, a strong float64 in NumPy
 * (np.sqrt(9.0)), so a float32 base gives float64 too. */
static void check(const char *src, me_dtype dt, me_dtype out, bool dsl, double (*ref)(double), bool strong_exponent) {
    char source[160];
    double in_storage[N];  /* aligned for every dtype stored into it */
    char *in = (char *)in_storage;
    double x[N];
    for (int i = 0; i < N; i++) {
        x[i] = input(dt, i);
        store(in, dt, i, x[i]);
    }
    me_variable vars[] = {{"x", dt}};
    const void *ptrs[] = {in};
    int err = 0;
    me_expr *e = NULL;
    int rc;
    if (dsl) {
        snprintf(source, sizeof(source), "def k(x):\n    return %s\n", src);
        int64_t shape[] = {N};
        int32_t grid[] = {N};
        rc = me_compile_nd_jit(source, vars, 1, out, 1, shape, grid, grid, ME_JIT_OFF, &err, &e);
    }
    else {
        rc = me_compile(src, vars, 1, out, &err, &e);
    }
    if (rc != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compile '%s' (dtype %d)\n", src, dt);
        failures++;
        return;
    }
    /* NumPy's result dtype: float32 stays float32 with a float literal,
     * everything else is float64. */
    const me_dtype want_dtype = dt == ME_FLOAT32 && !strong_exponent ? ME_FLOAT32 : ME_FLOAT64;
    if (me_get_dtype(e) != (out == ME_AUTO ? want_dtype : out)) {
        printf("  FAIL: '%s' (dtype %d) has dtype %d, expected %d\n", src, dt, me_get_dtype(e), want_dtype);
        failures++;
    }
    double od[N];
    float of[N];
    const bool f32 = me_get_dtype(e) == ME_FLOAT32;
    ME_EVAL_CHECK(e, ptrs, 1, f32 ? (void *)of : (void *)od, N);
    for (int i = 0; i < N; i++) {
        const double want = f32 ? (double)(float)ref((float)x[i]) : ref(x[i]);
        const double got = f32 ? (double)of[i] : od[i];
        if (got != want) {
            printf("  FAIL: '%s'%s (dtype %d) [%d] x=%g: got %.17g, expected %.17g\n", src, dsl ? " (DSL)" : "", dt,
                   i, x[i], got, want);
            failures++;
            break;
        }
    }
    me_free(e);
}

/* x ** abs(-2) is an integer power with a strong int64 exponent: int64 for
 * integer and bool bases, as in NumPy (np.abs(-2) is an int64 scalar), so
 * int8 100 ** abs(-2) is 10000, not a wrapped int8. */
static void check_int_exponent(me_dtype dt, bool dsl) {
    double in_storage[N];
    double x[N];
    for (int i = 0; i < N; i++) {
        x[i] = input(dt, i);
        store(in_storage, dt, i, x[i]);
    }
    me_variable vars[] = {{"x", dt}};
    const void *ptrs[] = {in_storage};
    int err = 0;
    me_expr *e = NULL;
    int rc;
    if (dsl) {
        int64_t shape[] = {N};
        int32_t grid[] = {N};
        rc = me_compile_nd_jit("def k(x):\n    return x ** abs(-2)\n", vars, 1, ME_AUTO, 1, shape, grid, grid,
                               ME_JIT_OFF, &err, &e);
    }
    else {
        rc = me_compile("x ** abs(-2)", vars, 1, ME_AUTO, &err, &e);
    }
    if (rc != ME_COMPILE_SUCCESS || me_get_dtype(e) != ME_INT64) {
        printf("  FAIL: x ** abs(-2)%s (dtype %d) has dtype %d, expected int64\n", dsl ? " (DSL)" : "", dt,
               e ? me_get_dtype(e) : -1);
        failures++;
        me_free(e);
        return;
    }
    int64_t out[N];
    ME_EVAL_CHECK(e, ptrs, 1, out, N);
    for (int i = 0; i < N; i++) {
        if ((double)out[i] != x[i] * x[i]) {
            printf("  FAIL: x ** abs(-2)%s (dtype %d) [%d] x=%g: got %lld\n", dsl ? " (DSL)" : "", dt, i, x[i],
                   (long long)out[i]);
            failures++;
            break;
        }
    }
    me_free(e);
}

int main(void) {
    printf("Test: integer ** float literal computes in float64\n");
    static const me_dtype dtypes[] = {ME_BOOL, ME_INT8, ME_INT16, ME_INT32, ME_INT64, ME_UINT8, ME_UINT32, ME_FLOAT32};
    for (size_t d = 0; d < sizeof(dtypes) / sizeof(dtypes[0]); d++) {
        for (int dsl = 0; dsl < 2; dsl++) {
            check("x ** 2.0", dtypes[d], ME_AUTO, dsl, r_sq, false);
            check("x ** 3.0", dtypes[d], ME_AUTO, dsl, r_cube, false);
            check("(x + 1) ** 2.0", dtypes[d], ME_AUTO, dsl, r_inc_sq, false);
            /* An exponent folded from a function is a strong float64. */
            check("x ** sqrt(9.0)", dtypes[d], ME_AUTO, dsl, r_cube, true);
            check("x ** floor(2.5)", dtypes[d], ME_AUTO, dsl, r_sq, true);
            if (dtypes[d] != ME_FLOAT32) {
                /* uint8 ** int64 is int64 only with NumPy's promotion of mixed
                 * signed/unsigned pairs, which is a separate PR. */
                if (dtypes[d] != ME_UINT8 && dtypes[d] != ME_UINT32) check_int_exponent(dtypes[d], dsl);
                check("x ** 2.0", dtypes[d], ME_FLOAT64, dsl, r_sq, false);
                check("x ** 3.0", dtypes[d], ME_FLOAT64, dsl, r_cube, false);
            }
        }
    }
    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("All float-exponent power tests passed\n");
    return 0;
}
