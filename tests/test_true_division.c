/*
 * True division and the real two-argument functions of integer and bool
 * operands compute in float64, as in NumPy (np.true_divide(int, int) and
 * np.hypot(int, int) are float64).  They used to run in the integer dtype
 * (int / int truncated, hypot(3, 4) of ints rounded) and bool / bool could
 * raise SIGFPE (integer 0 / 0).
 */

#include "../src/miniexpr.h"
#include "minctest.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 1031

static int failures = 0;

static size_t dtype_bytes(me_dtype dt) {
    switch (dt) {
    case ME_BOOL: case ME_INT8: case ME_UINT8: return 1;
    case ME_INT16: case ME_UINT16: return 2;
    case ME_INT32: case ME_UINT32: return 4;
    default: return 8;
    }
}

static void store(void *buf, me_dtype dt, int i, int v) {
    switch (dt) {
    case ME_BOOL: ((bool *)buf)[i] = v != 0; break;
    case ME_INT8: ((int8_t *)buf)[i] = (int8_t)v; break;
    case ME_INT16: ((int16_t *)buf)[i] = (int16_t)v; break;
    case ME_INT32: ((int32_t *)buf)[i] = (int32_t)v; break;
    case ME_INT64: ((int64_t *)buf)[i] = v; break;
    case ME_UINT8: ((uint8_t *)buf)[i] = (uint8_t)v; break;
    case ME_UINT16: ((uint16_t *)buf)[i] = (uint16_t)v; break;
    case ME_UINT32: ((uint32_t *)buf)[i] = (uint32_t)v; break;
    default: ((uint64_t *)buf)[i] = (uint64_t)v; break;
    }
}

static double ref_div(double a, double b) { return a / b; }
static double ref_sum_div(double a, double b) { return (a + b) / b; }
static double ref_div_add(double a, double b) { return a / b + 0.25; }
static double ref_div_mul(double a, double b) { return (a / b) * b; }
static double ref_half_mul(double a, double b) { return (a / 2) * b; }
static double ref_atan2_mul(double a, double b) { return atan2(a, b) * b; }
static double ref_hypot_gt(double a, double b) { return hypot(a, b) > 2; }
static double ref_logaddexp(double a, double b) {
    double m = fmax(a, b);
    return m + log1p(exp(-fabs(a - b)));
}

typedef struct {
    const char *expr;
    double (*ref)(double, double);
} binary_case;

static const binary_case binary_cases[] = {
    {"a / b", ref_div},
    {"hypot(a, b)", hypot},
    {"atan2(a, b)", atan2},
    {"nextafter(a, b)", nextafter},
    {"copysign(a, b)", copysign},
    {"logaddexp(a, b)", ref_logaddexp},
    /* The nodes above them are typed from their float result. */
    {"a / b + 0.25", ref_div_add},
    {"(a / b) * b", ref_div_mul},
    {"(a / 2) * b", ref_half_mul},
    {"atan2(a, b) * b", ref_atan2_mul},
};

/* A bool output checks a comparison of the float result. */
static void check(const char *expr, me_dtype dt, me_dtype out_dtype, double (*ref)(double, double)) {
    void *a = malloc(N * dtype_bytes(dt));
    void *b = malloc(N * dtype_bytes(dt));
    double av[N], bv[N];
    bool is_bool = dt == ME_BOOL;
    bool is_unsigned = dt == ME_UINT8 || dt == ME_UINT16 || dt == ME_UINT32 || dt == ME_UINT64;
    for (int i = 0; i < N; i++) {
        int x = is_bool ? (i % 3 != 0) : (is_unsigned ? i % 97 : i % 97 - 48);
        int y = is_bool ? (i % 2 != 0) : (is_unsigned ? 1 + i % 13 : (i % 2 ? 1 : -1) * (1 + i % 13));
        store(a, dt, i, x);
        store(b, dt, i, y);
        av[i] = x;
        bv[i] = y;
    }
    me_variable vars[] = {{"a", dt}, {"b", dt}};
    int err = 0;
    me_expr *e = NULL;
    if (me_compile(expr, vars, 2, out_dtype, &err, &e) != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compile '%s' (dtype %d)\n", expr, dt);
        failures++;
        goto done;
    }
    const me_dtype want_dtype = out_dtype == ME_BOOL ? ME_BOOL : ME_FLOAT64;
    if (me_get_dtype(e) != want_dtype) {
        printf("  FAIL: '%s' with dtype %d gives dtype %d, expected %d\n", expr, dt, me_get_dtype(e), want_dtype);
        failures++;
        me_free(e);
        goto done;
    }
    double out[N];
    const void *ptrs[] = {a, b};
    ME_EVAL_CHECK(e, ptrs, 2, out, N);
    for (int i = 0; i < N; i++) {
        double want = ref(av[i], bv[i]);
        double got = want_dtype == ME_BOOL ? ((const bool *)out)[i] : out[i];
        bool same = got == want || (isnan(want) && isnan(got)) || fabs(got - want) <= 1e-15 * fabs(want);
        if (!same) {
            printf("  FAIL: '%s' with dtype %d [%d] (%g, %g) = %.17g, expected %.17g\n", expr, dt, i,
                   av[i], bv[i], got, want);
            failures++;
            break;
        }
    }
    me_free(e);
done:
    free(a);
    free(b);
}

int main(void) {
    static const me_dtype dtypes[] = {ME_BOOL, ME_INT8, ME_INT16, ME_INT32, ME_INT64,
                                      ME_UINT8, ME_UINT16, ME_UINT32, ME_UINT64};
    printf("Test: true division and float functions of integer/bool operands\n");
    for (size_t k = 0; k < sizeof(binary_cases) / sizeof(binary_cases[0]); k++) {
        for (size_t d = 0; d < sizeof(dtypes) / sizeof(dtypes[0]); d++) {
            check(binary_cases[k].expr, dtypes[d], ME_AUTO, binary_cases[k].ref);
            check(binary_cases[k].expr, dtypes[d], ME_FLOAT64, binary_cases[k].ref);
        }
    }
    for (size_t d = 0; d < sizeof(dtypes) / sizeof(dtypes[0]); d++) {
        check("hypot(a, b) > 2", dtypes[d], ME_BOOL, ref_hypot_gt);
    }
    /* Nested: the division of an integer subexpression. */
    check("(a + b) / b", ME_INT64, ME_FLOAT64, ref_sum_div);
    check("(a + b) / b", ME_INT16, ME_AUTO, ref_sum_div);
    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("All true-division tests passed\n");
    return 0;
}
