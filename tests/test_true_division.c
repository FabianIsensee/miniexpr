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
static double ref_ldexp(double a, double b) { return ldexp(a, (int)b); }
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
    {"ldexp(a, b)", ref_ldexp},
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

/* Comparisons can compute in float64 while yielding bool. Binary float
 * math must retain its floating result when inference revisits the tree,
 * including inside reductions. Float16 is unsupported, so use float64. */
static void check_comparison_float_math(void) {
    static const binary_case cases[] = {
        {"(a > 0) / (b > 0)", ref_div},
        {"hypot(a > 0, b > 0)", hypot},
        {"atan2(a > 0, b > 0)", atan2},
        {"nextafter(a > 0, b > 0)", nextafter},
        {"copysign(a > 0, b > 0)", copysign},
        {"logaddexp(a > 0, b > 0)", ref_logaddexp},
        {"ldexp(a > 0, b > 0)", ref_ldexp},
    };
    double a[N], b[N];
    for (int i = 0; i < N; i++) {
        a[i] = i % 4 - 1;
        b[i] = 1;
    }
    const void *ptrs[] = {a, b};
    me_variable vars[] = {{"a", ME_FLOAT64}, {"b", ME_FLOAT64}};
    for (size_t k = 0; k < sizeof(cases) / sizeof(cases[0]); k++) {
        for (int arithmetic = 0; arithmetic < 2; arithmetic++) {
            for (int reduction = 0; reduction < 2; reduction++) {
                char expr[160];
                snprintf(expr, sizeof(expr), "%s(%s)%s%s", reduction ? "sum(" : "",
                         cases[k].expr, arithmetic ? " + 0.25" : "", reduction ? ")" : "");
                for (int explicit_output = 0; explicit_output < 2; explicit_output++) {
                    me_expr *e = NULL;
                    int err = 0;
                    me_dtype dtype = explicit_output ? ME_FLOAT64 : ME_AUTO;
                    if (me_compile(expr, vars, 2, dtype, &err, &e) != ME_COMPILE_SUCCESS ||
                        me_get_dtype(e) != ME_FLOAT64) {
                        printf("  FAIL: '%s' output %d does not compile to float64\n", expr, dtype);
                        failures++;
                        me_free(e);
                        continue;
                    }
                    double out[N];
                    ME_EVAL_CHECK(e, ptrs, 2, out, N);
                    double sum = 0;
                    for (int i = 0; i < N; i++) {
                        double want = cases[k].ref(a[i] > 0, b[i] > 0) + (arithmetic ? 0.25 : 0);
                        sum += want;
                        if (!reduction && !(out[i] == want || fabs(out[i] - want) <= 1e-14 * fabs(want))) {
                            printf("  FAIL: '%s' output %d [%d] = %.17g, expected %.17g\n",
                                   expr, dtype, i, out[i], want);
                            failures++;
                            break;
                        }
                    }
                    if (reduction && !(fabs(out[0] - sum) <= 1e-12 * fabs(sum))) {
                        printf("  FAIL: '%s' output %d = %.17g, expected %.17g\n", expr, dtype, out[0], sum);
                        failures++;
                    }
                    me_free(e);
                }
            }
        }
    }
}

int main(void) {
    check_comparison_float_math();
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
    /* Division of literals only, folded: 5 / 2 is 2.5, also as a DSL local. */
    {
        int err = 0;
        me_expr *e = NULL;
        double out[4] = {0};
        if (me_compile("5 / 2", NULL, 0, ME_AUTO, &err, &e) != ME_COMPILE_SUCCESS || me_get_dtype(e) != ME_FLOAT64) {
            printf("  FAIL: '5 / 2' does not compile to float64\n");
            failures++;
        }
        else {
            ME_EVAL_CHECK(e, NULL, 0, out, 4);
            if (out[0] != 2.5 || out[3] != 2.5) {
                printf("  FAIL: '5 / 2' = %g\n", out[0]);
                failures++;
            }
        }
        me_free(e);
        const int32_t x[4] = {1, 2, 3, 250};
        me_variable vars[] = {{"x", ME_INT32}};
        int64_t shape[] = {4};
        int32_t grid[] = {4};
        e = NULL;
        if (me_compile_nd_jit("def k(x):\n    y = 1 / 3\n    return x + y\n", vars, 1, ME_AUTO, 1, shape, grid,
                              grid, ME_JIT_OFF, &err, &e) != ME_COMPILE_SUCCESS ||
            me_get_dtype(e) != ME_FLOAT64) {
            printf("  FAIL: DSL x + (1 / 3) does not compile to float64\n");
            failures++;
        }
        else {
            const void *ptrs[] = {x};
            ME_EVAL_CHECK(e, ptrs, 1, out, 4);
            for (int i = 0; i < 4; i++) {
                if (fabs(out[i] - (x[i] + 1.0 / 3.0)) > 1e-12) {
                    printf("  FAIL: DSL x + (1 / 3) [%d] = %g\n", i, out[i]);
                    failures++;
                    break;
                }
            }
        }
        me_free(e);
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
