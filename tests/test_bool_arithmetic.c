/*
 * In expressions, bool + bool is logical or and bool * bool is logical and
 * (NumPy), also when the result feeds a larger expression; DSL kernels keep
 * Python's numeric bools.  They used to be added/multiplied as
 * int8, which stored 2 in bool outputs and fed 2 into parents:
 * (p + q) * 1.5 gave 3.0 where NumPy gives 1.5.
 */

#include "../src/miniexpr.h"
#include "minctest.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define N 1027

static int failures = 0;
static bool p[N], q[N];
static int8_t s8[N];

static void fill(void) {
    for (int i = 0; i < N; i++) {
        p[i] = (i % 3) != 0;
        q[i] = (i % 5) < 2;
        s8[i] = (int8_t)(i % 7 - 3);
    }
}

/* Only the int8 operand `s` is optional; every expression uses p and q. */
static int nvars_of(const char *src) {
    return strchr(src, 's') ? 3 : 2;
}

static me_expr *compile(const char *src, me_dtype out) {
    me_variable vars[] = {{"p", ME_BOOL}, {"q", ME_BOOL}, {"s", ME_INT8}};
    int err = 0;
    me_expr *e = NULL;
    if (me_compile(src, vars, nvars_of(src), out, &err, &e) != ME_COMPILE_SUCCESS) {
        printf("  FAIL: compile '%s'\n", src);
        failures++;
        return NULL;
    }
    return e;
}

static void check_bool(const char *src, bool (*ref)(int)) {
    me_expr *e = compile(src, ME_BOOL);
    if (!e) return;
    uint8_t out[N];
    const void *ptrs[] = {p, q, s8};
    ME_EVAL_CHECK(e, ptrs, nvars_of(src), out, N);
    for (int i = 0; i < N; i++) {
        /* Compare bytes: a bool output must hold 0 or 1. */
        if (out[i] != (uint8_t)ref(i)) {
            printf("  FAIL: '%s' [%d] = %u, expected %d\n", src, i, out[i], ref(i));
            failures++;
            break;
        }
    }
    me_free(e);
}

static void check_f64(const char *src, double (*ref)(int)) {
    me_expr *e = compile(src, ME_FLOAT64);
    if (!e) return;
    double out[N];
    const void *ptrs[] = {p, q, s8};
    ME_EVAL_CHECK(e, ptrs, nvars_of(src), out, N);
    for (int i = 0; i < N; i++) {
        if (fabs(out[i] - ref(i)) > 1e-12) {
            printf("  FAIL: '%s' [%d] = %g, expected %g\n", src, i, out[i], ref(i));
            failures++;
            break;
        }
    }
    me_free(e);
}

static void check_i8(const char *src, int (*ref)(int)) {
    me_expr *e = compile(src, ME_INT8);
    if (!e) return;
    int8_t out[N];
    const void *ptrs[] = {p, q, s8};
    ME_EVAL_CHECK(e, ptrs, nvars_of(src), out, N);
    for (int i = 0; i < N; i++) {
        if (out[i] != ref(i)) {
            printf("  FAIL: '%s' [%d] = %d, expected %d\n", src, i, out[i], ref(i));
            failures++;
            break;
        }
    }
    me_free(e);
}

static bool r_or(int i) { return p[i] || q[i]; }
static bool r_and_or(int i) { return (p[i] && q[i]) || p[i]; }
static bool r_not_where(int i) { return !(p[i] ? q[i] : p[i]); }
static double r_or_scaled(int i) { return (p[i] || q[i]) * 1.5; }
static double r_and_scaled(int i) { return (p[i] && q[i]) * 2.5; }
static double r_hypot_or(int i) { return hypot((double)(p[i] || q[i]), 2.0); }
static int r_or_plus_int(int i) { return (p[i] || q[i]) + s8[i]; }

/* & and ~ of number literals are integer operators: s > 2 & 3 is s > 2 and
 * s >= ~1 is s >= -2. At parse time the literals were typed by the bool
 * output, which made them logical operators (s > 1, s >= 0). */
static void check_literal_bitwise(void) {
    static const struct {
        const char *src;
        int bound;
    } cases[] = {{"s > 2 & 3", 2}, {"s >= ~1", -2}, {"(~2) <= s", -3}};
    for (size_t k = 0; k < sizeof(cases) / sizeof(cases[0]); k++) {
        me_variable vars[] = {{"s", ME_INT8}};
        int err = 0;
        me_expr *e = NULL;
        if (me_compile(cases[k].src, vars, 1, ME_BOOL, &err, &e) != ME_COMPILE_SUCCESS) {
            printf("  FAIL: compile '%s'\n", cases[k].src);
            failures++;
            continue;
        }
        bool out[N];
        const void *ptrs[] = {s8};
        ME_EVAL_CHECK(e, ptrs, 1, out, N);
        for (int i = 0; i < N; i++) {
            const bool want = k == 0 ? s8[i] > cases[k].bound : s8[i] >= cases[k].bound;
            if (out[i] != want) {
                printf("  FAIL: '%s' [%d] (s = %d) = %d\n", cases[k].src, i, s8[i], out[i]);
                failures++;
                break;
            }
        }
        me_free(e);
    }
}

/* DSL kernels are per-element Python code and keep Python's numeric bools
 * (True + True == 2), with the JIT on and off alike. */
static void check_dsl_numeric_bools(void) {
    static const char *src = "def k(p, q):\n    return (p + q) * 1.5\n";
    for (int jit = 1; jit <= 2; jit++) {
        me_variable vars[] = {{"p", ME_BOOL}, {"q", ME_BOOL}};
        int64_t shape[1] = {N};
        int32_t chunks[1] = {N}, blocks[1] = {N};
        int err = 0;
        me_expr *e = NULL;
        if (me_compile_nd_jit(src, vars, 2, ME_FLOAT64, 1, shape, chunks, blocks, jit, &err, &e) !=
            ME_COMPILE_SUCCESS) {
            printf("  FAIL: compile DSL kernel (jit_mode %d)\n", jit);
            failures++;
            continue;
        }
        static double out[N];
        const void *ptrs[] = {p, q};
        if (me_eval_nd(e, ptrs, 2, out, N, 0, 0, NULL) != ME_EVAL_SUCCESS) {
            printf("  FAIL: eval DSL kernel (jit_mode %d)\n", jit);
            failures++;
        }
        for (int i = 0; i < N; i++) {
            if (out[i] != (p[i] + q[i]) * 1.5) {
                printf("  FAIL: DSL (p + q) * 1.5 (jit_mode %d) [%d] = %g\n", jit, i, out[i]);
                failures++;
                break;
            }
        }
        me_free(e);
    }
}

int main(void) {
    printf("Test: bool arithmetic follows NumPy (+ is or, * is and)\n");
    fill();
    check_bool("p + q", r_or);
    check_bool("p * q + p", r_and_or);
    check_bool("~where(p, q, p)", r_not_where);
    check_f64("(p + q) * 1.5", r_or_scaled);
    check_f64("(p * q) * 2.5", r_and_scaled);
    check_f64("hypot(p + q, 2.0)", r_hypot_or);
    check_i8("(p + q) + s", r_or_plus_int);
    check_dsl_numeric_bools();
    check_literal_bitwise();
    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("All bool arithmetic tests passed\n");
    return 0;
}
