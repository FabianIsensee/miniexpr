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
static int r_p_plus_q(int i) { return p[i] + q[i]; }
static int r_q_minus_p(int i) { return q[i] - p[i]; }
static int r_two_plus_pq(int i) { return 2 + p[i] + q[i]; }

/* bool ** 2 and square() of bools are numbers, int8 in NumPy (np.square),
 * so adding them to a bool does not become a logical or. A comparison of
 * literals is a Python bool, so -(1 < 2) is -1 and (1 < 2) + (2 < 3) is 2. */
static void check_bool_powers(void) {
    static const struct {
        const char *src;
        int (*ref)(int);
    } cases[] = {
        {"p + q ** 2", r_p_plus_q},
        {"p ** 2 + q ** 2", r_p_plus_q},
        {"p + q ** 3", r_p_plus_q},
        {"square(p) + square(q)", r_p_plus_q},
        {"-(1 < 2) * p + q", r_q_minus_p},
        {"(1 < 2) + (2 < 3) + p + q", r_two_plus_pq},
    };
    me_variable vars[] = {{"p", ME_BOOL}, {"q", ME_BOOL}};
    const void *ptrs[] = {p, q};
    /* % of bools is int8 as well (np.remainder), so adding a bool to it is
     * not a logical or. */
    {
        int err = 0;
        me_expr *e = NULL;
        if (me_compile("p % q + p", vars, 2, ME_AUTO, &err, &e) != ME_COMPILE_SUCCESS || me_get_dtype(e) != ME_INT8) {
            printf("  FAIL: 'p %% q + p' infers dtype %d, expected int8\n", e ? me_get_dtype(e) : -1);
            failures++;
        }
        me_free(e);
        e = NULL;
        /* and so is ** of two bools (np.power). */
        if (me_compile("p ** q + p", vars, 2, ME_AUTO, &err, &e) != ME_COMPILE_SUCCESS || me_get_dtype(e) != ME_INT8) {
            printf("  FAIL: 'p ** q + p' infers dtype %d, expected int8\n", e ? me_get_dtype(e) : -1);
            failures++;
        }
        me_free(e);
    }
    for (size_t k = 0; k < sizeof(cases) / sizeof(cases[0]); k++) {
        for (int o = 0; o < 2; o++) {
            int err = 0;
            me_expr *e = NULL;
            if (me_compile(cases[k].src, vars, 2, o ? ME_AUTO : ME_INT8, &err, &e) != ME_COMPILE_SUCCESS) {
                printf("  FAIL: compile '%s'\n", cases[k].src);
                failures++;
                continue;
            }
            /* NumPy: int8 for ** 2 and square(), int64 for ** 3 and Python ints. */
            if (o && k < 2 && me_get_dtype(e) != ME_INT8) {
                printf("  FAIL: '%s' infers dtype %d, expected int8\n", cases[k].src, me_get_dtype(e));
                failures++;
            }
            int64_t out[N];
            ME_EVAL_CHECK(e, ptrs, 2, out, N);
            const me_dtype dt = me_get_dtype(e);
            for (int i = 0; i < N; i++) {
                const int64_t got = dt == ME_INT8 ? ((int8_t *)out)[i] : dt == ME_INT16 ? ((int16_t *)out)[i]
                                    : dt == ME_INT32 ? ((int32_t *)out)[i] : out[i];
                if (got != cases[k].ref(i)) {
                    printf("  FAIL: '%s' (output %d) [%d] = %lld, expected %d\n", cases[k].src, o ? ME_AUTO : ME_INT8,
                           i, (long long)got, cases[k].ref(i));
                    failures++;
                    break;
                }
            }
            me_free(e);
        }
    }
}

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

/* DSL kernels with an integer output: Python's True + True == 2. */
static void check_dsl_numeric_bools_int(void) {
    static const char *src = "def k(p, q):\n    return p + q\n";
    for (int jit = 1; jit <= 2; jit++) {
        me_variable vars[] = {{"p", ME_BOOL}, {"q", ME_BOOL}};
        int64_t shape[1] = {N};
        int32_t chunks[1] = {N}, blocks[1] = {N};
        int err = 0;
        me_expr *e = NULL;
        if (me_compile_nd_jit(src, vars, 2, ME_INT64, 1, shape, chunks, blocks, jit, &err, &e) !=
            ME_COMPILE_SUCCESS) {
            printf("  FAIL: compile DSL p + q (jit_mode %d)\n", jit);
            failures++;
            continue;
        }
        static int64_t out[N];
        const void *ptrs[] = {p, q};
        if (me_eval_nd(e, ptrs, 2, out, N, 0, 0, NULL) != ME_EVAL_SUCCESS) {
            printf("  FAIL: eval DSL p + q (jit_mode %d)\n", jit);
            failures++;
        }
        for (int i = 0; i < N; i++) {
            if (out[i] != (int64_t)p[i] + (int64_t)q[i]) {
                printf("  FAIL: DSL p + q (jit_mode %d) [%d] = %lld\n", jit, i, (long long)out[i]);
                failures++;
                break;
            }
        }
        me_free(e);
    }
}

/* A plain expression compiled with ME_JIT_ON keeps NumPy's bools, although
 * the JIT would wrap it in a DSL kernel. */
static void check_plain_with_jit_on(void) {
    me_variable vars[] = {{"p", ME_BOOL}, {"q", ME_BOOL}};
    int64_t shape[1] = {N};
    int32_t chunks[1] = {N}, blocks[1] = {N};
    int err = 0;
    me_expr *e = NULL;
    if (me_compile_nd_jit("(p + q) * 1.5", vars, 2, ME_FLOAT64, 1, shape, chunks, blocks, ME_JIT_ON, &err, &e) !=
        ME_COMPILE_SUCCESS) {
        printf("  FAIL: compile (p + q) * 1.5 with ME_JIT_ON\n");
        failures++;
        return;
    }
    static double out[N];
    const void *ptrs[] = {p, q};
    if (me_eval_nd(e, ptrs, 2, out, N, 0, 0, NULL) != ME_EVAL_SUCCESS) {
        printf("  FAIL: eval (p + q) * 1.5 with ME_JIT_ON\n");
        failures++;
    }
    for (int i = 0; i < N; i++) {
        if (out[i] != r_or_scaled(i)) {
            printf("  FAIL: (p + q) * 1.5 with ME_JIT_ON [%d] = %g, expected %g\n", i, out[i], r_or_scaled(i));
            failures++;
            break;
        }
    }
    me_free(e);
    e = NULL;
    if (me_compile_nd_jit("p + q", vars, 2, ME_AUTO, 1, shape, chunks, blocks, ME_JIT_ON, &err, &e) !=
            ME_COMPILE_SUCCESS ||
        me_get_dtype(e) != ME_BOOL) {
        printf("  FAIL: p + q with ME_JIT_ON is not bool\n");
        failures++;
    }
    me_free(e);
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
    check_bool_powers();
    check_dsl_numeric_bools();
    check_dsl_numeric_bools_int();
    check_plain_with_jit_on();
    check_literal_bitwise();
    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("All bool arithmetic tests passed\n");
    return 0;
}
