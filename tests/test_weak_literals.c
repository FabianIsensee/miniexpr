/*
 * Number literals are weak scalars, as Python numbers are in NumPy (NEP 50):
 * they take the dtype of the array operand they meet (float32 + 0.1 computes
 * in float32, int8 + 1 in int8) unless their kind is higher (int32 < 2.5
 * compares in float64, bool + 1 is int64).  They used to be typed by the
 * output dtype (or the first variable): int32 < 2.5 compared with 2, a bool
 * output turned 2 into true, and int8 < 300 compared with 44.  A literal
 * outside every integer range of the other operand (int64 < 2**63) compares
 * exactly, and an out-of-range integer literal in arithmetic (uint8 + 300,
 * which NumPy 2 rejects) widens the computation instead of wrapping.
 */

#include "../src/miniexpr.h"
#include "minctest.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define N 1001

static int failures = 0;

static int32_t i32[N];
static int64_t i64[N];
static uint64_t u64[N];
static int8_t i8[N];
static uint8_t u8[N];
static bool b1[N];
static float f32[N];

static void fill(void) {
    for (int i = 0; i < N; i++) {
        i32[i] = i % 11 - 5;
        i8[i] = (int8_t)((i * 37) % 256 - 128);
        u8[i] = (uint8_t)(i % 256);
        b1[i] = (i % 3) == 0;
        /* Values around 0.1 that differ between float32 and float64. */
        f32[i] = (i % 4 == 0) ? 0.1f : (float)(0.05 + 0.0001 * (i % 1000));
        /* The extremes, which float64 rounds onto the out-of-range literals. */
        i64[i] = (i % 3 == 0) ? INT64_MAX : (i % 3 == 1) ? INT64_MIN : i;
        u64[i] = (i % 2 == 0) ? UINT64_MAX : (uint64_t)i;
    }
}

typedef struct {
    const char *name;
    me_dtype dtype;
    const void *data;
} var_spec;

static const var_spec all_vars[] = {
    {"i", ME_INT32, i32}, {"j", ME_INT8, i8}, {"u", ME_UINT8, u8}, {"p", ME_BOOL, b1}, {"f", ME_FLOAT32, f32},
    {"w", ME_INT64, i64}, {"v", ME_UINT64, u64},
};

/* Compiles with the single variable the expression uses. */
static me_expr *compile1(const char *src, const char *var, me_dtype out, const void **data) {
    for (size_t k = 0; k < sizeof(all_vars) / sizeof(all_vars[0]); k++) {
        if (!strcmp(all_vars[k].name, var)) {
            me_variable v[] = {{all_vars[k].name, all_vars[k].dtype}};
            int err = 0;
            me_expr *e = NULL;
            if (me_compile(src, v, 1, out, &err, &e) != ME_COMPILE_SUCCESS) {
                printf("  FAIL: compile '%s'\n", src);
                failures++;
                return NULL;
            }
            *data = all_vars[k].data;
            return e;
        }
    }
    return NULL;
}

static void check_bool(const char *src, const char *var, bool (*ref)(int)) {
    const void *data = NULL;
    me_expr *e = compile1(src, var, ME_BOOL, &data);
    if (!e) return;
    bool out[N];
    const void *ptrs[] = {data};
    ME_EVAL_CHECK(e, ptrs, 1, out, N);
    for (int i = 0; i < N; i++) {
        if (out[i] != ref(i)) {
            printf("  FAIL: '%s' [%d] = %d, expected %d\n", src, i, out[i], ref(i));
            failures++;
            break;
        }
    }
    me_free(e);
}

static void check_dtype(const char *src, const char *var, me_dtype expected) {
    const void *data = NULL;
    me_expr *e = compile1(src, var, ME_AUTO, &data);
    if (!e) return;
    if (me_get_dtype(e) != expected) {
        printf("  FAIL: '%s' infers dtype %d, expected %d\n", src, me_get_dtype(e), expected);
        failures++;
    }
    me_free(e);
}

static double load_as_double(const void *buf, me_dtype dt, int i) {
    switch (dt) {
    case ME_BOOL: return ((const bool *)buf)[i];
    case ME_INT8: return ((const int8_t *)buf)[i];
    case ME_INT16: return ((const int16_t *)buf)[i];
    case ME_INT32: return ((const int32_t *)buf)[i];
    case ME_INT64: return (double)((const int64_t *)buf)[i];
    case ME_UINT8: return ((const uint8_t *)buf)[i];
    case ME_UINT16: return ((const uint16_t *)buf)[i];
    case ME_FLOAT32: return ((const float *)buf)[i];
    case ME_FLOAT64: return ((const double *)buf)[i];
    default: return NAN;
    }
}

/* A DSL statement of literals alone takes the first input's dtype under
 * ME_AUTO (so `acc = 0` accumulates in it), as a weak literal meeting it:
 * `t = -1` with a uint16 first input is -1, not 65535, and so are the
 * temporaries of chained comparisons (-1 < c <= 2). */
static void check_dsl_literal_locals(void) {
    static const struct {
        const char *src;
        me_dtype first;
        double want;
        me_dtype want_dtype; /* ME_AUTO: any */
    } kcases[] = {
        {"def k(a, c):\n    t = -1\n    return t\n", ME_UINT16, -1, ME_AUTO},
        {"def k(a, c):\n    return -1 < c <= 2\n", ME_UINT16, 1, ME_BOOL},
        {"def k(a, c):\n    t = -1\n    return t < c\n", ME_UINT16, 1, ME_BOOL},
        {"def k(a, c):\n    y = 300\n    return y\n", ME_INT8, 300, ME_AUTO},
        {"def k(a, c):\n    y = 0.5\n    return y\n", ME_BOOL, 0.5, ME_FLOAT64},
        {"def k(a, c):\n    return 7\n", ME_BOOL, 7, ME_INT64},
        {"def k(a, c):\n    acc = 0\n    for i in range(3):\n        acc = acc + a\n    return acc\n", ME_FLOAT32, 4.5,
         ME_FLOAT32},
        {"def k(a, c):\n    t = 2\n    return a + t\n", ME_INT8, 3, ME_INT8},
    };
    for (size_t k = 0; k < sizeof(kcases) / sizeof(kcases[0]); k++) {
        /* a = 1.5 (float32), 1 (int8 after the cast) or True, c = 0.5 */
        float af[4] = {1.5f, 1.5f, 1.5f, 1.5f};
        int8_t ai8[4] = {1, 1, 1, 1};
        uint16_t au16[4] = {1, 1, 1, 1};
        bool ab[4] = {true, true, true, true};
        double c[4] = {0.5, 0.5, 0.5, 0.5};
        const me_dtype first = kcases[k].first;
        const void *a = first == ME_FLOAT32 ? (const void *)af : first == ME_INT8 ? (const void *)ai8
                        : first == ME_UINT16 ? (const void *)au16 : (const void *)ab;
        const double want = kcases[k].want;
        me_variable vars[] = {{"a", first}, {"c", ME_FLOAT64}};
        int64_t shape[] = {4};
        int32_t grid[] = {4};
        int err = 0;
        me_expr *e = NULL;
        if (me_compile_nd_jit(kcases[k].src, vars, 2, ME_AUTO, 1, shape, grid, grid, ME_JIT_OFF, &err, &e) !=
            ME_COMPILE_SUCCESS) {
            printf("  FAIL: compile DSL kernel %zu\n", k);
            failures++;
            continue;
        }
        double out[4] = {0};
        const void *ptrs[] = {a, c};
        ME_EVAL_CHECK(e, ptrs, 2, out, 4);
        const me_dtype dt = me_get_dtype(e);
        const double got = load_as_double(out, dt, 0);
        if (got != want || (kcases[k].want_dtype != ME_AUTO && dt != kcases[k].want_dtype)) {
            printf("  FAIL: DSL kernel %zu = %g (dtype %d), expected %g (dtype %d)\n", k, got, dt, want,
                   kcases[k].want_dtype);
            failures++;
        }
        me_free(e);
    }
}

static bool r_i_lt(int i) { return i32[i] < 2.5; }
static bool r_i_ge(int i) { return i32[i] >= 2.5; }
static bool r_i_eq(int i) { return i32[i] == 2.5; }
/* Literals outside the operand's range: the outcome is the same for all. */
static bool r_true(int i) { (void)i; return true; }
static bool r_false(int i) { (void)i; return false; }
static bool r_f_gt(int i) { return f32[i] > 0.1f; }
static bool r_w_is_max(int i) { return i64[i] == INT64_MAX; }
static bool r_i_lt_half5(int i) { return i32[i] < 2.5; }
static bool r_i_lt_half(int i) { return i32[i] < 0.5; }

int main(void) {
    printf("Test: number literals are weak scalars (NEP 50)\n");
    fill();

    /* Comparisons compute in the dtype NumPy uses. */
    check_bool("i < 2.5", "i", r_i_lt);
    check_bool("i >= 2.5", "i", r_i_ge);
    check_bool("i == 2.5", "i", r_i_eq);
    check_bool("j < 300", "j", r_true);
    check_bool("u > -1", "u", r_true);
    check_bool("u < 300", "u", r_true);
    check_bool("p < 2", "p", r_true);
    check_bool("p == 2", "p", r_false);
    /* No integer dtype holds both: compared exactly, not through float64. */
    check_bool("w < 9223372036854775808", "w", r_true);
    check_bool("9223372036854775808 > w", "w", r_true);
    check_bool("w >= -10000000000000000000", "w", r_true);
    /* strtod() reads INT64_MAX as 2**63: no fixed outcome for such a literal. */
    check_bool("w == 9223372036854775807", "w", r_w_is_max);
    check_bool("w == 9223372036854775808", "w", r_false);
    check_bool("v < 18446744073709551616", "v", r_true);
    check_bool("v > -1", "v", r_true);
    check_bool("v == -1", "v", r_false);
    /* float32 compares with 0.1 rounded to float32, as NumPy does. */
    check_bool("f > 0.1", "f", r_f_gt);
    /* Folded literal arithmetic stays a weak literal (5/2 is 2.5, 150*2 is
     * 300), also under a bool output. */
    check_bool("i < 5/2", "i", r_i_lt_half5);
    check_bool("i < 1/2", "i", r_i_lt_half);
    check_bool("j < 150*2", "j", r_true);
    check_bool("u > 0 - 1", "u", r_true);
    check_bool("i + 1 < 2**63", "i", r_true);

    /* where() with literal branches: an int64 output must not truncate the
     * literal of the condition. */
    {
        const void *data = NULL;
        me_expr *e = compile1("where(f > 0.1, 1, 0)", "f", ME_INT64, &data);
        if (e) {
            int64_t out[N];
            const void *ptrs[] = {data};
            ME_EVAL_CHECK(e, ptrs, 1, out, N);
            for (int i = 0; i < N; i++) {
                if (out[i] != (f32[i] > 0.1f)) {
                    printf("  FAIL: 'where(f > 0.1, 1, 0)' [%d] = %lld\n", i, (long long)out[i]);
                    failures++;
                    break;
                }
            }
            me_free(e);
        }
    }

    /* Inferred dtypes follow NumPy for array op literal. */
    check_dtype("j + 1", "j", ME_INT8);
    check_dtype("j + 1.5", "j", ME_FLOAT64);
    check_dtype("i * 2", "i", ME_INT32);
    check_dtype("i + 0.5", "i", ME_FLOAT64);
    check_dtype("f * 2", "f", ME_FLOAT32);
    check_dtype("f + 3.0", "f", ME_FLOAT32);
    check_dtype("f - -0.5", "f", ME_FLOAT32);
    check_dtype("p + 1", "p", ME_INT64);
    check_dtype("p * 1.5", "p", ME_FLOAT64);
    check_dtype("where(p, 1, 0)", "p", ME_INT64);
    check_dtype("where(p, 1.5, 0)", "p", ME_FLOAT64);
    /* Arithmetic of literals is a literal again (3**2 folds to a weak 9). */
    check_dtype("j + 3 ** 2", "j", ME_INT8);
    check_dtype("f + 2 * 1.5", "f", ME_FLOAT32);
    check_dtype("sqrt(f) + 3 ** 2", "f", ME_FLOAT32);
    check_dtype("i + 1/3", "i", ME_FLOAT64);
    /* A function of literals is a strong NumPy scalar (np.abs(-2) is int64,
     * np.sin(1.0) float64), not typed by the first variable; pi stays weak. */
    check_dtype("p + abs(-2)", "p", ME_INT64);
    check_dtype("j + abs(-200)", "j", ME_INT64);
    check_dtype("p + ceil(0.5)", "p", ME_FLOAT64);
    check_dtype("f * sin(1.0)", "f", ME_FLOAT64);
    check_dtype("f * pi", "f", ME_FLOAT32);
    check_dtype("i * pi", "i", ME_FLOAT64);
    check_dtype("j + e", "j", ME_FLOAT64);
    {
        const void *data = NULL;
        me_expr *e = compile1("i * pi / 180", "i", ME_FLOAT64, &data);
        if (e) {
            double out[N];
            const void *ptrs[] = {data};
            ME_EVAL_CHECK(e, ptrs, 1, out, N);
            for (int i = 0; i < N; i++) {
                if (fabs(out[i] - i32[i] * 3.14159265358979323846 / 180) > 1e-12) {
                    printf("  FAIL: 'i * pi / 180' [%d] = %g\n", i, out[i]);
                    failures++;
                    break;
                }
            }
            me_free(e);
        }
    }
    /* Bitwise operators of literals give a weak literal too, also under a
     * bool output, where the literals are typed bool at parse time. */
    check_dtype("j + (1 << 2)", "j", ME_INT8);
    check_bool("(j > 0) >= (1 << 2)", "j", r_false);
    check_bool("(j > 0) == 7 % 5", "j", r_false);
    check_dtype("j + 5 % 3", "j", ME_INT8);
    /* A negative power of an integer literal is a float (1 ** -1 is 1.0). */
    check_dtype("i + 1 ** -1", "i", ME_FLOAT64);
    /* An integer zero folded from literals has no sign (-0 and 8 % -4 are 0). */
    {
        const void *data = NULL;
        me_expr *e = compile1("f / -0 + f / (8 % -4)", "f", ME_FLOAT32, &data);
        if (e) {
            float out[N];
            const void *ptrs[] = {data};
            ME_EVAL_CHECK(e, ptrs, 1, out, N);
            for (int i = 0; i < N; i++) {
                if (!(out[i] > 0) || !isinf(out[i])) {
                    printf("  FAIL: 'f / -0 + f / (8 %% -4)' [%d] = %g, expected +inf\n", i, out[i]);
                    failures++;
                    break;
                }
            }
            me_free(e);
        }
    }
    {
        const void *data = NULL;
        me_expr *e = compile1("j + abs(-200)", "j", ME_AUTO, &data);
        if (e) {
            int64_t out[N];
            const void *ptrs[] = {data};
            ME_EVAL_CHECK(e, ptrs, 1, out, N);
            for (int i = 0; i < N; i++) {
                if (out[i] != i8[i] + 200) {
                    printf("  FAIL: 'j + abs(-200)' [%d] = %lld, expected %d\n", i, (long long)out[i], i8[i] + 200);
                    failures++;
                    break;
                }
            }
            me_free(e);
        }
    }
    check_dsl_literal_locals();
    /* NumPy 2 rejects these; miniexpr computes in a dtype that holds both
     * (as NumPy 1 did) instead of wrapping the literal. */
    check_dtype("u + 300", "u", ME_UINT16);
    check_dtype("j * 1000", "j", ME_INT16);
    check_dtype("i + 3000000000", "i", ME_INT64);
    {
        const void *data = NULL;
        me_expr *e = compile1("u + 300", "u", ME_AUTO, &data);
        if (e) {
            uint16_t out[N];
            const void *ptrs[] = {data};
            ME_EVAL_CHECK(e, ptrs, 1, out, N);
            for (int i = 0; i < N; i++) {
                if (out[i] != u8[i] + 300) {
                    printf("  FAIL: 'u + 300' [%d] = %u, expected %d\n", i, out[i], u8[i] + 300);
                    failures++;
                    break;
                }
            }
            me_free(e);
        }
    }

    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("All weak-literal tests passed\n");
    return 0;
}
