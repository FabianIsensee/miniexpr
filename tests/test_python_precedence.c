/* Operator precedence and associativity follow Python, which python-blosc2
 * relies on when it renders expressions without redundant parentheses.
 * Expected values were computed with Python (NumPy scalars), element-wise;
 * a comparison chain is Python's a < b and b < c per element. */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "../src/miniexpr.h"

#define N 6

static const double A[N] = {-3, -1, 0, 1, 2, 3};
static const double B[N] = {2, 3, 1, 2, 0, 1};
static const double C[N] = {1, 2, 0, 3, 1, 2};

typedef struct {
    me_dtype dtype;    /* operand dtype */
    const char *vars;  /* variables used, in order (the plain API binds exactly these) */
    const char *expr;
    double expected[N];
} precedence_case;

static const precedence_case cases[] = {
    /* Unary operators bind looser than "**" on their left ... */
    {ME_INT64, "a", "-a ** 2", {-9.0, -1.0, 0.0, -1.0, -4.0, -9.0}},
    {ME_FLOAT64, "a", "-a ** 2", {-9.0, -1.0, -0.0, -1.0, -4.0, -9.0}},
    {ME_INT64, "a", "~a ** 2", {-10.0, -2.0, -1.0, -2.0, -5.0, -10.0}},
    {ME_INT64, "a", "-2 ** 2 + a", {-7.0, -5.0, -4.0, -3.0, -2.0, -1.0}},
    {ME_FLOAT64, "a", "-2 ** 2 + a", {-7.0, -5.0, -4.0, -3.0, -2.0, -1.0}},
    {ME_FLOAT64, "a", "-a ** 0.5", {NAN, NAN, -0.0, -1.0, -1.4142135623730951, -1.7320508075688772}},
    {ME_INT64, "a", "2 * -a ** 2", {-18.0, -2.0, 0.0, -2.0, -8.0, -18.0}},
    {ME_FLOAT64, "a", "2 * -a ** 2", {-18.0, -2.0, -0.0, -2.0, -8.0, -18.0}},
    {ME_INT64, "ab", "a - -b ** 2", {1.0, 8.0, 1.0, 5.0, 2.0, 4.0}},
    {ME_FLOAT64, "ab", "a - -b ** 2", {1.0, 8.0, 1.0, 5.0, 2.0, 4.0}},
    {ME_INT64, "a", "- -a ** 2", {9.0, 1.0, 0.0, 1.0, 4.0, 9.0}},
    {ME_FLOAT64, "a", "- -a ** 2", {9.0, 1.0, 0.0, 1.0, 4.0, 9.0}},
    {ME_FLOAT64, "a", "-(a) ** 2", {-9.0, -1.0, -0.0, -1.0, -4.0, -9.0}},
    {ME_FLOAT64, "a", "-abs(a) ** 2", {-9.0, -1.0, -0.0, -1.0, -4.0, -9.0}},
    {ME_FLOAT64, "a", "-a ** 2 < 0", {1.0, 1.0, 0.0, 1.0, 1.0, 1.0}},
    /* ... but not on their right. */
    {ME_FLOAT64, "b", "2 ** -b", {0.25, 0.125, 0.5, 0.25, 1.0, 0.5}},
    {ME_FLOAT64, "ab", "-a ** -b", {-0.1111111111111111, 1.0, -INFINITY, -1.0, -1.0, -0.3333333333333333}},
    /* "**" is right-associative. */
    {ME_INT64, "abc", "a ** b ** c", {9.0, -1.0, 0.0, 1.0, 1.0, 3.0}},
    {ME_FLOAT64, "abc", "a ** b ** c", {9.0, -1.0, 0.0, 1.0, 1.0, 3.0}},
    {ME_FLOAT64, "abc", "a ** -b ** c", {0.1111111111111111, -1.0, INFINITY, 1.0, 1.0, 0.3333333333333333}},
    {ME_FLOAT64, "abc", "-a ** b ** c", {-9.0, 1.0, -0.0, -1.0, -1.0, -3.0}},
    /* Explicit grouping and function arguments are unchanged. */
    {ME_INT64, "a", "(-a) ** 2", {9.0, 1.0, 0.0, 1.0, 4.0, 9.0}},
    {ME_FLOAT64, "a", "(-a) ** 2", {9.0, 1.0, 0.0, 1.0, 4.0, 9.0}},
    {ME_INT64, "abc", "(a ** b) ** c", {9.0, 1.0, 1.0, 1.0, 1.0, 9.0}},
    {ME_FLOAT64, "abc", "(a ** b) ** c", {9.0, 1.0, 1.0, 1.0, 1.0, 9.0}},
    {ME_FLOAT64, "a", "abs(-a) ** 2", {9.0, 1.0, 0.0, 1.0, 4.0, 9.0}},
    /* Comparisons chain. */
    {ME_INT64, "abc", "a < b < c", {0.0, 0.0, 0.0, 1.0, 0.0, 0.0}},
    {ME_FLOAT64, "abc", "a < b < c", {0.0, 0.0, 0.0, 1.0, 0.0, 0.0}},
    {ME_INT64, "abc", "a < b == c", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
    {ME_INT64, "abc", "a == b < c", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
    {ME_INT64, "abc", "a <= b != c", {1.0, 1.0, 1.0, 1.0, 0.0, 0.0}},
    {ME_INT64, "abc", "a < b < c < 3", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
    {ME_FLOAT64, "abc", "a < b < c < 3", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
    {ME_INT64, "a", "-1 <= a < 2", {0.0, 1.0, 1.0, 1.0, 0.0, 0.0}},
    {ME_FLOAT64, "a", "-1 <= a < 2", {0.0, 1.0, 1.0, 1.0, 0.0, 0.0}},
    {ME_INT64, "abc", "(a < b) < c", {0.0, 1.0, 0.0, 1.0, 1.0, 1.0}},
    {ME_INT64, "abc", "not a < b < c", {1.0, 1.0, 1.0, 0.0, 1.0, 1.0}},
    {ME_INT64, "abc", "(a < b < c) + 1", {1.0, 1.0, 1.0, 2.0, 1.0, 1.0}},
    {ME_FLOAT64, "abc", "where(a < b < c, a, c)", {1.0, 2.0, 0.0, 1.0, 1.0, 2.0}},
    /* A function argument is a full expression, including not/and/or. */
    {ME_FLOAT64, "ab", "where(a > 0 and b > 0, a, b)", {2.0, 3.0, 1.0, 1.0, 0.0, 3.0}},
    {ME_INT64, "abc", "where(not a == b, a, c)", {-3.0, -1.0, 0.0, 1.0, 2.0, 3.0}},
    /* Levels that already matched Python. */
    {ME_INT64, "abc", "a < b & c", {1.0, 1.0, 0.0, 1.0, 0.0, 0.0}},
    {ME_INT64, "abc", "a & b == c", {0.0, 0.0, 1.0, 0.0, 0.0, 0.0}},
    {ME_INT64, "ab", "a ^ b != 0", {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}},
    {ME_INT64, "abc", "a == b | c", {0.0, 0.0, 0.0, 0.0, 0.0, 1.0}},
    {ME_INT64, "ab", "not a == b", {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}},
    {ME_INT64, "bc", "c << b + c", {8.0, 64.0, 0.0, 96.0, 2.0, 16.0}},
    {ME_INT64, "abc", "a & b ^ c | a", {-3.0, -1.0, 0.0, 3.0, 3.0, 3.0}},
    {ME_INT64, "abc", "a - b - c", {-6.0, -6.0, -1.0, -4.0, 1.0, 0.0}},
    {ME_FLOAT64, "abc", "a - b - c", {-6.0, -6.0, -1.0, -4.0, 1.0, 0.0}},
    {ME_INT64, "ab", "-a * b", {6.0, 3.0, 0.0, -2.0, 0.0, -3.0}},
    {ME_FLOAT64, "ab", "-a * b", {6.0, 3.0, -0.0, -2.0, -0.0, -3.0}},
};

static double output_value(me_dtype dtype, const void *out, int i) {
    switch (dtype) {
    case ME_BOOL: return ((const bool *)out)[i] ? 1.0 : 0.0;
    case ME_INT64: return (double)((const int64_t *)out)[i];
    case ME_FLOAT64: return ((const double *)out)[i];
    default: return NAN;
    }
}

/* mode 0: plain expression through me_compile/me_eval; mode 1: the same
 * expression returned from a DSL kernel (interpreter). */
static int check_case(const precedence_case *tc, int mode) {
    static const double *const columns[] = {A, B, C};
    static const char *const names[] = {"a", "b", "c"};
    int64_t ints[3][N];
    double floats[3][N];
    me_variable vars[3];
    const void *inputs[3];
    int nvars = 0;
    char source[256];
    for (int v = 0; v < 3; v++) {
        if (mode == 0 && !strchr(tc->vars, names[v][0])) {
            continue;
        }
        for (int i = 0; i < N; i++) {
            ints[v][i] = (int64_t)columns[v][i];
            floats[v][i] = columns[v][i];
        }
        vars[nvars] = (me_variable){.name = names[v], .dtype = tc->dtype};
        inputs[nvars] = tc->dtype == ME_INT64 ? (const void *)ints[v] : (const void *)floats[v];
        nvars++;
    }
    if (mode == 0) {
        snprintf(source, sizeof(source), "%s", tc->expr);
    }
    else {
        snprintf(source, sizeof(source), "def kernel(a, b, c):\n    return %s\n", tc->expr);
    }

    const char *mode_name = mode == 0 ? "expression" : "DSL";
    const char *dtype_name = tc->dtype == ME_INT64 ? "int64" : "float64";
    int64_t shape[] = {N};
    int32_t chunks[] = {N}, blocks[] = {N};
    int error = 0;
    me_expr *expr = NULL;
    int rc = mode == 0 ? me_compile(source, vars, nvars, ME_AUTO, &error, &expr) :
                         me_compile_nd_jit(source, vars, nvars, ME_AUTO, 1, shape, chunks, blocks,
                                           ME_JIT_OFF, &error, &expr);
    if (rc != ME_COMPILE_SUCCESS || !expr) {
        printf("FAIL %s %s: '%s' does not compile (rc=%d, at %d)\n", mode_name, dtype_name, tc->expr,
               rc, error);
        return 1;
    }
    me_dtype out_dtype = me_get_dtype(expr);
    union {
        bool b[N];
        int64_t i[N];
        double d[N];
    } out;
    memset(&out, 0, sizeof(out));
    rc = mode == 0 ? me_eval(expr, inputs, nvars, &out, N, NULL) :
                     me_eval_nd(expr, inputs, nvars, &out, N, 0, 0, NULL);
    me_free(expr);
    if (rc != ME_EVAL_SUCCESS) {
        printf("FAIL %s %s: '%s' does not evaluate (rc=%d)\n", mode_name, dtype_name, tc->expr, rc);
        return 1;
    }
    for (int i = 0; i < N; i++) {
        double got = output_value(out_dtype, &out, i);
        double want = tc->expected[i];
        if (!(got == want || (isnan(got) && isnan(want)))) {
            printf("FAIL %s %s: '%s' at a=%g b=%g c=%g: got %.17g, expected %.17g\n", mode_name,
                   dtype_name, tc->expr, A[i], B[i], C[i], got, want);
            return 1;
        }
    }
    return 0;
}

int main(void) {
    int failures = 0;
    int ncases = (int)(sizeof(cases) / sizeof(cases[0]));
    for (int mode = 0; mode < 2; mode++) {
        for (int c = 0; c < ncases; c++) {
            failures += check_case(&cases[c], mode);
        }
    }
    if (failures) {
        printf("%d of %d precedence checks failed\n", failures, 2 * ncases);
        return 1;
    }
    printf("All %d precedence checks passed\n", 2 * ncases);
    return 0;
}
