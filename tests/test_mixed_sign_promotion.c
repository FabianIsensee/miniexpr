/*
 * Mixed signed/unsigned integer promotion follows NumPy: a signed type that
 * holds every value of the unsigned operand is the result (int16 + uint8 ->
 * int16, int64 + uint32 -> int64); only int64 + uint64 has no integer common
 * type and goes to float64.
 */

#include "../src/miniexpr.h"
#include "minctest.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define N 8

static int failures = 0;

static const char *dtype_name(me_dtype dt) {
    static const char *names[] = {"auto", "bool", "int8", "int16", "int32", "int64", "uint8", "uint16",
                                  "uint32", "uint64", "float32", "float64", "complex64", "complex128"};
    return (dt >= ME_AUTO && dt <= ME_COMPLEX128) ? names[dt] : "?";
}

/* NumPy's np.promote_types() for every signed/unsigned pair. */
static void test_inferred_dtypes(void) {
    static const me_dtype signed_types[] = {ME_INT8, ME_INT16, ME_INT32, ME_INT64};
    static const me_dtype unsigned_types[] = {ME_UINT8, ME_UINT16, ME_UINT32, ME_UINT64};
    static const me_dtype expected[4][4] = {
        /*            UINT8     UINT16    UINT32    UINT64 */
        /* INT8  */ {ME_INT16, ME_INT32, ME_INT64, ME_FLOAT64},
        /* INT16 */ {ME_INT16, ME_INT32, ME_INT64, ME_FLOAT64},
        /* INT32 */ {ME_INT32, ME_INT32, ME_INT64, ME_FLOAT64},
        /* INT64 */ {ME_INT64, ME_INT64, ME_INT64, ME_FLOAT64},
    };

    printf("Test: inferred dtype of signed + unsigned (both operand orders)\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int order = 0; order < 2; order++) {
                me_variable vars[] = {{"s", signed_types[i]}, {"u", unsigned_types[j]}};
                const char *src = order == 0 ? "s + u" : "u + s";
                int err = 0;
                me_expr *expr = NULL;
                if (me_compile(src, vars, 2, ME_AUTO, &err, &expr) != ME_COMPILE_SUCCESS) {
                    printf("  FAIL: compile %s (%s, %s)\n", src, dtype_name(signed_types[i]),
                           dtype_name(unsigned_types[j]));
                    failures++;
                    continue;
                }
                me_dtype got = me_get_dtype(expr);
                if (got != expected[i][j]) {
                    printf("  FAIL: %s with s=%s, u=%s -> %s, expected %s\n", src, dtype_name(signed_types[i]),
                           dtype_name(unsigned_types[j]), dtype_name(got), dtype_name(expected[i][j]));
                    failures++;
                }
                me_free(expr);
            }
        }
    }
}

/* int64 + uint32 must stay exact beyond 2**53 (a float64 detour rounds). */
static void test_int64_uint32_exact(void) {
    printf("Test: int64 + uint32 is exact beyond 2**53\n");
    int64_t a[N];
    uint32_t b[N];
    int64_t out[N];
    for (int i = 0; i < N; i++) {
        a[i] = ((int64_t)1 << 53) + 1 + 2 * i;
        b[i] = (uint32_t)(i % 3);
    }
    me_variable vars[] = {{"a", ME_INT64}, {"b", ME_UINT32}};
    for (int order = 0; order < 2; order++) {
        const char *src = order == 0 ? "a + b" : "b + a";
        int err = 0;
        me_expr *expr = NULL;
        if (me_compile(src, vars, 2, ME_INT64, &err, &expr) != ME_COMPILE_SUCCESS) {
            printf("  FAIL: compile %s\n", src);
            failures++;
            continue;
        }
        const void *ptrs[] = {a, b};
        memset(out, 0, sizeof(out));
        ME_EVAL_CHECK(expr, ptrs, 2, out, N);
        for (int i = 0; i < N; i++) {
            int64_t want = a[i] + (int64_t)b[i];
            if (out[i] != want) {
                printf("  FAIL: %s [%d] = %lld, expected %lld\n", src, i, (long long)out[i], (long long)want);
                failures++;
                break;
            }
        }
        me_free(expr);
    }
}

/* int16 - uint8 and int32 * uint16 compute in the signed type. */
static void test_small_mixed_values(void) {
    printf("Test: int16 - uint8 and int32 * uint16 values\n");
    int16_t s16[N];
    uint8_t u8[N];
    int16_t out16[N];
    int32_t s32[N];
    uint16_t u16[N];
    int32_t out32[N];
    for (int i = 0; i < N; i++) {
        s16[i] = (int16_t)(i * 1000 - 3000);
        u8[i] = (uint8_t)(250 - i);
        s32[i] = (int32_t)(i * 7 - 20);
        u16[i] = (uint16_t)(60000 + i);
    }

    me_variable vars16[] = {{"s", ME_INT16}, {"u", ME_UINT8}};
    int err = 0;
    me_expr *expr = NULL;
    if (me_compile("s - u", vars16, 2, ME_AUTO, &err, &expr) != ME_COMPILE_SUCCESS ||
        me_get_dtype(expr) != ME_INT16) {
        printf("  FAIL: int16 - uint8 should compile to int16\n");
        failures++;
    }
    else {
        const void *ptrs[] = {s16, u8};
        ME_EVAL_CHECK(expr, ptrs, 2, out16, N);
        for (int i = 0; i < N; i++) {
            if (out16[i] != (int16_t)(s16[i] - u8[i])) {
                printf("  FAIL: int16 - uint8 [%d] = %d, expected %d\n", i, out16[i], s16[i] - u8[i]);
                failures++;
                break;
            }
        }
    }
    me_free(expr);

    me_variable vars32[] = {{"s", ME_INT32}, {"u", ME_UINT16}};
    expr = NULL;
    if (me_compile("s * u", vars32, 2, ME_AUTO, &err, &expr) != ME_COMPILE_SUCCESS ||
        me_get_dtype(expr) != ME_INT32) {
        printf("  FAIL: int32 * uint16 should compile to int32\n");
        failures++;
    }
    else {
        const void *ptrs[] = {s32, u16};
        ME_EVAL_CHECK(expr, ptrs, 2, out32, N);
        for (int i = 0; i < N; i++) {
            if (out32[i] != s32[i] * (int32_t)u16[i]) {
                printf("  FAIL: int32 * uint16 [%d] = %d, expected %d\n", i, out32[i], s32[i] * (int32_t)u16[i]);
                failures++;
                break;
            }
        }
    }
    me_free(expr);
}

int main(void) {
    test_inferred_dtypes();
    test_int64_uint32_exact();
    test_small_mixed_values();
    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("All mixed-sign promotion tests passed\n");
    return 0;
}
