/* The DSL JIT emits expression text as C, and C groups two Python forms
 * differently: `not`/`!` is a tight unary operator in C (Python: `not a == b`
 * is `not (a == b)`), and C comparisons bind tighter than `&`, `^` and `|`
 * (Python: `a < b & c` is `a < (b & c)`). The IR text therefore makes
 * Python's grouping explicit, and JIT kernels agree with the interpreter.
 * Expected values were computed with Python on int64 scalars. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/dsl_jit_cgen.h"
#include "../src/dsl_jit_ir.h"
#include "../src/dsl_parser.h"
#include "../src/miniexpr.h"

#define N 6

static bool resolve_int64(void *ctx, const me_dsl_expr *expr, me_dsl_jit_ir_resolve_mode mode,
                          me_dtype *out_dtype) {
    (void)ctx;
    (void)expr;
    (void)mode;
    *out_dtype = ME_INT64;
    return true;
}

/* Returns the IR text of `return <expr>` (caller frees), or NULL. */
static char *ir_return_text(const char *expr, char **c_source) {
    char source[512];
    snprintf(source, sizeof(source), "def kernel(a, b, c):\n    return %s\n", expr);
    me_dsl_error error;
    me_dsl_program *program = me_dsl_parse(source, &error);
    if (!program) {
        printf("FAIL parse '%s': %s\n", expr, error.message);
        return NULL;
    }
    const char *names[] = {"a", "b", "c"};
    me_dtype dtypes[] = {ME_INT64, ME_INT64, ME_INT64};
    me_dsl_jit_ir_program *ir = NULL;
    bool ok = me_dsl_jit_ir_build(program, names, dtypes, 3, resolve_int64, NULL, &ir, &error);
    me_dsl_program_free(program);
    if (!ok || !ir || ir->block.nstmts != 1 || ir->block.stmts[0]->kind != ME_DSL_JIT_IR_STMT_RETURN) {
        printf("FAIL IR build '%s': %s\n", expr, ok ? "unexpected statements" : error.message);
        me_dsl_jit_ir_free(ir);
        return NULL;
    }
    const char *ir_text = ir->block.stmts[0]->as.return_stmt.expr.text;
    char *text = malloc(strlen(ir_text) + 1);
    if (text) {
        memcpy(text, ir_text, strlen(ir_text) + 1);
    }
    if (c_source && !me_dsl_jit_codegen_c(ir, ME_INT64, NULL, c_source, &error)) {
        printf("FAIL codegen '%s': %s\n", expr, error.message);
        free(text);
        text = NULL;
    }
    me_dsl_jit_ir_free(ir);
    return text;
}

static int check_ir_text(void) {
    static const struct {
        const char *expr;
        const char *grouped;
    } cases[] = {
        {"a < b & c", "a < (b & c)"},
        {"a & b == c", "(a & b) == c"},
        {"a ^ b != 0", "(a ^ b) != 0"},
        {"a == b | c", "a == (b | c)"},
        {"a | b < c & a", "(a | b) < (c & a)"},
        {"a != b ^ c", "a != (b ^ c)"},
        {"not a == b", "not (a == b)"},
        {"not a + b", "not (a + b)"},
        {"not not a == b", "not not (a == b)"},
        {"!a == b", "!(a == b)"},
        {"where(not a == b, a, c)", "where(not (a == b), a, c)"},
        {"a < b and not c", "a < b and not c"},
        /* Already unambiguous in C: copied unchanged. */
        {"(a < b) & (b < c)", "(a < b) & (b < c)"},
        {"not (a == b)", "not (a == b)"},
        {"a + b < c << 1", "a + b < c << 1"},
        {"where(a < 2, 1, 0) ^ c", "where(a < 2, 1, 0) ^ c"},
        {"a & b | c", "a & b | c"},
    };
    int failures = 0;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        char *text = ir_return_text(cases[i].expr, NULL);
        if (!text || strcmp(text, cases[i].grouped)) {
            printf("FAIL IR text for '%s': '%s', expected '%s'\n", cases[i].expr,
                   text ? text : "(null)", cases[i].grouped);
            failures++;
        }
        free(text);
    }
    /* The generated C carries the grouping. */
    char *c_source = NULL;
    char *text = ir_return_text("not a == b", &c_source);
    if (!text || !c_source || !strstr(c_source, "! (a == b)")) {
        printf("FAIL generated C for 'not a == b' lacks '! (a == b)'\n");
        failures++;
    }
    free(text);
    free(c_source);
    return failures;
}

static int evaluate(const char *expr, int jit, int64_t *out, bool *has_jit) {
    static const int64_t a[N] = {-3, -1, 0, 1, 2, 3};
    static const int64_t b[N] = {2, 3, 1, 2, 0, 1};
    static const int64_t c[N] = {1, 2, 0, 3, 1, 2};
    char source[512];
    snprintf(source, sizeof(source), "# me:compiler=tcc\ndef kernel(a, b, c):\n    return %s\n", expr);
    me_variable vars[] = {{.name = "a", .dtype = ME_INT64}, {.name = "b", .dtype = ME_INT64},
                          {.name = "c", .dtype = ME_INT64}};
    const void *inputs[] = {a, b, c};
    int64_t shape[] = {N};
    int32_t chunks[] = {N}, blocks[] = {N};
    int error = 0;
    me_expr *compiled = NULL;
    if (me_compile_nd_jit(source, vars, 3, ME_INT64, 1, shape, chunks, blocks, jit, &error,
                          &compiled) != ME_COMPILE_SUCCESS || !compiled) {
        printf("FAIL compile '%s' (at %d)\n", expr, error);
        return 1;
    }
    *has_jit = me_expr_has_jit_kernel(compiled);
    int rc = me_eval_nd(compiled, inputs, 3, out, N, 0, 0, NULL);
    me_free(compiled);
    if (rc != ME_EVAL_SUCCESS) {
        printf("FAIL eval '%s' (rc=%d)\n", expr, rc);
        return 1;
    }
    return 0;
}

static int check_values(void) {
    static const struct {
        const char *expr;
        int64_t expected[N];
    } cases[] = {
        {"a < b & c", {1, 1, 0, 1, 0, 0}},
        {"a & b == c", {0, 0, 1, 0, 0, 0}},
        {"a ^ b != 0", {1, 1, 1, 1, 1, 1}},
        {"a == b | c", {0, 0, 0, 0, 0, 1}},
        {"a | b < c & a", {1, 1, 0, 0, 0, 0}},
        {"a != b ^ c", {1, 1, 1, 0, 1, 0}},
        {"not a == b", {1, 1, 1, 1, 1, 1}},
        {"not not a == b", {0, 0, 0, 0, 0, 0}},
        {"where(not a == b, a, c)", {-3, -1, 0, 1, 2, 3}},
        /* A chain with a ~ operand (the chain lowering rejected ~). */
        {"-3 < ~a < 2", {0, 1, 1, 1, 0, 0}},
    };
    int64_t probe[N];
    bool available = false;
    if (evaluate("a + b", ME_JIT_ON, probe, &available)) {
        return 1;
    }
    if (!available) {
        printf("No DSL JIT backend; checking the interpreter only\n");
    }
    int failures = 0;
    for (int jit = ME_JIT_ON; jit <= ME_JIT_OFF; jit++) {
        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
            int64_t out[N] = {0};
            bool has_jit = false;
            if (evaluate(cases[i].expr, jit, out, &has_jit)) {
                failures++;
                continue;
            }
            if (jit == ME_JIT_ON && available && !has_jit) {
                printf("FAIL '%s' did not get a JIT kernel\n", cases[i].expr);
                failures++;
            }
            for (int k = 0; k < N; k++) {
                if (out[k] != cases[i].expected[k]) {
                    printf("FAIL '%s' (%s) at %d: %lld, expected %lld\n", cases[i].expr,
                           has_jit ? "JIT" : "interpreter", k, (long long)out[k],
                           (long long)cases[i].expected[k]);
                    failures++;
                    break;
                }
            }
        }
    }
    return failures;
}

int main(void) {
    int failures = check_ir_text() + check_values();
    if (failures) {
        printf("%d grouping checks failed\n", failures);
        return 1;
    }
    printf("All grouping checks passed\n");
    return 0;
}
