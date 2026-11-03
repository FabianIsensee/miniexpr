# Type Inference and Constants

## Important: How Constants Are Typed

Number literals in an expression are typed like Python numbers in NumPy
([NEP 50](https://numpy.org/neps/nep-0050-scalar-promotion.html)): they are
*weak* and take the dtype of the array operand they meet, unless their kind is
higher (bool < integer < floating point < complex).

### The Rule

| Operand dtype | `x + 2` | `x + 2.5` | `x < 2.5` computes in |
|---------------|---------|-----------|-----------------------|
| `ME_BOOL`     | int64   | float64   | float64               |
| `ME_INT8`     | int8    | float64   | float64               |
| `ME_INT32`    | int32   | float64   | float64               |
| `ME_UINT8`    | uint8   | float64   | float64               |
| `ME_FLOAT32`  | float32 | float32   | float32               |
| `ME_FLOAT64`  | float64 | float64   | float64               |

- An integer literal outside the range of an integer operand compares by value
  (`int8 < 300` is true, `uint8 > -1` is true), as in NumPy 2.
- Literals combined only with literals (`2 * 3.5`, `-1`) stay weak; a function
  of literals (`sin(1.0)`) is a float64 value, as NumPy's `np.sin(1.0)`.
- The output dtype does not change how literals or intermediates are typed: the
  expression is computed as NumPy would and only the result is converted.

## Example: FLOAT32 Variables with Constants

```c
// Input: float32 array
float data[] = {1.0f, 2.0f, 3.0f, 4.0f};

// Variable with explicit dtype
me_variable vars[] = {{"x", ME_FLOAT32}};

// Expression with constant
me_expr *expr = NULL;
if (me_compile("x + 3.0", vars, 1, ME_AUTO, &err, &expr) != ME_COMPILE_SUCCESS) { /* handle error */ }
// Result: Constants inferred as FLOAT32
// me_get_dtype(expr) returns ME_FLOAT32 ✓
```

### What Happens

1. You specify variable `x` as `ME_FLOAT32`
2. You use `ME_AUTO` for output
3. The weak literal `3.0` adopts `FLOAT32` from `x`
4. Result is `FLOAT32` (no unexpected promotion to FLOAT64)

## When This Matters

This is especially important for:

### 1. **Memory Efficiency with FLOAT32**

```c
// Working with single-precision arrays
float positions[1000000];  // 4MB instead of 8MB

me_variable vars[] = {{"pos", ME_FLOAT32}};
me_expr *expr = NULL;
if (me_compile("pos * 2.5 + 1.0", vars, 1, ME_AUTO, &err, &expr) != ME_COMPILE_SUCCESS) { /* handle error */ }
// Constants 2.5 and 1.0 are FLOAT32
// Result is FLOAT32 → saves memory!
```

### 2. **NumPy Compatibility**

```python
# Python/NumPy code
import numpy as np
data = np.array([1, 2, 3], dtype=np.float32)  # float32 array

# miniexpr with ME_AUTO matches this dtype
# Constants in "data + 3.0" are treated as float32
# ("np.arange(3, dtype=np.int32) + 0.5" is float64, and so is miniexpr's result)
```

### 3. **GPU/SIMD Optimization**

Many hardware accelerators work best with consistent types:

```c
// All FLOAT32 → can use SIMD instructions
float a[N], result[N];
me_variable vars[] = {{"a", ME_FLOAT32}};
me_expr *expr = NULL;
if (me_compile("sqrt(a*a + 2.5)", vars, 1, ME_AUTO, &err, &expr) != ME_COMPILE_SUCCESS) { /* handle error */ }
// Entire computation in FLOAT32 → faster on GPU/SIMD
```

## Explicit Type Control

If you want different behavior, specify the output dtype explicitly. There are two modes:

### Mode 1: All Variables ME_AUTO + Explicit Output

When all variables are `ME_AUTO` and you specify an output dtype, all variables use that type:

```c
// All variables use FLOAT64 (homogeneous)
me_variable vars[] = {{"x"}, {"y"}};  // Both ME_AUTO
me_expr *expr = NULL;
if (me_compile("x + y", vars, 2, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) { /* handle error */ }
// Result: FLOAT64 (all variables treated as FLOAT64)
```

### Mode 2: Explicit Variable Types + Explicit Output

When variables have explicit types and you specify an output dtype, variables keep their types during computation, and the result is cast to the output type:

```c
// Variables keep their types, result is cast to FLOAT64
me_variable vars[] = {{"x", ME_FLOAT32}, {"y", ME_FLOAT32}};
me_expr *expr = NULL;
if (me_compile("x + 3.0", vars, 2, ME_FLOAT64, &err, &expr) != ME_COMPILE_SUCCESS) { /* handle error */ }
// Computation: FLOAT32 + weak 3.0 → FLOAT32
// Output: Cast to FLOAT64
// Result: FLOAT64 (cast from FLOAT32 computation)
```

This is useful for:
- **Memory efficiency**: Compute in FLOAT32, output in FLOAT64 when needed
- **Heterogeneous inputs**: Mixed types (INT32 + FLOAT64) with specific output requirements
- **Type safety**: Explicit control over both input and output types

See `examples/08_explicit_output_dtype.c` for complete examples.

## Best Practices

### ✅ Recommended: Use ME_AUTO with Explicit Variable Dtypes

```c
me_variable vars[] = {
    {"temperature", ME_FLOAT32},
    {"pressure", ME_FLOAT32}
};
me_expr *expr = NULL;
if (me_compile("temperature * 1.8 + 32.0", vars, 1, ME_AUTO, &err, &expr) != ME_COMPILE_SUCCESS) { /* handle error */ }
// Constants match variable type → consistent FLOAT32
```

### ✅ For Mixed Types, Still Use ME_AUTO

```c
me_variable vars[] = {
    {"count", ME_INT32},
    {"price", ME_FLOAT64}
};
me_expr *expr = NULL;
if (me_compile("count * price * 1.08", vars, 2, ME_AUTO, &err, &expr) != ME_COMPILE_SUCCESS) { /* handle error */ }
// count * price is FLOAT64; the weak 1.08 keeps it FLOAT64
// Result: FLOAT64 ✓
```

### ❌ Avoid: Mixing Variable Types Without Explicit Dtypes

```c
// DON'T DO THIS:
me_variable vars[] = {{"x"}};  // No dtype specified!
me_expr *expr = NULL;
if (me_compile("x + 3.0", vars, 1, ME_AUTO, &err, &expr) != ME_COMPILE_SUCCESS) { /* handle error */ }
// Ambiguous: what type is x? what type is 3.0?
```

## Summary Table

| Scenario | Variable Dtype | Output Dtype | Computed In | Result Type |
|----------|---------------|--------------|-------------|-------------|
| Float32 + 3.0 | `ME_FLOAT32` | `ME_AUTO` | `FLOAT32` | `FLOAT32` ✓ |
| Float64 + 3.0 | `ME_FLOAT64` | `ME_AUTO` | `FLOAT64` | `FLOAT64` ✓ |
| Int32 + 3 | `ME_INT32` | `ME_AUTO` | `INT32` | `INT32` ✓ |
| Int32 + 0.5 | `ME_INT32` | `ME_AUTO` | `FLOAT64` | `FLOAT64` ✓ |
| Float32 + 3.0 | `ME_FLOAT32` | `ME_FLOAT64` | `FLOAT32` | `FLOAT64` ✓ (cast) |
| Mixed types | Both explicit | `ME_AUTO` | NumPy promotion | Promoted as needed ✓ |
| All ME_AUTO | All `ME_AUTO` | `ME_FLOAT64` | `FLOAT64` | `FLOAT64` ✓ (homogeneous) |
| Explicit vars | Both explicit | `ME_FLOAT32` | NumPy promotion | `FLOAT32` ✓ (cast) |

## See Also

- `examples/03_mixed_types.c` - Complete example with ME_AUTO
- `examples/08_explicit_output_dtype.c` - Explicit variable types + explicit output
- `doc/data-types.md` - Full type system documentation
- `tests/test_constant_type_inference.c` - Test validating this behavior
