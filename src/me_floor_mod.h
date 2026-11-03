/*********************************************************************
  Copyright (c) 2026 Blosc Development Team
  License: BSD 3-Clause (see LICENSE.txt)
**********************************************************************/
#ifndef MINIEXPR_ME_FLOOR_MOD_H
#define MINIEXPR_ME_FLOOR_MOD_H

#include <math.h>
#include <stdint.h>
#include <string.h>

/* NumPy floor modulo, used by the `%` operator and remainder() (np.remainder,
 * np.mod, Python %): the result takes the sign of the divisor, so -7 % 3 == 2
 * and 7 % -3 == -2. fmod() keeps C fmod (np.fmod: the sign of the dividend).
 *
 * Floating point follows npy_remainder: fmod's exact result moves to the
 * divisor's side when the signs differ, and a zero result takes the divisor's
 * sign. x % 0, x % nan and inf % y are nan; finite x % inf is x when the signs
 * agree, else inf with the divisor's sign. Sign tests read the sign bits, not
 * ordered comparisons, so a NaN raises no spurious invalid flag.
 *
 * Integers: x % 0 is 0 and MIN % -1 is 0, as in NumPy; C would trap on both.
 * The JIT C generator (dsl_jit_cgen.c) emits a textual copy of the floating
 * rule; keep the two in step. */

/* Fixup of fmod's result r: add b where the signs differ (r != 0), give a
 * zero r the sign of b. Written with masks, not branches, so the tile loops
 * vectorize and random operand signs cost no mispredictions. r == 0 is a quiet
 * comparison (false for NaN), r + 0.0 is exact for nonzero r, and a NaN r
 * stays NaN. */
static inline double me_floor_mod_fixup_f64(double r, double b) {
    uint64_t ru, bu, out;
    memcpy(&ru, &r, sizeof ru);
    memcpy(&bu, &b, sizeof bu);
    const uint64_t zero = (uint64_t)0 - (uint64_t)(r == 0.0);
    const uint64_t flip = ((uint64_t)0 - ((ru ^ bu) >> 63)) & ~zero;
    const uint64_t addu = bu & flip;
    double add, moved;
    memcpy(&add, &addu, sizeof add);
    moved = r + add;
    memcpy(&out, &moved, sizeof out);
    out = (out & ~zero) | (bu & zero & (UINT64_C(1) << 63));
    memcpy(&moved, &out, sizeof moved);
    return moved;
}

static inline float me_floor_mod_fixup_f32(float r, float b) {
    uint32_t ru, bu, out;
    memcpy(&ru, &r, sizeof ru);
    memcpy(&bu, &b, sizeof bu);
    const uint32_t zero = (uint32_t)0 - (uint32_t)(r == 0.0f);
    const uint32_t flip = ((uint32_t)0 - ((ru ^ bu) >> 31)) & ~zero;
    const uint32_t addu = bu & flip;
    float add, moved;
    memcpy(&add, &addu, sizeof add);
    moved = r + add;
    memcpy(&out, &moved, sizeof out);
    out = (out & ~zero) | (bu & zero & UINT32_C(0x80000000));
    memcpy(&moved, &out, sizeof moved);
    return moved;
}

static inline double me_floor_mod_f64(double a, double b) {
    return me_floor_mod_fixup_f64(fmod(a, b), b);
}

static inline float me_floor_mod_f32(float a, float b) {
    return me_floor_mod_fixup_f32(fmodf(a, b), b);
}

/* r + b cannot overflow: r and b have opposite signs and |r| < |b|. */
#define ME_FLOOR_MOD_SIGNED(suffix, type) \
static inline type me_floor_mod_##suffix(type a, type b) { \
    if (b == 0 || b == -1) return 0; \
    type r = (type)(a % b); \
    return (r != 0 && ((r < 0) != (b < 0))) ? (type)(r + b) : r; \
}

#define ME_FLOOR_MOD_UNSIGNED(suffix, type) \
static inline type me_floor_mod_##suffix(type a, type b) { \
    return b ? (type)(a % b) : (type)0; \
}

ME_FLOOR_MOD_SIGNED(i8, int8_t)
ME_FLOOR_MOD_SIGNED(i16, int16_t)
ME_FLOOR_MOD_SIGNED(i32, int32_t)
ME_FLOOR_MOD_SIGNED(i64, int64_t)
ME_FLOOR_MOD_UNSIGNED(u8, uint8_t)
ME_FLOOR_MOD_UNSIGNED(u16, uint16_t)
ME_FLOOR_MOD_UNSIGNED(u32, uint32_t)
ME_FLOOR_MOD_UNSIGNED(u64, uint64_t)

#undef ME_FLOOR_MOD_SIGNED
#undef ME_FLOOR_MOD_UNSIGNED

#endif
