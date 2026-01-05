// Finds the largest possible 2-digit joltage per bank.
// EXPECT: lws == # batteries per bank
// EXPECT: gws == # batteries
__kernel void find_largest2(__global ulong* const batteries,
                            __global ulong* result) {
  // NOTE: Can use only 2 reduce passes calculating jmax_pos alongside jmax...
  //       BUT this requires rolloing our own (probably less efficient) reduce.
  ulong joltage = batteries[get_global_id(0)];
  // Max joltage in the bank (excluding the last element).
  ulong jmax = work_group_reduce_max(
      get_local_id(0) < get_local_size(0) - 1 ? joltage : 0);
  // Leftmost position of that joltage.
  // NOTE: The following does not work! Returns 0 every time.
  //       Oddly, work_group_scan_inclusive_min(...) _does_ WAI.
  // TODO: Investigate this.
  // work_group_reduce_min(joltage == jmax ? get_local_id(0)
  //                                       : get_local_size(0));
  size_t jmax_pos =
      get_local_size(0) -
      work_group_reduce_max(
          joltage == jmax ? get_local_size(0) - get_local_id(0) : 0);
  // Max subsequent joltage in the bank.
  ulong right_jmax =
      work_group_reduce_max(get_local_id(0) > jmax_pos ? joltage : 0);

  if (!get_local_id(0)) {
    result[get_group_id(0)] = jmax * 10 + right_jmax;
  }
  // DEBUG ONLY -- must comment out `const` above.
  // batteries[get_global_id(0)] = jmax_pos;
}

// Finds the max digit + position via RTL scan, constrained by position.
// For a digit n and a bank index i, result[n * gws(0) + i] = value where:
//   * value := (digit << 60) | position
//   * digit := the value of the max digit at or right of position,
//              excluding the rightmost n digits.
//   * position := index of the digit in the bank, INCREASING RTL.
//                 That makes the rightmost digit 0, increasing to the left.
// EXPECT: lws == (# batteries per bank, 1)
// EXPECT: gws == (# batteries, # digits to select max values for)
// EXPECT: input 0 <= batteries[i] < 10.
__kernel void find_maxn(__global ulong* const batteries,
                        __global ulong* result) {
  // Chosing value to use the digit as 4 MSB, and a leftward increasing bank
  // index as 60 LSB.  This will sort primarily by digit value, but prefer
  // leftward occurences to break ties.
  // NOTE: This could technically be calculated in a separate kernel to avoid
  // re-computing for each get_global_id(1) value... but that's cumbersome for
  // presumeably minimal gain.
  size_t idx = (get_group_id(0) + 1) * get_local_size(0) - get_local_id(0) - 1;
  // Default to 0 to exclude rightmost digits.
  ulong value = get_local_id(0) >= get_global_id(1)
                    ? (batteries[idx] << 60) | get_local_id(0)
                    : 0;

  result[get_global_id(1) * get_global_size(0) + idx] =
      work_group_scan_inclusive_max(value);
}

// Assembles the n-digit maximum joltage per bank from result of find_maxn.
__kernel void get_joltage(__global ulong* const digit_maxn, ulong nbanks,
                          ulong bank_size, ulong ndigits,
                          __global ulong* result) {
  if (get_global_id(0) >= nbanks) return;

  ulong joltage = 0;  // Accumulator for joltage value.
  size_t pos = 0;     // Position (lhs index) to look for next digit.
  int n = ndigits;    // The current digit to find.
  while (n--) {
    ulong value = digit_maxn[n * (nbanks * bank_size) +
                             get_global_id(0) * bank_size + pos];
    // The next pos should be 1 after the position this value points to.
    // Since we are 0-indexed, we can just avoid subtracting 1 here.
    pos = bank_size - (value & 0x0FFFFFFFFFFFFFFF);
    joltage = joltage * 10 + (value >> 60);  // Shift in digit value.
  }

  result[get_global_id(0)] = joltage;
}

__kernel void sum(__global ulong* const input, ulong len,
                  __global ulong* output) {
  ulong s = work_group_reduce_add(
      get_global_id(0) < len ? input[get_global_id(0)] : 0);
  if (!get_local_id(0)) {
    output[get_group_id(0)] = s;
  }
}
