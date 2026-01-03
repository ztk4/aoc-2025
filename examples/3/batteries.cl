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
  //       Oddly, work_group_scan_exclusive_min(...) _does_ WAI.
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

__kernel void sum(__global ulong* const input, ulong len,
                  __global ulong* output) {
  ulong s = work_group_reduce_add(
      get_global_id(0) < len ? input[get_global_id(0)] : 0);
  if (!get_local_id(0)) {
    output[get_group_id(0)] = s;
  }
}
