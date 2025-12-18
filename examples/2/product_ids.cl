// True if an ID is invalid.
bool invalid(ulong id) {
  ulong lo = 0, power = 1;
  for (; lo < id; power *= 10) {
    lo += (id % 10) * power;
    id /= 10;
  }

  // If lo < (the prior) power, then we are "using" a leading 0 of hi.
  return lo == id && lo >= (power / 10);
}

// Generates invalid IDs, and sums the results within a local group.
__kernel void sum_invalid(ulong start, ulong end, __global ulong* sums) {
  ulong id = start + get_global_id(0);
  ulong sum = work_group_reduce_add(id > end || !invalid(id) ? 0 : id);
  if (!get_local_id(0)) {
    sums[get_group_id(0)] = sum;
  }
}

// True if an ID is invalid considering chunks of n digits.
bool invalid_for_chunk(ulong id, size_t n) {
  ulong power = 1;
  while (n-- > 0) power *= 10;

  ulong chunk = id % power;
  // Invalid chunks can't have leading 0s.
  if (chunk < power / 10) return false;

  do {
    id /= power;
    if (id % power != chunk) return false;
  } while (id > power);

  return true;
}

// Generates invalid IDs per the extended definition.
// NOTE: I originally wanted to do a 2D kernel here (sum reduce on dim 0,
// boolean or reduce on dim 1), BUT that prevents me from using `work_group_any`
// which feels like a more efficient paradigm. In this case, that's at the cost
// of needing a much larger buffer for this step... but maybe that's OK?
__kernel void get_invalid_extended(ulong start, __global ulong* invalid) {
  ulong id = start + get_global_id(0);

  bool is_invalid = work_group_any(invalid_for_chunk(id, get_global_id(1)));
  if (!get_local_id(1) && is_invalid) {
    invalid[get_global_id(0)] = id;
  }
}

__kernel void sum(__global ulong* input, ulong length, __global ulong* sums) {
  ulong sum = work_group_reduce_add(
      get_global_id(0) < length ? input[get_global_id(0)] : 0);
  if (!get_local_id(0)) {
    sums[get_group_id(0)] = sum;
  }
}
