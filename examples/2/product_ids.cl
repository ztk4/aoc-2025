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
__kernel void sum_invalid(ulong start, ulong end, __global ulong* sums,
                          __local ulong* scratch) {
  ulong id = start + get_global_id(0);
  scratch[get_local_id(0)] = id > end || !invalid(id) ? 0 : id;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (!get_local_id(0)) {
    ulong sum = 0;
    for (int i = 0; i < get_local_size(0); ++i) {
      sum += scratch[i];
    }
    sums[get_group_id(0)] = sum;
  }
}

__kernel void sum(__global ulong* input, ulong length, __global ulong* sums,
                  __local ulong* scratch) {
  scratch[get_local_id(0)] =
      get_global_id(0) < length ? input[get_global_id(0)] : 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (!get_local_id(0)) {
    ulong sum = 0;
    for (int i = 0; i < get_local_size(0); ++i) {
      sum += scratch[i];
    }
    sums[get_group_id(0)] = sum;
  }
}
