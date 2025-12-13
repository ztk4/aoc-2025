__kernel void accumulate_local(__global uint* buffer, uint length, uint modulus,
                               __global uint* group_sums,
                               __local uint* scratch) {
  bool active = get_global_id(0) < length;
  // We read from scratch unconditionally, so best to initialize all values.
  scratch[get_local_id(0)] = active ? buffer[get_global_id(0)] : 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  // Perform accumulate locally.
  if (!get_local_id(0)) {
    uint cumsum = scratch[0];
    for (int i = 1; i < get_local_size(0); ++i) {
      cumsum = (scratch[i] += cumsum);
    }
    group_sums[get_group_id(0)] = scratch[get_local_size(0) - 1];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // Write back out to the global buffer, applying modulus.
  if (active) {
    buffer[get_global_id(0)] = scratch[get_local_id(0)] % modulus;
  }
}

__kernel void accumulate_global(__global uint* buffer, uint length,
                                uint modulus, __global const uint* group_sums,
                                __local uint* scratch) {
  // Get accumulate_local sums from all groups strictly less than this one.
  for (int i = get_local_id(0); i < get_group_id(0); i += get_local_size(0)) {
    scratch[i] = group_sums[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (!get_local_id(0)) {
    // Now let's sum these predecessor max sums.
    uint sum = 0;
    for (int i = 0; i < get_group_id(0); ++i) {
      sum += scratch[i];
    }
    scratch[0] = sum;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  // Finally, add the offset to each value.
  if (get_global_id(0) < length) {
    buffer[get_global_id(0)] =
        (buffer[get_global_id(0)] + scratch[0]) % modulus;
  }
}

__kernel void count(__global const uint* input, uint length, uint value,
                    __global uint* output, __local uint* scratch) {
  // Defaulting to ~value to a) avoid UB and b) avoid counting these as matches!
  scratch[get_local_id(0)] =
      get_global_id(0) < length ? input[get_global_id(0)] : ~value;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (!get_local_id(0)) {
    // Count matches and output.
    uint count = 0;
    for (int i = 0; i < get_local_size(0); ++i) {
      if (scratch[i] == value) ++count;
    }

    output[get_group_id(0)] = count;
  }
}

__kernel void sum(__global const uint* input, __global uint* output,
                  __local uint* scratch) {
  scratch[get_local_id(0)] = input[get_global_id(0)];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (!get_local_id(0)) {
    uint sum = 0;
    for (int i = 0; i < get_local_size(0); ++i) {
      sum += scratch[i];
    }

    output[get_group_id(0)] = sum;
  }
}
