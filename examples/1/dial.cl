// Computes buffer[i] as the buffer[i] % modulus over [0, modulus),
// and crossings[i] as abs(buffer[i] / modulus).
// Additionally, preserves the sign of buffer into is_nonneg.
__kernel void divmod_crossings(__global long* buffer, ulong length,
                               long modulus, __global long* crossings,
                               __global char* is_nonneg) {
  if (get_global_id(0) >= length) return;

  const long dividend = buffer[get_global_id(0)];
  buffer[get_global_id(0)] = dividend % modulus + (dividend < 0 ? modulus : 0);
  crossings[get_global_id(0)] = abs(dividend / modulus);
  is_nonneg[get_global_id(0)] = dividend >= 0;
}

// Produces a cumulative sum over buffer in chunks under modulus.
// Additionally writes the final sum of each group to group_sums.
__kernel void accumulate_local(__global long* buffer, ulong length,
                               long modulus, __global long* group_sums,
                               __local long* scratch) {
  bool active = get_global_id(0) < length;
  // We read from scratch unconditionally, so best to initialize all values.
  scratch[get_local_id(0)] = active ? buffer[get_global_id(0)] : 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  // Perform accumulate locally.
  if (!get_local_id(0)) {
    long cumsum = scratch[0];
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

// Completes a partial cumulative sum from accumulate_local.
// Each group calculates a prefix sum over group_sums, and adds this value to
// the existing sums in each group (also under modulus).
__kernel void accumulate_global(__global long* buffer, ulong length,
                                long modulus, __global const long* group_sums,
                                __local long* scratch) {
  // Get accumulate_local sums from all groups strictly less than this one.
  for (int i = get_local_id(0); i < get_group_id(0); i += get_local_size(0)) {
    scratch[i] = group_sums[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (!get_local_id(0)) {
    // Now let's sum these predecessor max sums.
    long sum = 0;
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

// Counts the number of ocurrences of value per group, writing to output.
__kernel void count(__global const long* input, ulong length, long value,
                    __global long* output, __local long* scratch) {
  // Defaulting to ~value to a) avoid UB and b) avoid counting these as matches!
  scratch[get_local_id(0)] =
      get_global_id(0) < length ? input[get_global_id(0)] : ~value;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (!get_local_id(0)) {
    // Count matches and output.
    long count = 0;
    for (int i = 0; i < get_local_size(0); ++i) {
      if (scratch[i] == value) ++count;
    }

    output[get_group_id(0)] = count;
  }
}

// Calculates the number of zero crossings as a result of each rotation.
// Expects that cumsum is the cumulative sum of rotations under modulus,
// and that crossings are the number of crossings a move causes regardless
// of the starting position. Writes to crossing the total number of crossings
// each move causes (considering position).
__kernel void get_zero_crossings(__global const long* cumsum, ulong length,
                                 long modulus, __global const char* is_nonneg,
                                 __global long* crossings) {
  // We look at the transition to the next cell => stop at length - 1.
  if (get_global_id(0) >= length - 1) return;

  // Analyzing the transition from value at [prev] -> [next].
  uint prev = get_global_id(0);
  uint next = prev + 1;
  long csprev = cumsum[prev];
  long csnext = cumsum[next];

  // true if this rotation was rightwards (or stationary), false if leftwards.
  bool rot_nonneg = is_nonneg[next];
  // true if this rotation increased the result.
  bool res_incr = csnext > csprev;
  // true if this rotation decreased the result.
  bool res_decr = csnext < csprev;

  // To avoid double counting, ignore cases where we start on 0.
  if (csprev) {
    // If result moved opposite to rot_nonneg, that indicates we crossed the 0
    // position. Landing on 0 is also a crossing!
    if (res_decr && rot_nonneg || res_incr && !rot_nonneg || !csnext) {
      ++crossings[next];
    }
  }
}

// Sums input in groups, writing to output.
__kernel void sum(__global const long* input, ulong length,
                  __global long* output, __local long* scratch) {
  scratch[get_local_id(0)] =
      get_global_id(0) < length ? input[get_global_id(0)] : 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (!get_local_id(0)) {
    long sum = 0;
    for (int i = 0; i < get_local_size(0); ++i) {
      sum += scratch[i];
    }

    output[get_group_id(0)] = sum;
  }
}
