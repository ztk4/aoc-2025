/// 2-stage grid-stride kernel for checking IDs against ranges. ///

// Part 1: Stride kernel that counts inclusive IDs.
// `ids` is a lsit of `n_ids` ID values
// `ranges` is a list of `n_ranges` as inclusive (lower, upper)-bounds.
// `counts` will be each group's cummulative count of ID values that are on at
// least one given range.
// NB:
//   * get_work_dim() == 1
//   * length(counts) >= get_num_groups(0)
kernel void count_in_range_gs(global const long* ids, int n_ids,
                              global const long2* ranges, int n_ranges,
                              global long* counts) {
  long count = 0;
  for (int idx = get_group_id(0); idx < n_ids; idx += get_num_groups(0)) {
    long id = ids[idx];

    bool within = false;
    for (int ridx = get_local_id(0); ridx < n_ranges;
         ridx += get_local_size(0)) {
      long2 range = ranges[ridx];
      within = within || (id >= range.x && id <= range.y);
    }

    if (work_group_any(within)) ++count;
  }

  if (!get_local_id(0)) {
    counts[get_group_id(0)] = count;
  }
}

// Part 2: Single group sum reduce.
// NB: get_local_size(0) == length(counts).
kernel void reduce_counts(global const long* counts, global long* result) {
  long res = work_group_reduce_add(counts[get_local_id(0)]);
  if (!get_local_id(0)) *result = res;
}

/// Bitonic Merge Sort ///

// Merge: Takes a sequence of bitonic sequences of length k, and merges both
// halves of each into a monotonic sequence
// NOTE: The resulting sorted sequences will alternate in sort direction:
//       first increasing, then descreasing, etc.
// NOTE: Sequence will be sorted by x; y is assumed to be auxilliary data.
// NOTE: `size` gives the length of scratch := the number of values processed by
//       each work group.
// NB:
//  * get_local_size(0) MUST be a power of 2 AND <= (size/2).
//  * k MUST be a power of 2
//  * Each aligned, consecutive sequence of length k must ALREADY be bitonic.
//  * Scratch length `size` must be a multiple of k.
kernel void bitonic_merge(global long2* sequences, const int k,
                          local long2* scratch, const int size) {
  if (k <= 1) return;

  for (int2 idx = (int)get_local_id(0) + (int2)(0, get_group_id(0) * size);
       idx.x < size; idx += (int)get_local_size(0)) {
    scratch[idx.x] = sequences[idx.y];
  }

  // Whether the first set of k in size should merge ascending or descending.
  bool asc0 = !(get_group_id(0) & 1);
  for (int c = k >> 1; c > 0; c >>= 1) {
    // Ensure all writes to scratch are committed.
    barrier(CLK_LOCAL_MEM_FENCE);

    // We want indices that map to the lower halves of each pair of sequences.
    // E.g. c=2 -> 0, 1, 4, 5, 8, 9, ...; c=4 -> 0, 1, 2, 3, 8, 9, 10, 11, ...
    for (int base = get_local_id(0); base < (size >> 1);
         base += get_local_size(0)) {
      // Maps `base` on [0, size >> 1) onto [0, size) as the lower halves.
      // Does this by always unsetting `c`, and adding on `lws` in that case.
      int idx = (base & ~c) + (base & c ? get_local_size(0) : 0);
      bool asc = asc0 ^ !!(idx & k);  // Update with parity over sets of k.
      long4 pair = (long4)(scratch[idx], scratch[idx + c]);
      // printf("[%ld, %ld] ^%dv => [%ld, %ld]\n", pair.x, pair.z, asc,
      //        (pair.x > pair.z) ^ asc ? pair.x : pair.z,
      //        (pair.x > pair.z) ^ asc ? pair.z : pair.x);
      pair = (pair.x > pair.z) ^ asc ? pair.xyzw : pair.zwxy;
      scratch[idx] = pair.xy;
      scratch[idx + c] = pair.zw;
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  for (int2 idx = (int)get_local_id(0) + (int2)(0, get_group_id(0) * size);
       idx.x < size; idx += (int)get_local_size(0)) {
    sequences[idx.y] = scratch[idx.x];
  }
}
