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
// NB: get_local_size(0) == get_global_size(0) == length(counts).
kernel void sum_reduce(global const long* counts, global long* result) {
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

  int hsize = size >> 1;
  // Whether the first set of k in size should merge ascending or descending.
  bool asc0 = !(get_group_id(0) & 1);
  for (int c = k >> 1; c > 0; c >>= 1) {
    // Ensure all writes to scratch are committed.
    barrier(CLK_LOCAL_MEM_FENCE);

    // We want indices that map to the lower halves of each pair of sequences.
    // E.g. c=2 -> 0, 1, 4, 5, 8, 9, ...; c=4 -> 0, 1, 2, 3, 8, 9, 10, 11, ...
    for (int base = get_local_id(0); base < hsize; base += get_local_size(0)) {
      // Maps `base` on [0, hsize) onto [0, size) as the lower halves.
      // In other words, the half of [0, size) with bit `c` unset.
      // Does this by always unsetting `c` when set and OR'ing in `hsize`.
      int idx = (base & ~c) | (base & c ? hsize : 0);
      bool asc = asc0 ^ !!(idx & k);  // Update with parity over sets of k.
      long4 pair = (long4)(scratch[idx], scratch[idx + c]);
      // This sorts pairs by (x, -y), i.e. x asc, y desc.
      bool cmp = pair.x == pair.z ? pair.y < pair.w : pair.x > pair.z;
      // printf("[%ld, %ld] ^%dv => [%ld, %ld]\n", pair.x, pair.z, asc,
      //        cmp ^ asc ? pair.x : pair.z,
      //        cmp ^ asc ? pair.z : pair.x);
      pair = cmp ^ asc ? pair.xyzw : pair.zwxy;
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

/// 3-Part Reduce and Propagate CumSum ///

// NOTE: While looking into if I can improve from 3 kernel dispatches (with 4n
// global memory accesses), I got completely nerd-sniped by
// https://escholarship.org/content/qt0bk9z4bt/qt0bk9z4bt.pdf.
// I am _so_ tempted to try and implement that, BUT there's quite a lot of
// messing about with global atomics and packing data in funny ways to leverage
// relaxed memory access -- maybe I'll try this another time.

// 1. Block Scan: Performs group-local cumsum scans on the *SECOND* element.
// NB:
//  * Choose LWS s.t. each thread can attain ~150 registers.
//  * Choose GWS s.t. there are at most CL_DEVICE_MAX_WORK_GROUP_SIZE groups,
//    so that part 2. can be done within a single group.
kernel void cumsum_block_partial_reduce(global long2* buffer, const int len) {
  // I'm being lazy here and having a maximum per-thread scan size in private
  // memory. On my hardware, this can still process very large inputs:
  //   * Each CU has 64K 32-bit registers, with a per-thread max of 255.
  //   * With 64*2 = 128 registers per thread -> 512 threads per CU.
  //   * Assuming at most 1024 threads in Part 3. we get:
  //   => Max # Elts: 1024 * 512 * 64 = 2^25 = 33554432.
  // ALSO for this application, this is so overkill, size.x will just be 1 XD
  long scratch[64];

  // Generally for index/size, we have (thread value, global value):
  int2 size = (int2)((len - 1) / get_global_size(0) + 1, len);
  if (size.x > 64) {
    if (!get_global_id(0)) printf("\nERROR: Too Big!\n");
    return;
  }

  // Thread-local scan of consecutive elements.
  long acc = 0;
  for (int2 idx = (int2)(0, get_global_id(0) * size.x); all(idx < size);
       ++idx) {
    acc += buffer[idx.y].y;  // Remember, scanning the second element only!
    scratch[idx.x] = acc;
  }

  // Workgroup-wide scan to produce the group-local spine.
  // NOTE: Exclusive means we get results [0, a, a+b, ...].
  long prefix = work_group_scan_exclusive_add(acc);

  // Thread-local update with workgroup-wide prefix + global write.
  for (int2 idx = (int2)(0, get_global_id(0) * size.x); all(idx < size);
       ++idx) {
    buffer[idx.y].y = prefix + scratch[idx.x];
  }
}

// 2. Full Reduce: Scans the group-local max values to produce the spine.
// This updates the last value in each block to be a fully-reduced, inclusive
// prefix sum.
// NB, where ' denotes parameters from Part 1.
//   * block_size = ceil(len / GWS') * LWS' (it's OK for this to exceed len)
//   * Choose GWS = LWS as (GWS'/LWS')
kernel void cumsum_full_reduce(global long2* buffer, const int len,
                               const int block_size) {
  int idx = (get_local_id(0) + 1) * block_size - 1;
  bool oob = idx >= len;
  // Scan over the group's max, or 0 if OOB.
  long prefix = work_group_scan_inclusive_add(oob ? 0 : buffer[idx].y);
  // Overwrite the value with this fully-reduced prefix.
  if (!oob) {
    buffer[idx].y = prefix;
  }
}

// 3. Propagate: Applies the inclusive prefix from the previous block.
// NB: Use the same GWS and LWS as Part 1. to preserve block size.
kernel void cumsum_propagate(global long2* buffer, const int len) {
  int block_size = ((len - 1) / get_global_size(0) + 1) * get_local_size(0);
  int block_idx = get_group_id(0) * block_size;
  if (!block_idx || block_idx >= len) return;  // No-op/OOB.

  // NOTE: the -1 prevents us from updating the last element, which is already
  // fully-reduced as a pre-condition to this kernel!
  int end = min(block_idx + block_size - 1, len);
  long prefix = buffer[block_idx - 1].y;
  for (int idx = block_idx + get_local_id(0); idx < end;
       idx += get_local_size(0)) {
    buffer[idx].y += prefix;
  }
}

/// 2-Part Sum Reduce -- Count interior ///

// 1. GS Partial Reduce: Reduces sorted half-intervals with prefix-sums into
// the sum of interior size per block.
// NB: len(partials) >= GWS/LWS, len(partials) < CL_DEVICE_MAX_WORK_GROUP_SIZE.
kernel void interior_sum_block_partial_reduce(global const long2* halfi,
                                              const int len,
                                              global long* partials) {
  long sum = 0;
  for (int idx = get_global_id(0); idx < len; idx += get_global_size(0)) {
    long2 el = halfi[idx];
    if (el.y > 0) {
      // This endpoint is interior to at least one interval.
      // NOTE: This will never be OOB given that halfi[-1].y MUST BE 0.
      sum += halfi[idx + 1].x - el.x;
    } else if (el.y == 0 && el.x > 0) {
      // The above sum treats intervals as half-open to avoid double-counting
      // fully interior endpoints. BUT we do need to account for one of the
      // endpoints adjacent to the exterior (choosing the end).
      // NOTE: el.x signifies a sentinel value leftover from bitonic sort.
      ++sum;
    }
  }

  sum = work_group_reduce_add(sum);
  if (!get_local_id(0)) {
    partials[get_group_id(0)] = sum;
  }
}

// 2. Single Group Reduce: Reduces a sum in a single work group.
// Use `sum_reduce` from above!
