/// 2-stage grid-stride kernel for checking IDs against ranges. ///

// Part 1: Stride kernel that counts inclusive IDs.
// `ids` is a lsit of `n_ids` ID values
// `lo` and `hi` are lists of `n_ranges` lower- and upper-bounds respectively.
// `counts` will be each group's cummulative count of ID values that are on at
// least one given range.
// NB:
//   * get_work_dim() == 1
//   * length(counts) >= get_num_groups(0)
kernel void count_in_range_gs(global const ulong* ids, int n_ids,
                              global const ulong* lo, global const ulong* hi,
                              int n_ranges, global ulong* counts) {
  ulong count = 0;
  for (int idx = get_group_id(0); idx < n_ids; idx += get_num_groups(0)) {
    ulong id = ids[idx];

    bool within = false;
    for (int ridx = get_local_id(0); ridx < n_ranges;
         ridx += get_local_size(0)) {
      within = within || (id >= lo[ridx] && id <= hi[ridx]);
    }

    if (work_group_any(within)) ++count;
  }

  if (!get_local_id(0)) {
    counts[get_group_id(0)] = count;
  }
}

// Part 2: Single group sum reduce.
// NB: get_local_size(0) == length(counts).
kernel void reduce_counts(global const ulong* counts, global ulong* result) {
  ulong res = work_group_reduce_add(counts[get_local_id(0)]);
  if (!get_local_id(0)) *result = res;
}
