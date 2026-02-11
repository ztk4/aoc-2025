/// 2-Part Sum Reduce -- Evaluates equations and then reduces ///

// Part 1: Evaluate equations and partially reduce results by summation.
// `equations` is flat buffer of `nprob` problems split over `lines`.
// Each `equations[n + p*nprob]` is part `p` of problem `n`.
// The last part of each equation should be the operation to reduce by.
// The partial reductions are written to `partials`.
// NB:
//   * GWS/LWS <= len(partials) < CL_DEVICE_MAX_WORK_GROUP_SIZE
kernel void sum_equations_partial_reduce(global const ulong* equations,
                                         ulong nprob, ulong lines,
                                         global ulong* partials) {
  if (!nprob || !lines) return;

  ulong sum = 0;
  for (ulong n = get_global_id(0); n < nprob; n += get_global_size(0)) {
    // NOTE: !add -> mul.
    bool add = equations[n + nprob * (lines - 1)] == (ulong)'+';
    ulong result = add ? 0 : 1;

    for (ulong idx = n, p = 0; p < lines - 1; idx += nprob, ++p) {
      result = add ? result + equations[idx] : result * equations[idx];
    }

    sum += result;
  }

  sum = work_group_reduce_add(sum);
  if (!get_local_id(0)) {
    partials[get_group_id(0)] = sum;
  }
}

// Part 2: Single-group sum reduce
// NB: LWS = GWS = len(partials)
kernel void sum_full_reduce(global const ulong* partials,
                            global ulong* result) {
  long res = work_group_reduce_add(partials[get_global_id(0)]);
  if (!get_global_id(0)) *result = res;
}
