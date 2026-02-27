/// 2-Part Sum Reduce -- Evaluates equations and then reduces ///

// Part 1: Evaluate equations and partially reduce results by summation.
// `equations` is flat buffer of `nprob` problems split over `lines`.
// Each `equations[n + p*nprob]` is part `p` of problem `n`.
// The last part of each equation should be the operation to reduce by.
// The partial reductions are written to `partials`.
// NB: GWS/LWS <= len(partials) < CL_DEVICE_MAX_WORK_GROUP_SIZE
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

/// Number Parsing -- Converts columnar numbers to ulongs ///

// Converts each column of chars into a single number.
// Empty columns convert to 0.
// NB: GWS >= llen.
kernel void parse_numbers(global const char* chars, ulong nlines, ulong llen,
                          global ulong* numbers) {
  if (get_global_id(0) > llen) return;

  ulong number = 0;
  for (ulong idx = get_global_id(0), l = 0; l < nlines; idx += llen, ++l) {
    char ch = chars[idx];
    if (ch >= '0' && ch <= '9') number = 10 * number + (ch - '0');
  }

  numbers[get_global_id(0)] = number;
}

/// 2-Part Sum Reduce -- Evaluates (columnar) equations and then reduces ///

// Part 1: Evalute equations and partially reduce results by summation.
// `numbers` is a flat buffer of numbers, with 0 separating equations.
// `ops` is a flat buffer of ops, where the idx corresponding to the first
// number in an equation denotes the operation to perform.
// The partial reductions are written to `partials`.
// NB: GWS/LWS <= len(partials) < CL_DEVICE_MAX_WORK_GROUP_SIZE
kernel void sum_col_equations_partial_reduce(global const ulong* numbers,
                                             global const char* ops, ulong len,
                                             global ulong* partials) {
  ulong sum = 0;
  for (ulong idx = get_global_id(0); idx < len; idx += get_global_size(0)) {
    bool add;
    switch (ops[idx]) {
      case '+':
        add = true;
        break;
      case '*':
        add = false;
        break;
      default:
        continue;
    }

    ulong result = add ? 0 : 1;
    for (ulong* num = numbers + idx; *num; ++num) {
      result = add ? result + *num : result * *num;
    }

    sum += result;
  }

  sum = work_group_reduce_add(sum);
  if (!get_local_id(0)) {
    partials[get_group_id(0)] = sum;
  }
}

// Part 2: Single-group sum reduce.
// See `sum_full_reduce` above.
