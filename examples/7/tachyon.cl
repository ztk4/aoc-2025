// Helper for linearizing a 2D coord.
inline uint linearize2d(uint2 coord, uint2 size) {
  return coord.x + coord.y * size.x;
}

/// Build Vertical Lookup ///

// Given a manifold, builds a lookup set returning the global index of the next
// splitter in each column. Assumes the manifold is row-major.
// NB: GWS = LWS*size.x
kernel void build_vertical_lookup(global const char* manifold, uint2 size,
                                  global uint* lookup) {
  uint prev_splitter = UINT_MAX;
  // Consider chunks of LWS starting from the bottom of the column.
  uint nchunks = (size.y - 1) / get_local_size(0) + 1;
  for (int y = nchunks * get_local_size(0) - get_local_id(0) - 1; y >= 0;
       y -= get_local_size(0)) {
    uint idx = linearize2d((uint2)(get_group_id(0), y), size);
    // Check if there is a splitter here.
    bool splitter = y < size.y ? manifold[idx] == '^' : false;
    // Find the next y-coord of a splitter from here.
    uint next = work_group_scan_inclusive_min(splitter ? idx : prev_splitter);
    // Update the carry.
    prev_splitter = work_group_broadcast(next, get_local_size(0) - 1);
    // Record the result.
    if (y < size.y) lookup[idx] = next;
  }
}

/// Single-Group Propagate ///

// Uses a single work group to iteratively propagate a tachyon from S.
// Splitters that are used will be converted to '*'s.
// NOTE: This same algorithm could be modeled as a kernel that is iteratively
// called, but given the generally small width of the image, I felt using a
// single kernel with group-wide syncrhonization made more sense.
// NB: GWS = LWS, len(scratch) == len(manifold).
kernel void propagate(global const uint* lookup, uint2 size,
                      global char* manifold, local char* scratch) {
  // First, let's copy the manifold into local memory.
  for (uint idx = get_local_id(0); idx < size.x * size.y;
       idx += get_local_size(0)) {
    scratch[idx] = manifold[idx];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Now let's take iterative passes per row of the image.
  for (uint y = 0; y < size.y; ++y) {
    for (uint x = get_local_id(0); x < size.x; x += get_local_size(0)) {
      uint idx = linearize2d((uint2)(x, y), size);
      switch (scratch[idx]) {
        case 'S':
          scratch[lookup[idx]] = '*';
          break;
        case '*':
          // NOTE: This can race/contend, but always to the same value.
          // Might be performance implications of this though...
          if (x - 1 >= 0) {
            scratch[lookup[idx - 1]] = '*';
          }
          if (x + 1 < size.x) {
            scratch[lookup[idx + 1]] = '*';
          }
          break;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Now let's copy the result back to manifold.
  for (uint idx = get_local_id(0); idx < size.x * size.y;
       idx += get_local_size(0)) {
    manifold[idx] = scratch[idx];
  }
}

/// 2-Part GS-Count ///

// Counts the number of ocurrences of the given value.
// NB: GWS/LWS <= len(partials) < CL_DEVICE_MAX_WORK_GROUP_SIZE
kernel void count_partial_reduce(global const char* buffer, ulong size,
                                 char value, global ulong* partials) {
  ulong count = 0;
  for (ulong idx = get_global_id(0); idx < size; idx += get_global_size(0)) {
    if (buffer[idx] == value) ++count;
  }

  count = work_group_reduce_add(count);
  if (!get_local_id(0)) partials[get_group_id(0)] = count;
}

// Single-group sum reduce.
// NB: GWS = LWS = len(partials)
kernel void sum_full_reduce(global const ulong* partials,
                            global ulong* result) {
  ulong sum = work_group_reduce_add(partials[get_local_id(0)]);
  if (!get_local_id(0)) *result = sum;
}
