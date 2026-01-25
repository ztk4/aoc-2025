inline size_t linearize2d(int2 idx, int2 dims) {
  return idx.x + idx.y * dims.x;
}

// Tracks whether any thread in find_accessible modified the map.
global atomic_int modified = ATOMIC_VAR_INIT(0);

kernel void find_accessible(global const char* map, int2 dims, int limit,
                            global char* buffer) {
  int2 idx = (int2)(get_global_id(0), get_global_id(1));
  bool oob = any(idx >= dims);

  char value = oob ? 0 : map[linearize2d(idx, dims)];
  long count = -1;
  if (value > 0) {
    // NOTE: The paper at idx will offset count's initial value of -1.
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        int2 coord = idx + (int2)(dx, dy);
        if (all(coord < dims && coord >= 0) &&
            map[linearize2d(coord, dims)] > 0) {
          ++count;
        }
      }
    }

    if (count < limit) value = -1;
  }

  // Write a copy of map to buffer, where value -> -1 for removed paper.
  if (!oob) buffer[linearize2d(idx, dims)] = value;

  // Let's do a work group reduce to check for modifications.
  bool removed = work_group_any(count >= 0 && count < limit);
  // NOTE: Checking the atomic first to avoid contended writes + cache
  // invalidation.
  if (!get_local_linear_id() && removed &&
      !atomic_load_explicit(&modified, memory_order_relaxed)) {
    // printf("Modified!\n");
    atomic_store_explicit(&modified, true, memory_order_relaxed);
  } else if (!get_local_linear_id()) {
    // printf("Not modified.\n");
  }
}

inline size_t round_to_multiple(size_t size, size_t multiple) {
  return ((size + multiple - 1) / multiple) * multiple;
}

global int it = 0;

// Iteratively calls `find_accessible` shuffling between map and buffer.
// If >0, `max_passes` limits the number of shuffles. The final buffer could be
// in either `map` or `buffer. `finished` will be set to 1 if either `map` or
// `buffer` cannot be further iterated.
__attribute__((reqd_work_group_size(1, 1, 1))) kernel void find_all_accessible(
    global char* map, int2 dims, int limit, global char* buffer, int max_passes,
    global int* finished) {
  // printf("find_all_accessible\n");
  int status;
  queue_t queue = get_default_queue();
  // printf("Q: %ld\n", (queue_t)queue);

  // Reset flag for this iteration.
  atomic_store_explicit(&modified, false, memory_order_relaxed);

  // Hard-coding in a preferred group size of 32x32...
  // I _could_ parameterize this from the host, but...
  // queue_t queue = get_default_queue();
  ndrange_t range = ndrange_2D(
      (size_t[]){round_to_multiple(dims.x, 32), round_to_multiple(dims.y, 32)},
      (size_t[]){32, 32});

  // Schedule a find pass over map + a tracking event.
  clk_event_t find_event;
  status = enqueue_kernel(queue, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, range, 0, NULL,
                          &find_event, ^{
                            find_accessible(map, dims, limit, buffer);
                          });
  // printf("Enqueue find_accessible: %d\n", status);

  // Schedule a small block to optionally iterate again afterwards.
  status = enqueue_kernel(
      queue, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange_1D(1), 1, &find_event, NULL,
      ^{
        // printf("\nChecking modified atomic.\n");
        if (atomic_load_explicit(&modified, memory_order_relaxed)) {
          if (max_passes != 0) {
            // printf("Scheduling iteration %d\n", ++it);
            //   NOTE: We swap buffer and map for the next iteration.
            enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
                           ndrange_1D(1), ^{
                             find_all_accessible(buffer, dims, limit, map,
                                                 max_passes - 1, finished);
                           });
          }
        } else {
          *finished = 1;
          // printf("Not modified; exit.\n");
        }
      });
  // printf("Enqueue find_all_accessible: %d\n", status);

  // Release our refcount on the event.
  release_event(find_event);
}

kernel void count_accessible2(global const long* scratch, global long* result) {
  // printf("Count Pass 2\n");
  long count = work_group_reduce_add(scratch[get_global_id(0)]);
  if (!get_global_id(0)) {
    *result = count;
  }
}

// 2-pass kernel using grid-stride.
// Expects: len(scratch) == num_groups.
// Ideal: num_groups should be a multiple of hardware's preferred group size.
kernel void count_accessible(global const char* map, int2 dims,
                             global long* result, global long* scratch) {
  // printf("Count Accessible\n");
  int2 idx;

  long count = 0;
  for (idx.x = get_global_id(0); idx.x < dims.x; idx.x += get_global_size(0)) {
    for (idx.y = get_global_id(1); idx.y < dims.y;
         idx.y += get_global_size(1)) {
      if (map[linearize2d(idx, dims)] == -1) {
        ++count;
      }
    }
  }

  count = work_group_reduce_add(count);
  if (!get_local_linear_id()) {
    scratch[linearize2d((int2)(get_group_id(0), get_group_id(1)),
                        (int2)(get_num_groups(0), get_num_groups(1)))] = count;
  }

  // Sadly, this enqueue_kernel just doesn't ever schedule and I can't figure
  // out why. Obviously this is unsupported... but weird that it works in
  // find_all... Changing to NO_WAIT works... but I need kernel-wide
  // synchronization. Moving this to host enqueued :(
  /*
if (!get_global_linear_id()) {
int size = get_num_groups(0) * get_num_groups(1);
printf("size: %d\n", size);
int status = enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
                          ndrange_1D(size, size), ^{
                            count_accessible2(scratch, result);
                          });
printf("Enqueue Count Pass 2: %d\n", status);
}
  */
}
