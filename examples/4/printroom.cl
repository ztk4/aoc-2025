// This sampler uses absolute coordinates, and returns a border color (zero) for
// OOBs.
// const sampler_t abs_border_sampler =
//    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

/*
kernel void find_accessible(read_only image2d_t map, int limit,
                            global long* accessible) {
  int2 dim = get_image_dim(map);
  int2 idx = (int2)(get_global_id(0), get_global_id(1));
  if (any(idx >= dim)) return;

  // Not a location with a paper roll -> no-op.
  if (!read_imagei(map, abs_border_sampler, idx).r) return;

  // naive 3x3 hollow box kernel
  long count = -1;  // offset by -1 to discount "self".
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      if (read_imagei(map, abs_border_sampler, idx + (int2)(dx, dy)).x) {
        ++count;
      }
    }
  }

  accessible[idx.y * dim.y + idx.x] = count < limit ? 1 : 0;
}

kernel void sum(global const long* input, ulong size, global long* output) {
  long value = get_global_id(0) < size ? input[get_global_id(0)] : 0;
  long sum = work_group_reduce_add(value);

  if (!get_local_id(0)) {
    output[get_group_id(0)] = sum;
  }
}
*/

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
        if (all(coord < dims) && map[linearize2d(coord, dims)] > 0) {
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

__attribute__((reqd_work_group_size(1, 1, 1))) kernel void find_all_accessible(
    ulong queue, global char* map, int2 dims, int limit, global char* buffer) {
  // printf("find_all_accessible\n");
  int status;
  printf("%ld\n", (queue_t)queue);

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
  status = enqueue_kernel((queue_t)queue, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, range,
                          0, NULL, &find_event, ^{
                            find_accessible(map, dims, limit, buffer);
                          });
  printf("Enqueue find_accessible: %d\n", status);

  // Schedule a small block to optionally iterate again afterwards.
  status = enqueue_kernel(
      (queue_t)queue, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange_1D(1), 1,
      &find_event, NULL, ^{
        // printf("\nChecking modified atomic.\n");
        if (atomic_load_explicit(&modified, memory_order_relaxed)) {
          printf("Scheduling iteration %d\n", ++it);
          //  NOTE: We swap buffer and map for the next iteration.
          enqueue_kernel((queue_t)queue, CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
                         ndrange_1D(1), ^{
                           find_all_accessible(queue, buffer, dims, limit, map);
                         });
        } else {
          printf("Not modified; exit.\n");
        }
      });
  printf("Enqueue find_all_accessible: %d\n", status);

  // Release our refcount on the event.
  release_event(find_event);
}

// 2-pass kernel using grid-stride.
// Expects: len(scratch) == num_groups.
// Ideal: num_groups should be a multiple of hardware's preferred group size.
kernel void count_accessible(global const char* map, int2 dims,
                             global long* result, global long* scratch) {
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

  if (!get_global_linear_id()) {
    int size = get_num_groups(0) * get_num_groups(1);
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
                   ndrange_1D(size, size), ^{
                     long count =
                         work_group_reduce_add(scratch[get_global_id(0)]);
                     if (!get_global_id(0)) {
                       *result = count;
                     }
                   });
  }
}
