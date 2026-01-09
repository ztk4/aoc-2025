// This sampler uses absolute coordinates, and returns a border color (zero) for
// OOBs.
const sampler_t abs_border_sampler =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

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
