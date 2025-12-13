__kernel void sum(__global ulong* input, __global ulong* output,
                  __local ulong* scratch) {
  // Use each thread to copy it's data out of global memory.
  scratch[get_local_id(0)] = input[get_global_id(0)];

  // Barrier to ensure writes to scratch complete across the group.
  barrier(CLK_LOCAL_MEM_FENCE);

  // Have thread 0 do the reduce operation.
  if (!get_local_id(0)) {
    ulong res = 0;
    for (int i = 0; i < get_local_size(0); ++i) {
      res += scratch[i];
    }

    output[get_group_id(0)] = res;
  }
}
