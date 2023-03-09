---
parent: Memory access properties
grand_parent: Extended API
nav_order: 5
---

# `cuda::discard_memory`

```cuda
__device__ void discard_memory(void volatile* ptr, size_t nbytes);
```

**Preconditions**: `ptr` points to a valid allocation of size greater or equal to `nbytes`.

**Effects**: `discard_memory` is a weak memory operation that behaves like a write for the purpose of conflicting accesses. After executing it, a value must be written to the memory before reading from it; otherwise, the behavior is _undefined_.

**Note**: intended to avoid writing-back memory modifications to main memory, e.g., when using global memory as temporary scratch space.

# Example

This kernel needs a scratch pad that does not fit in shared memory, so it uses an allocation in global memory instead:

```cuda
#include <cuda/annotated_ptr>
__device__ int compute(int* scratch, size_t N);

__global__ void kernel(int const* in, int* out, int* scratch, size_t N) {
    // Each thread reads N elements into the scratch pad:
    for (int i = 0; i < N; ++i) {
        int idx = threadIdx.x + i * blockDim.x;
        scratch[idx] = in[idx];
    }
    __syncthreads();

    // All threads compute on the scratch pad:
    int result = compute(scratch, N);

    // All threads discard the scratch pad memory to _hint_ that it does not need to be flushed from the cache:
    cuda::discard_memory(scratch + threadIdx.x * N, N * sizeof(int));
    __syncthreads();

    out[threadIdx.x] = result;
}
```
