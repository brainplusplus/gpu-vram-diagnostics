// VRAM Diagnostics — Pattern Fill Kernel
// Fills a VRAM region with a 32-bit test pattern using grid-stride loop.

extern "C" __global__ void pattern_fill_u32(
    unsigned int* data,
    unsigned int pattern,
    unsigned long long num_elements
) {
    unsigned long long idx = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * (unsigned long long)blockDim.x;

    for (unsigned long long i = idx; i < num_elements; i += stride) {
        data[i] = pattern;
    }
}
