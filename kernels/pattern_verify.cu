// VRAM Diagnostics — Pattern Verify Kernel
// Reads VRAM and compares against expected pattern.
// Counts total mismatches via atomicAdd.
// Logs first N error addresses into a ring buffer for chip identification.

#define MAX_ERROR_LOG 1024

struct ErrorEntry {
    unsigned long long byte_offset;  // Byte offset within the buffer
    unsigned int expected;
    unsigned int actual;
};

extern "C" __global__ void pattern_verify_u32(
    const unsigned int* data,
    unsigned int expected,
    unsigned long long num_elements,
    unsigned int* error_count,
    ErrorEntry* error_log,
    unsigned int* error_log_index
) {
    unsigned long long idx = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * (unsigned long long)blockDim.x;

    for (unsigned long long i = idx; i < num_elements; i += stride) {
        unsigned int val = data[i];
        if (val != expected) {
            unsigned int err_idx = atomicAdd(error_count, 1);

            // Log the first MAX_ERROR_LOG errors for chip identification
            unsigned int log_pos = atomicAdd(error_log_index, 1);
            if (log_pos < MAX_ERROR_LOG) {
                error_log[log_pos].byte_offset = i * sizeof(unsigned int);
                error_log[log_pos].expected = expected;
                error_log[log_pos].actual = val;
            }
        }
    }
}
