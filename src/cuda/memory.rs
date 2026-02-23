/// RAII wrapper for device (GPU) memory allocations.
use super::ffi::{self, CUdeviceptr, CudaResult};

pub struct DeviceBuffer {
    pub ptr: CUdeviceptr,
    pub size: usize,
}

impl DeviceBuffer {
    /// Allocate `size` bytes on the current GPU.
    pub fn alloc(size: usize) -> CudaResult<Self> {
        let ptr = ffi::cuda_mem_alloc(size)?;
        Ok(DeviceBuffer { ptr, size })
    }

    /// Copy data from device to host.
    pub fn copy_to_host(&self, dst: &mut [u8]) -> CudaResult<()> {
        assert!(
            dst.len() <= self.size,
            "Host buffer ({}) larger than device buffer ({})",
            dst.len(),
            self.size
        );
        ffi::cuda_memcpy_dtoh(dst, self.ptr)
    }



    /// Zero the device memory.
    pub fn zero(&self) -> CudaResult<()> {
        ffi::cuda_memset(self.ptr, 0, self.size)
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            let _ = ffi::cuda_mem_free(self.ptr);
        }
    }
}
