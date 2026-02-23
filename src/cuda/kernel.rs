/// CUDA module (PTX) loading and kernel launch wrappers.
use super::ffi::{self, CUfunction, CUmodule, CudaResult};
use std::ffi::{c_void, CString};
use std::ptr;

/// RAII wrapper for a loaded CUDA module (PTX code).
pub struct CudaModule {
    module: CUmodule,
}

impl CudaModule {
    /// Load a PTX module from bytes. The bytes must be null-terminated.
    pub fn load_ptx(ptx: &[u8]) -> CudaResult<Self> {
        let module = ffi::cu_module_load_data(ptx.as_ptr() as *const c_void)?;
        Ok(CudaModule { module })
    }

    /// Get a kernel function by name from this module.
    pub fn get_function(&self, name: &str) -> CudaResult<CudaKernel> {
        let c_name = CString::new(name).expect("Kernel name contains null byte");
        let func = ffi::cu_module_get_function(self.module, c_name.as_ptr() as *const u8)?;
        Ok(CudaKernel { function: func })
    }
}

impl Drop for CudaModule {
    fn drop(&mut self) {
        if !self.module.is_null() {
            let _ = ffi::cu_module_unload(self.module);
        }
    }
}

/// A handle to a CUDA kernel function.
#[derive(Clone, Copy)]
pub struct CudaKernel {
    function: CUfunction,
}

impl CudaKernel {
    /// Launch this kernel with the given grid/block dimensions and arguments.
    ///
    /// # Safety
    /// The caller must ensure `args` pointers are valid and match the kernel signature.
    pub unsafe fn launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        args: &mut [*mut c_void],
    ) -> CudaResult<()> {
        ffi::cu_launch_kernel(
            self.function,
            grid.0, grid.1, grid.2,
            block.0, block.1, block.2,
            shared_mem,
            ptr::null_mut(), // default stream
            args.as_mut_ptr(),
            ptr::null_mut(),
        )
    }
}

/// Helper to compute grid dimensions for a 1D kernel launch.
pub fn compute_grid_1d(num_elements: u64, block_size: u32) -> (u32, u32, u32) {
    let grid_x = ((num_elements + block_size as u64 - 1) / block_size as u64).min(65535) as u32;
    (grid_x, 1, 1)
}
