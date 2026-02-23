/// CUDA device enumeration and information queries.
use super::ffi::{self, CUdevice, CudaResult};

#[derive(Debug, Clone)]
pub struct CudaDevice {
    pub ordinal: i32,
    pub handle: CUdevice,
    pub name: String,
    pub total_memory: usize,
    pub compute_major: i32,
    pub compute_minor: i32,
    pub memory_bus_width: i32,
    pub sm_count: i32,
}

impl CudaDevice {
    pub fn enumerate() -> CudaResult<Vec<CudaDevice>> {
        let count = ffi::cuda_device_count()?;
        let mut devices = Vec::with_capacity(count as usize);

        for i in 0..count {
            let handle = ffi::cuda_device_get(i)?;
            let name = ffi::cuda_device_name(handle)?;
            let total_memory = ffi::cuda_device_total_mem(handle)?;
            let compute_major = ffi::cuda_device_attribute(
                handle,
                ffi::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            )?;
            let compute_minor = ffi::cuda_device_attribute(
                handle,
                ffi::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            )?;
            let memory_bus_width = ffi::cuda_device_attribute(
                handle,
                ffi::CU_DEVICE_ATTRIBUTE_MEMORY_BUS_WIDTH,
            )?;
            let sm_count = ffi::cuda_device_attribute(
                handle,
                ffi::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )?;

            devices.push(CudaDevice {
                ordinal: i,
                handle,
                name,
                total_memory,
                compute_major,
                compute_minor,
                memory_bus_width,
                sm_count,
            });
        }

        Ok(devices)
    }

    /// Returns (free_bytes, total_bytes) for the current context.
    pub fn get_memory_info() -> CudaResult<(usize, usize)> {
        ffi::cuda_mem_get_info()
    }

    /// Number of memory controllers based on bus width (32-bit per controller).
    pub fn memory_controller_count(&self) -> i32 {
        self.memory_bus_width / 32
    }

    /// Total VRAM chips (2 per controller for GDDR6X).
    pub fn estimated_chip_count(&self) -> i32 {
        self.memory_controller_count() * 2
    }

    pub fn display_label(&self) -> String {
        let mem_gb = self.total_memory as f64 / (1024.0 * 1024.0 * 1024.0);
        format!(
            "GPU {}: {} [{:.0} GB] (SM {})",
            self.ordinal, self.name, mem_gb, self.sm_count
        )
    }
}

impl std::fmt::Display for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mem_gb = self.total_memory as f64 / (1024.0 * 1024.0 * 1024.0);
        write!(
            f,
            "{} ({:.1} GB, {}-bit bus, SM {}.{}, {} SMs)",
            self.name,
            mem_gb,
            self.memory_bus_width,
            self.compute_major,
            self.compute_minor,
            self.sm_count,
        )
    }
}
