/// Safe wrappers over the CUDA Driver API.
///
/// Uses runtime dynamic loading (libloading) to load nvcuda.dll,
/// avoiding the need for a static import library which NVIDIA
/// does not ship for the Driver API on Windows.
use std::ffi::{c_void, CStr};
use std::ptr;
use std::sync::OnceLock;

// ─── CUDA types ───────────────────────────────────────────────────────────────

pub type CUdevice = i32;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUdeviceptr = u64;
pub type CUresult = u32;
pub type CUstream = *mut c_void;

pub const CUDA_SUCCESS: CUresult = 0;

// ─── CU_DEVICE_ATTRIBUTE constants ────────────────────────────────────────────

pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;
pub const CU_DEVICE_ATTRIBUTE_MEMORY_BUS_WIDTH: i32 = 13;
pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: i32 = 16;

// ─── Error type ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("CUDA error: code {0}")]
pub struct CudaError(pub CUresult);

pub type CudaResult<T> = Result<T, CudaError>;

#[inline]
pub fn check(result: CUresult) -> CudaResult<()> {
    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(CudaError(result))
    }
}

// ─── Function pointer type definitions ────────────────────────────────────────

type FnCuInit = unsafe extern "C" fn(u32) -> CUresult;
type FnCuDeviceGetCount = unsafe extern "C" fn(*mut i32) -> CUresult;
type FnCuDeviceGet = unsafe extern "C" fn(*mut CUdevice, i32) -> CUresult;
type FnCuDeviceGetName = unsafe extern "C" fn(*mut u8, i32, CUdevice) -> CUresult;
type FnCuDeviceTotalMem = unsafe extern "C" fn(*mut usize, CUdevice) -> CUresult;
type FnCuDeviceGetAttribute = unsafe extern "C" fn(*mut i32, i32, CUdevice) -> CUresult;
type FnCuCtxCreate = unsafe extern "C" fn(*mut CUcontext, u32, CUdevice) -> CUresult;
type FnCuCtxDestroy = unsafe extern "C" fn(CUcontext) -> CUresult;
type FnCuCtxSynchronize = unsafe extern "C" fn() -> CUresult;
type FnCuMemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> CUresult;
type FnCuMemAlloc = unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult;
type FnCuMemFree = unsafe extern "C" fn(CUdeviceptr) -> CUresult;
type FnCuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult;
type FnCuMemsetD8 = unsafe extern "C" fn(CUdeviceptr, u8, usize) -> CUresult;
type FnCuModuleLoadData = unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult;
type FnCuModuleUnload = unsafe extern "C" fn(CUmodule) -> CUresult;
type FnCuModuleGetFunction =
    unsafe extern "C" fn(*mut CUfunction, CUmodule, *const u8) -> CUresult;
type FnCuLaunchKernel = unsafe extern "C" fn(
    CUfunction,
    u32, u32, u32,
    u32, u32, u32,
    u32,
    CUstream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CUresult;

// ─── Dynamic library handle ──────────────────────────────────────────────────

struct CudaApi {
    _lib: libloading::Library,
    cu_init: FnCuInit,
    cu_device_get_count: FnCuDeviceGetCount,
    cu_device_get: FnCuDeviceGet,
    cu_device_get_name: FnCuDeviceGetName,
    cu_device_total_mem: FnCuDeviceTotalMem,
    cu_device_get_attribute: FnCuDeviceGetAttribute,
    cu_ctx_create: FnCuCtxCreate,
    cu_ctx_destroy: FnCuCtxDestroy,
    cu_ctx_synchronize: FnCuCtxSynchronize,
    cu_mem_get_info: FnCuMemGetInfo,
    cu_mem_alloc: FnCuMemAlloc,
    cu_mem_free: FnCuMemFree,
    cu_memcpy_dtoh: FnCuMemcpyDtoH,
    cu_memset_d8: FnCuMemsetD8,
    cu_module_load_data: FnCuModuleLoadData,
    cu_module_unload: FnCuModuleUnload,
    cu_module_get_function: FnCuModuleGetFunction,
    cu_launch_kernel: FnCuLaunchKernel,
}

// Safety: the library handle and function pointers are valid for the lifetime
// of the process once loaded. CUDA functions are thread-safe.
unsafe impl Send for CudaApi {}
unsafe impl Sync for CudaApi {}

static CUDA: OnceLock<Result<CudaApi, String>> = OnceLock::new();

fn api() -> &'static CudaApi {
    CUDA.get_or_init(|| load_cuda_api())
        .as_ref()
        .expect("CUDA library failed to load")
}

fn load_cuda_api() -> Result<CudaApi, String> {
    unsafe {
        let lib = libloading::Library::new("nvcuda.dll")
            .map_err(|e| format!("Failed to load nvcuda.dll: {}", e))?;

        macro_rules! load_fn {
            ($lib:expr, $name:expr, $type:ty) => {{
                let sym: libloading::Symbol<$type> = $lib
                    .get($name)
                    .map_err(|e| format!("Failed to load {}: {}", String::from_utf8_lossy($name), e))?;
                *sym
            }};
        }

        let api = CudaApi {
            cu_init: load_fn!(lib, b"cuInit\0", FnCuInit),
            cu_device_get_count: load_fn!(lib, b"cuDeviceGetCount\0", FnCuDeviceGetCount),
            cu_device_get: load_fn!(lib, b"cuDeviceGet\0", FnCuDeviceGet),
            cu_device_get_name: load_fn!(lib, b"cuDeviceGetName\0", FnCuDeviceGetName),
            cu_device_total_mem: load_fn!(lib, b"cuDeviceTotalMem_v2\0", FnCuDeviceTotalMem),
            cu_device_get_attribute: load_fn!(lib, b"cuDeviceGetAttribute\0", FnCuDeviceGetAttribute),
            cu_ctx_create: load_fn!(lib, b"cuCtxCreate_v2\0", FnCuCtxCreate),
            cu_ctx_destroy: load_fn!(lib, b"cuCtxDestroy_v2\0", FnCuCtxDestroy),
            cu_ctx_synchronize: load_fn!(lib, b"cuCtxSynchronize\0", FnCuCtxSynchronize),
            cu_mem_get_info: load_fn!(lib, b"cuMemGetInfo_v2\0", FnCuMemGetInfo),
            cu_mem_alloc: load_fn!(lib, b"cuMemAlloc_v2\0", FnCuMemAlloc),
            cu_mem_free: load_fn!(lib, b"cuMemFree_v2\0", FnCuMemFree),
            cu_memcpy_dtoh: load_fn!(lib, b"cuMemcpyDtoH_v2\0", FnCuMemcpyDtoH),
            cu_memset_d8: load_fn!(lib, b"cuMemsetD8_v2\0", FnCuMemsetD8),
            cu_module_load_data: load_fn!(lib, b"cuModuleLoadData\0", FnCuModuleLoadData),
            cu_module_unload: load_fn!(lib, b"cuModuleUnload\0", FnCuModuleUnload),
            cu_module_get_function: load_fn!(lib, b"cuModuleGetFunction\0", FnCuModuleGetFunction),
            cu_launch_kernel: load_fn!(lib, b"cuLaunchKernel\0", FnCuLaunchKernel),
            _lib: lib,
        };

        Ok(api)
    }
}

// ─── Safe wrapper functions ───────────────────────────────────────────────────

pub fn cuda_init() -> CudaResult<()> {
    // Force load the library first
    let result = CUDA.get_or_init(|| load_cuda_api());
    match result {
        Ok(api) => check(unsafe { (api.cu_init)(0) }),
        Err(_e) => Err(CudaError(999)),
    }
}

pub fn cuda_device_count() -> CudaResult<i32> {
    let mut count = 0i32;
    check(unsafe { (api().cu_device_get_count)(&mut count) })?;
    Ok(count)
}

pub fn cuda_device_get(ordinal: i32) -> CudaResult<CUdevice> {
    let mut dev = 0;
    check(unsafe { (api().cu_device_get)(&mut dev, ordinal) })?;
    Ok(dev)
}

pub fn cuda_device_name(dev: CUdevice) -> CudaResult<String> {
    let mut buf = [0u8; 256];
    check(unsafe { (api().cu_device_get_name)(buf.as_mut_ptr(), 256, dev) })?;
    let name = unsafe { CStr::from_ptr(buf.as_ptr() as *const i8) };
    Ok(name.to_string_lossy().into_owned())
}

pub fn cuda_device_total_mem(dev: CUdevice) -> CudaResult<usize> {
    let mut bytes = 0usize;
    check(unsafe { (api().cu_device_total_mem)(&mut bytes, dev) })?;
    Ok(bytes)
}

pub fn cuda_device_attribute(dev: CUdevice, attrib: i32) -> CudaResult<i32> {
    let mut val = 0i32;
    check(unsafe { (api().cu_device_get_attribute)(&mut val, attrib, dev) })?;
    Ok(val)
}

pub fn cuda_ctx_create(dev: CUdevice) -> CudaResult<CUcontext> {
    let mut ctx: CUcontext = ptr::null_mut();
    check(unsafe { (api().cu_ctx_create)(&mut ctx, 0, dev) })?;
    Ok(ctx)
}

pub fn cuda_ctx_destroy(ctx: CUcontext) -> CudaResult<()> {
    check(unsafe { (api().cu_ctx_destroy)(ctx) })
}

pub fn cuda_synchronize() -> CudaResult<()> {
    check(unsafe { (api().cu_ctx_synchronize)() })
}

pub fn cuda_mem_get_info() -> CudaResult<(usize, usize)> {
    let mut free = 0usize;
    let mut total = 0usize;
    check(unsafe { (api().cu_mem_get_info)(&mut free, &mut total) })?;
    Ok((free, total))
}

pub fn cuda_mem_alloc(size: usize) -> CudaResult<CUdeviceptr> {
    let mut dptr: CUdeviceptr = 0;
    check(unsafe { (api().cu_mem_alloc)(&mut dptr, size) })?;
    Ok(dptr)
}

pub fn cuda_mem_free(dptr: CUdeviceptr) -> CudaResult<()> {
    check(unsafe { (api().cu_mem_free)(dptr) })
}

pub fn cuda_memcpy_dtoh(dst: &mut [u8], src: CUdeviceptr) -> CudaResult<()> {
    check(unsafe {
        (api().cu_memcpy_dtoh)(dst.as_mut_ptr() as *mut c_void, src, dst.len())
    })
}

pub fn cuda_memset(dst: CUdeviceptr, value: u8, count: usize) -> CudaResult<()> {
    check(unsafe { (api().cu_memset_d8)(dst, value, count) })
}

// ── Module & kernel functions (used by kernel.rs) ─────────────────────────────

pub fn cu_module_load_data(image: *const c_void) -> CudaResult<CUmodule> {
    let mut module: CUmodule = ptr::null_mut();
    check(unsafe { (api().cu_module_load_data)(&mut module, image) })?;
    Ok(module)
}

pub fn cu_module_unload(module: CUmodule) -> CudaResult<()> {
    check(unsafe { (api().cu_module_unload)(module) })
}

pub fn cu_module_get_function(module: CUmodule, name: *const u8) -> CudaResult<CUfunction> {
    let mut func: CUfunction = ptr::null_mut();
    check(unsafe { (api().cu_module_get_function)(&mut func, module, name) })?;
    Ok(func)
}

pub fn cu_launch_kernel(
    f: CUfunction,
    gx: u32, gy: u32, gz: u32,
    bx: u32, by: u32, bz: u32,
    shared: u32,
    stream: CUstream,
    args: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> CudaResult<()> {
    check(unsafe {
        (api().cu_launch_kernel)(f, gx, gy, gz, bx, by, bz, shared, stream, args, extra)
    })
}
