/// Core VRAM scanning engine.
///
/// Implements the sliding-window scan with micro-batch kernel execution
/// to avoid TDR timeouts. Runs in a background thread and communicates
/// progress via SharedProgress (Arc<Mutex<ScanProgress>>).
use crate::cuda::device::CudaDevice;
use crate::cuda::ffi;
use crate::cuda::kernel::{compute_grid_1d, CudaKernel, CudaModule};
use crate::cuda::memory::DeviceBuffer;
use crate::patterns::TestPattern;
use crate::progress::{ErrorEntry, PatternSummary, SharedProgress};
use std::ffi::c_void;
use std::time::Instant;

/// Maximum error entries logged per kernel invocation (must match kernel define).
const MAX_ERROR_LOG: usize = 1024;
const BLOCK_SIZE: u32 = 256;

/// Embedded PTX (included at compile time from build.rs output).
const PATTERN_FILL_PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pattern_fill.ptx"));
const PATTERN_VERIFY_PTX: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/pattern_verify.ptx"));

/// Run the VRAM diagnostic scan.
pub fn run_scan(
    device: &CudaDevice,
    chunk_size: usize,
    micro_batch_size: usize,
    passes: u32,
    headroom: usize,
    progress: SharedProgress,
) -> Result<(), String> {
    // Mark as running
    {
        let mut p = progress.lock().unwrap();
        p.running = true;
        p.status = "Initializing CUDA...".to_string();
    }

    let start_time = Instant::now();

    // Create CUDA context
    let ctx = ffi::cuda_ctx_create(device.handle).map_err(|e| format!("ctx create: {}", e))?;

    let result = run_scan_inner(
        device,
        chunk_size,
        micro_batch_size,
        passes,
        headroom,
        &progress,
        start_time,
    );

    // Cleanup context
    let _ = ffi::cuda_ctx_destroy(ctx);

    let mut p = progress.lock().unwrap();
    p.running = false;
    p.finished = true;
    p.elapsed_secs = start_time.elapsed().as_secs_f64();

    match result {
        Ok(()) => {
            if p.total_errors == 0 {
                p.status = "✅ Complete — No errors found".to_string();
            } else {
                p.status = format!("⚠ Complete — {} errors found!", p.total_errors);
            }
            Ok(())
        }
        Err(e) => {
            p.error_message = Some(e.clone());
            p.status = format!("❌ Error: {}", e);
            Err(e)
        }
    }
}

fn run_scan_inner(
    device: &CudaDevice,
    chunk_size: usize,
    micro_batch_size: usize,
    passes: u32,
    headroom: usize,
    progress: &SharedProgress,
    start_time: Instant,
) -> Result<(), String> {
    // Load PTX modules
    update_status(progress, "Loading CUDA kernels...");

    // Append null terminator for CUDA module loader
    let mut fill_ptx = PATTERN_FILL_PTX.to_vec();
    fill_ptx.push(0);
    let mut verify_ptx = PATTERN_VERIFY_PTX.to_vec();
    verify_ptx.push(0);

    let fill_module =
        CudaModule::load_ptx(&fill_ptx).map_err(|e| format!("Load fill PTX: {}", e))?;
    let verify_module =
        CudaModule::load_ptx(&verify_ptx).map_err(|e| format!("Load verify PTX: {}", e))?;

    let fill_kernel = fill_module
        .get_function("pattern_fill_u32")
        .map_err(|e| format!("Get fill kernel: {}", e))?;
    let verify_kernel = verify_module
        .get_function("pattern_verify_u32")
        .map_err(|e| format!("Get verify kernel: {}", e))?;

    // Query available VRAM
    let (free_mem, _total_mem) =
        CudaDevice::get_memory_info().map_err(|e| format!("MemGetInfo: {}", e))?;

    let usable_mem = free_mem.saturating_sub(headroom);
    let num_chunks = (usable_mem / chunk_size).max(1) as u32;
    let patterns = TestPattern::basic_patterns();

    // Initialize progress
    {
        let mut p = progress.lock().unwrap();
        p.total_passes = passes;
        p.total_chunks = num_chunks;
        p.pattern_summaries = patterns
            .iter()
            .map(|pat| PatternSummary {
                pattern: *pat,
                total_errors: 0,
                avg_speed_mbps: 0.0,
            })
            .collect();
    }

    let num_controllers = device.memory_controller_count() as u32;

    // Main scan loop: passes → chunks → patterns → micro-batches
    for pass in 0..passes {
        // Check if we should stop
        if !progress.lock().unwrap().running {
            return Ok(());
        }

        for chunk_idx in 0..num_chunks {
            // Update progress
            {
                let mut p = progress.lock().unwrap();
                if !p.running {
                    return Ok(());
                }
                p.current_pass = pass + 1;
                p.current_chunk = chunk_idx + 1;
                p.status = format!(
                    "Pass {}/{} — Chunk {}/{}",
                    pass + 1,
                    passes,
                    chunk_idx + 1,
                    num_chunks
                );

                let total_steps = (passes * num_chunks * patterns.len() as u32) as f32;
                let current_step =
                    (pass * num_chunks * patterns.len() as u32 + chunk_idx * patterns.len() as u32)
                        as f32;
                p.progress = current_step / total_steps;
                p.elapsed_secs = start_time.elapsed().as_secs_f64();
            }

            // Allocate chunk
            let chunk_buf = DeviceBuffer::alloc(chunk_size)
                .map_err(|e| format!("Alloc chunk {} ({}MB): {}", chunk_idx, chunk_size / (1024*1024), e))?;

            // Allocate error counter + error log on device
            let error_count_buf = DeviceBuffer::alloc(4)
                .map_err(|e| format!("Alloc error counter: {}", e))?;
            let error_log_buf =
                DeviceBuffer::alloc(MAX_ERROR_LOG * std::mem::size_of::<ErrorEntry>())
                    .map_err(|e| format!("Alloc error log: {}", e))?;
            let error_log_index_buf = DeviceBuffer::alloc(4)
                .map_err(|e| format!("Alloc error log index: {}", e))?;

            for (pat_idx, pattern) in patterns.iter().enumerate() {
                // Check if we should stop
                if !progress.lock().unwrap().running {
                    return Ok(());
                }

                {
                    let mut p = progress.lock().unwrap();
                    p.current_pattern_name = pattern.short_name().to_string();
                }

                let pattern_val = pattern.as_u32();
                let pat_start = Instant::now();
                let mut total_chunk_errors: u64 = 0;

                // Process chunk in micro-batches to stay under TDR limit
                let mut offset: usize = 0;
                while offset < chunk_size {
                    let batch_size = micro_batch_size.min(chunk_size - offset);
                    let num_elements = (batch_size / 4) as u64; // u32 elements
                    let batch_ptr = chunk_buf.ptr + offset as u64;

                    // --- Fill micro-batch ---
                    launch_fill(
                        &fill_kernel,
                        batch_ptr,
                        pattern_val,
                        num_elements,
                    )?;
                    ffi::cuda_synchronize()
                        .map_err(|e| format!("Sync after fill: {}", e))?;

                    // --- Verify micro-batch ---
                    // Zero the error counter and log index
                    error_count_buf.zero().map_err(|e| format!("Zero counter: {}", e))?;
                    error_log_index_buf.zero().map_err(|e| format!("Zero log idx: {}", e))?;

                    launch_verify(
                        &verify_kernel,
                        batch_ptr,
                        pattern_val,
                        num_elements,
                        error_count_buf.ptr,
                        error_log_buf.ptr,
                        error_log_index_buf.ptr,
                    )?;
                    ffi::cuda_synchronize()
                        .map_err(|e| format!("Sync after verify: {}", e))?;

                    // Read error count
                    let mut err_count_host = [0u8; 4];
                    error_count_buf
                        .copy_to_host(&mut err_count_host)
                        .map_err(|e| format!("Read error count: {}", e))?;
                    let batch_errors = u32::from_le_bytes(err_count_host);

                    if batch_errors > 0 {
                        // Read error log entries
                        let mut log_index_host = [0u8; 4];
                        error_log_index_buf
                            .copy_to_host(&mut log_index_host)
                            .map_err(|e| format!("Read log index: {}", e))?;
                        let log_count =
                            (u32::from_le_bytes(log_index_host) as usize).min(MAX_ERROR_LOG);

                        let entry_size = std::mem::size_of::<ErrorEntry>();
                        let mut log_buf = vec![0u8; log_count * entry_size];
                        let log_slice_buf = DeviceBuffer {
                            ptr: error_log_buf.ptr,
                            size: log_count * entry_size,
                        };
                        log_slice_buf
                            .copy_to_host(&mut log_buf)
                            .map_err(|e| format!("Read error log: {}", e))?;
                        // Prevent double-free since log_slice_buf doesn't own the memory
                        std::mem::forget(log_slice_buf);

                        // Parse error entries and adjust offsets
                        let entries: &[ErrorEntry] = unsafe {
                            std::slice::from_raw_parts(
                                log_buf.as_ptr() as *const ErrorEntry,
                                log_count,
                            )
                        };

                        let mut p = progress.lock().unwrap();
                        for entry in entries {
                            let adjusted = ErrorEntry {
                                byte_offset: entry.byte_offset + offset as u64,
                                expected: entry.expected,
                                actual: entry.actual,
                            };
                            p.all_errors.push(adjusted);

                            // Update controller-level error tracking
                            let total_stride =
                                256 * num_controllers as u64;
                            let position = adjusted.byte_offset % total_stride;
                            let ctrl_id = (position / 256) as usize;
                            if ctrl_id < p.controller_errors.len() {
                                p.controller_errors[ctrl_id] += 1;
                            }
                        }
                        total_chunk_errors += batch_errors as u64;
                    }

                    offset += batch_size;
                }

                let pat_duration = pat_start.elapsed();
                let speed_mbps =
                    (chunk_size as f64 / (1024.0 * 1024.0)) / pat_duration.as_secs_f64();

                // Update pattern summary
                {
                    let mut p = progress.lock().unwrap();
                    p.total_bytes_tested += chunk_size as u64;
                    p.total_errors += total_chunk_errors;

                    if let Some(summary) = p.pattern_summaries.get_mut(pat_idx) {
                        summary.total_errors += total_chunk_errors;
                        // Running average of speed
                        let count = (pass * num_chunks + chunk_idx + 1) as f64;
                        summary.avg_speed_mbps =
                            summary.avg_speed_mbps * ((count - 1.0) / count)
                                + speed_mbps / count;
                    }
                }
            }
            // DeviceBuffer dropped here → cuMemFree
        }
    }

    // Final update
    {
        let mut p = progress.lock().unwrap();
        p.progress = 1.0;
    }

    Ok(())
}

fn update_status(progress: &SharedProgress, status: &str) {
    let mut p = progress.lock().unwrap();
    p.status = status.to_string();
}

fn launch_fill(
    kernel: &CudaKernel,
    data_ptr: u64,
    pattern: u32,
    num_elements: u64,
) -> Result<(), String> {
    let grid = compute_grid_1d(num_elements, BLOCK_SIZE);
    let block = (BLOCK_SIZE, 1, 1);

    let mut data_ptr_val = data_ptr;
    let mut pattern_val = pattern;
    let mut num_elements_val = num_elements;

    let mut args: [*mut c_void; 3] = [
        &mut data_ptr_val as *mut u64 as *mut c_void,
        &mut pattern_val as *mut u32 as *mut c_void,
        &mut num_elements_val as *mut u64 as *mut c_void,
    ];

    unsafe {
        kernel
            .launch(grid, block, 0, &mut args)
            .map_err(|e| format!("Fill kernel launch: {}", e))
    }
}

fn launch_verify(
    kernel: &CudaKernel,
    data_ptr: u64,
    expected: u32,
    num_elements: u64,
    error_count_ptr: u64,
    error_log_ptr: u64,
    error_log_index_ptr: u64,
) -> Result<(), String> {
    let grid = compute_grid_1d(num_elements, BLOCK_SIZE);
    let block = (BLOCK_SIZE, 1, 1);

    let mut data_ptr_val = data_ptr;
    let mut expected_val = expected;
    let mut num_elements_val = num_elements;
    let mut error_count_val = error_count_ptr;
    let mut error_log_val = error_log_ptr;
    let mut error_log_index_val = error_log_index_ptr;

    let mut args: [*mut c_void; 6] = [
        &mut data_ptr_val as *mut u64 as *mut c_void,
        &mut expected_val as *mut u32 as *mut c_void,
        &mut num_elements_val as *mut u64 as *mut c_void,
        &mut error_count_val as *mut u64 as *mut c_void,
        &mut error_log_val as *mut u64 as *mut c_void,
        &mut error_log_index_val as *mut u64 as *mut c_void,
    ];

    unsafe {
        kernel
            .launch(grid, block, 0, &mut args)
            .map_err(|e| format!("Verify kernel launch: {}", e))
    }
}
