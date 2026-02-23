#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod chip_map;
mod config;
mod cuda;
mod gui;
mod patterns;
mod progress;
mod report;
mod scanner;

use clap::Parser;
use config::Config;

fn main() {
    env_logger::init();

    let config = Config::parse();

    if config.effective_cli() {
        run_cli(config);
    } else {
        if let Err(e) = gui::run_gui() {
            eprintln!("GUI error: {}", e);
            std::process::exit(1);
        }
    }
}

fn run_cli(config: Config) {
    use indicatif::{ProgressBar, ProgressStyle};

    println!("VRAM Diagnostics v{}", env!("CARGO_PKG_VERSION"));
    println!("Mode: CLI");
    println!(
        "Chunk: {} MB | Passes: {} | Device: {}",
        config.chunk_size, config.passes, config.device
    );
    println!();

    // Initialize CUDA
    if let Err(e) = cuda::ffi::cuda_init() {
        eprintln!("CUDA initialization failed: {}", e);
        std::process::exit(1);
    }

    let devices = match cuda::CudaDevice::enumerate() {
        Ok(devs) => devs,
        Err(e) => {
            eprintln!("GPU enumeration failed: {}", e);
            std::process::exit(1);
        }
    };

    if devices.is_empty() {
        eprintln!("No NVIDIA GPUs found.");
        std::process::exit(1);
    }

    let device_idx = config.device as usize;
    if device_idx >= devices.len() {
        eprintln!(
            "Device index {} out of range (found {} GPUs)",
            config.device,
            devices.len()
        );
        std::process::exit(1);
    }

    let device = &devices[device_idx];
    println!("GPU: {}", device);
    println!();

    let num_controllers = device.memory_controller_count() as u32;
    let shared_progress = progress::create_shared_progress(num_controllers);

    // Progress bar
    let pb = ProgressBar::new(100);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.green/dark_gray}] {pos}% {msg}")
            .unwrap()
            .progress_chars("█▓░"),
    );

    // Run scan in background thread
    let device_clone = device.clone();
    let chunk_size = config.chunk_size_bytes();
    let micro_batch = config.micro_batch_bytes();
    let passes = config.passes;
    let headroom = config.headroom_bytes();
    let progress_clone = shared_progress.clone();

    let handle = std::thread::spawn(move || {
        scanner::run_scan(
            &device_clone,
            chunk_size,
            micro_batch,
            passes,
            headroom,
            progress_clone,
        )
    });

    // Poll progress
    loop {
        let p = shared_progress.lock().unwrap().clone();
        pb.set_position((p.progress * 100.0) as u64);
        pb.set_message(p.status.clone());

        if p.finished {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }

    pb.finish_with_message("Done");

    let _ = handle.join();

    // Generate report
    let p = shared_progress.lock().unwrap().clone();
    let report = report::ScanReport::from_progress(
        &p,
        &device.name,
        device.ordinal,
        device.total_memory as u64,
        config.chunk_size * 1024 * 1024,
    );

    report.print_console();

    // Save JSON if requested
    if let Some(ref output_path) = config.output {
        let json = report.to_json();
        if let Err(e) = std::fs::write(output_path, &json) {
            eprintln!("Failed to write JSON report: {}", e);
        } else {
            println!("JSON report saved to: {}", output_path);
        }
    }
}
