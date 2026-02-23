# VRAM Diagnostics

A powerful, native Rust-based diagnostic tool designed to test NVIDIA GPU VRAM for hardware faults. 

## Motivation
This project was born out of necessity: dealing with a graphics card that had malfunctioning VRAM, and needing a reliable, low-level testing tool during the repair/servicing process. Dedicated hardware diagnostics (like NVIDIA MATS/MODS) are often proprietary, hard to obtain, or require booting into specific Linux environments. 

This tool provides a native Windows alternative, allowing you to run intensive memory pattern tests directly on your GPU to identify bad memory chips, all while keeping the OS stable.

## Features
- **Multiple Test Patterns:** Uses industry-standard memory diagnostic patterns to catch different types of hardware faults (All Zero, All One, Walking 1, Walking 0, Checkerboard, Inverse Checkerboard, Random).
- **Custom CUDA Kernels:** Direct PTX execution for blazing-fast memory filling and verification, maximizing memory bandwidth.
- **Sliding Window Architecture:** Tests VRAM in manageable chunks to bypass Windows Display Driver Model (WDDM) allocation limits and prevent OS crashes or TDR (Timeout Detection and Recovery) triggers.
- **Graphical User Interface (GUI):** A clean, easy-to-use GUI built with `egui` to select GPUs, adjust chunk sizes, set test passes, and monitor progress in real-time.
- **Command-Line Interface (CLI):** For automated testing, logging, and headless environments.
- **Chip Mapping Analysis:** Attempts to estimate which physical memory chip might be faulty based on the stride (distance) between memory errors.

## Requirements
- **OS:** Windows 10 / 11 (64-bit)
- **GPU:** NVIDIA Graphics Card with CUDA support (Compute Capability 5.0+)
- **Drivers:** NVIDIA Display Driver installed (provides the required `nvcuda.dll`)

## Usage

You can run the application normally to launch the Graphical Interface:

```bash
cargo run --release
```

### GUI Mode
1. **GPU:** Select the target NVIDIA GPU from the dropdown menu.
2. **Chunk:** Select the VRAM allocation chunk size (e.g., 512MB, 1024MB). *Note: Larger chunks are faster but may trigger driver timeouts or out-of-memory errors on Windows if the VRAM is heavily fragmented.*
3. **Passes:** Choose how many full-memory passes to run.
4. Click **START** and let the diagnostic run. If the text flashes red and the error count goes up, your VRAM has physical faults.

### CLI Mode (Advanced)
If you prefer running tests from the terminal or saving automated reports, you can use the `--cli` flag:

```bash
cargo run --release -- --cli --chunk-size 1024 --passes 3 --device 0
```

To export the diagnostic results as a JSON file:
```bash
cargo run --release -- --cli --output report.json
```

## Limitations & Known Issues
1. **WDDM Allocation Limits:** Because this tool runs in standard Windows, it is bound by WDDM (Windows Display Driver Model). Windows reserves a significant portion of VRAM for desktop rendering and system processes. You can never test 100% of the VRAM in Windows (typically only ~80-90% is accessible).
2. **TDR (Timeout Detection and Recovery):** If a chunk size is too large or the GPU is too slow, the CUDA kernel execution may take longer than 2 seconds, causing Windows to reset the display driver (screen flashes black). If this happens, **lower the Chunk Size**.
3. **Chip Identification Accuracy:** The chip mapping feature (`Chip Map`) makes an educated guess based on error strides (e.g., 32-byte or 64-byte gaps). Because NVIDIA's memory controllers interleave addresses across multiple physical chips in complex (and proprietary) ways, the exact faulty chip (e.g., "Chip U43") cannot be guaranteed with 100% accuracy. Use it as a guiding hint, not absolute truth.
4. **NVIDIA Only:** Due to the reliance on the CUDA Driver API, this tool currently only supports NVIDIA GPUs.

## Building from Source
1. Install [Rust](https://rustup.rs/).
2. Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (Required for `nvcc` to compile the `.cu` kernel files).
3. Ensure Microsoft Visual Studio Build Tools with C++ workloads are installed.
4. Run:
```bash
git clone <repository_url>
cd VramDiagnostics
cargo build --release
```

## Disclaimer
Use this tool at your own risk. Subjecting failing hardware to intense, repeated stress tests can sometimes accelerate degradation. This software is provided "as is", without warranty of any kind.
