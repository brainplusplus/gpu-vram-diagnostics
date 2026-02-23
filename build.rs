use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernel_dir = Path::new("kernels");

    let kernels = ["pattern_fill", "pattern_verify"];

    let nvcc = find_nvcc().expect(
        "Could not find nvcc. Please install CUDA Toolkit 12.x+ and ensure nvcc is on PATH \
         or set CUDA_PATH environment variable.",
    );

    // Find MSVC cl.exe for nvcc host compiler
    let ccbin = find_msvc_cl();

    for kernel in &kernels {
        let cu_file = kernel_dir.join(format!("{}.cu", kernel));
        let ptx_file = out_dir.join(format!("{}.ptx", kernel));

        println!("cargo:rerun-if-changed={}", cu_file.display());

        let mut cmd = Command::new(&nvcc);
        cmd.args(["--ptx", "-O3", "--use_fast_math", "-o"]);
        cmd.arg(&ptx_file);
        cmd.arg(&cu_file);

        // Pass MSVC compiler path if found
        if let Some(ref cl_dir) = ccbin {
            cmd.arg(format!("-ccbin={}", cl_dir.display()));
        }

        let status = cmd.status().expect("Failed to execute nvcc");

        if !status.success() {
            panic!(
                "nvcc failed to compile {} (exit code: {:?})",
                cu_file.display(),
                status.code()
            );
        }

        println!(
            "cargo:warning=Compiled {} -> {}",
            cu_file.display(),
            ptx_file.display()
        );
    }

    // Add Windows Application Icon
    #[cfg(windows)]
    {
        let mut res = winres::WindowsResource::new();
        res.set_icon("assets/icon.ico");
        if let Err(e) = res.compile() {
            println!("cargo:warning=Failed to compile Windows resource (icon). Make sure assets/icon.ico exists. Error: {}", e);
        }
    }
}

fn find_nvcc() -> Option<PathBuf> {
    // Check PATH first
    if Command::new("nvcc").arg("--version").output().is_ok() {
        return Some(PathBuf::from("nvcc"));
    }

    // Check CUDA_PATH environment variable
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let nvcc_path = PathBuf::from(&cuda_path).join("bin").join("nvcc.exe");
        if nvcc_path.exists() {
            return Some(nvcc_path);
        }
    }

    // Check common Windows install locations
    let cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA";
    if let Ok(entries) = std::fs::read_dir(cuda_base) {
        let mut versions: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.join("bin").join("nvcc.exe").exists())
            .collect();
        versions.sort();
        if let Some(latest) = versions.last() {
            return Some(latest.join("bin").join("nvcc.exe"));
        }
    }

    None
}

/// Find MSVC cl.exe directory for nvcc -ccbin flag.
fn find_msvc_cl() -> Option<PathBuf> {
    // Check if cl.exe is already on PATH
    if Command::new("cl.exe").output().is_ok() {
        return None; // nvcc will find it on PATH
    }

    // Search common Visual Studio install locations
    let vs_paths = [
        r"C:\Program Files\Microsoft Visual Studio\2022",
        r"C:\Program Files\Microsoft Visual Studio\2019",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019",
    ];

    for vs_base in &vs_paths {
        let editions = ["Enterprise", "Professional", "Community", "BuildTools"];
        for edition in &editions {
            let msvc_base = PathBuf::from(vs_base).join(edition).join("VC").join("Tools").join("MSVC");
            if !msvc_base.exists() {
                continue;
            }
            // Find latest MSVC version
            if let Ok(entries) = std::fs::read_dir(&msvc_base) {
                let mut versions: Vec<PathBuf> = entries
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .collect();
                versions.sort();
                if let Some(latest) = versions.last() {
                    let cl_path = latest.join("bin").join("Hostx64").join("x64");
                    if cl_path.join("cl.exe").exists() {
                        return Some(cl_path);
                    }
                }
            }
        }
    }

    None
}
