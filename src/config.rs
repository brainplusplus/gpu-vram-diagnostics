/// CLI configuration and argument parsing.
use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "vram-diagnostics",
    version,
    about = "Professional-grade VRAM diagnostic tool for NVIDIA GPUs"
)]
pub struct Config {
    /// Run in CLI mode (TUI progress bars) instead of GUI.
    #[arg(long, default_value_t = false)]
    pub cli: bool,

    /// Chunk size in megabytes for VRAM scanning.
    #[arg(long, default_value_t = 512)]
    pub chunk_size: u64,

    /// Number of passes over VRAM.
    #[arg(long, default_value_t = 2)]
    pub passes: u32,

    /// GPU device index to test.
    #[arg(long, default_value_t = 0)]
    pub device: i32,

    /// Output JSON report to file.
    #[arg(long)]
    pub output: Option<String>,

    /// Plain text log mode (for piping to files). Implies --cli.
    #[arg(long, default_value_t = false)]
    pub log: bool,

    /// Micro-batch size in MB (for TDR safety). Each kernel processes this much at a time.
    #[arg(long, default_value_t = 32)]
    pub micro_batch: u64,

    /// VRAM headroom in MB to leave free (for Windows desktop).
    #[arg(long, default_value_t = 512)]
    pub headroom: u64,
}

impl Config {
    pub fn chunk_size_bytes(&self) -> usize {
        (self.chunk_size * 1024 * 1024) as usize
    }

    pub fn micro_batch_bytes(&self) -> usize {
        (self.micro_batch * 1024 * 1024) as usize
    }

    pub fn headroom_bytes(&self) -> usize {
        (self.headroom * 1024 * 1024) as usize
    }

    /// If --log is set, force CLI mode.
    pub fn effective_cli(&self) -> bool {
        self.cli || self.log
    }
}
