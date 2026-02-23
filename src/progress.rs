/// Shared progress state between scanner thread and GUI/CLI.
use crate::patterns::TestPattern;
use std::sync::{Arc, Mutex};

/// A single error entry logged by the GPU kernel.
#[derive(Debug, Clone, Copy, serde::Serialize)]
#[repr(C)]
pub struct ErrorEntry {
    pub byte_offset: u64,
    pub expected: u32,
    pub actual: u32,
}

/// Result for a single pattern test on a single chunk.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PatternResult {
    pub pattern: TestPattern,
    pub error_count: u32,
    pub errors: Vec<ErrorEntry>,
    pub speed_mbps: f64,
    pub duration_ms: f64,
}

/// Result for a single chunk.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ChunkResult {
    pub chunk_id: u32,
    pub offset_bytes: u64,
    pub size_bytes: usize,
    pub pattern_results: Vec<PatternResult>,
}

/// Current scan progress — shared between scanner thread and UI.
#[derive(Debug, Clone)]
pub struct ScanProgress {
    /// Current status message.
    pub status: String,

    /// Overall progress 0.0 → 1.0.
    pub progress: f32,

    /// Current pass (1-indexed).
    pub current_pass: u32,
    pub total_passes: u32,

    /// Current chunk within pass.
    pub current_chunk: u32,
    pub total_chunks: u32,

    /// Current pattern within chunk.
    pub current_pattern_name: String,

    /// Per-pattern results (updated live, indexed by pattern).
    pub pattern_summaries: Vec<PatternSummary>,

    /// All error entries collected so far (for chip identification).
    pub all_errors: Vec<ErrorEntry>,

    /// Memory controller error distribution (controller_index → error_count).
    pub controller_errors: Vec<u32>,

    /// Is the scan complete?
    pub finished: bool,

    /// Is the scan running?
    pub running: bool,

    /// Error message if scan failed.
    pub error_message: Option<String>,

    /// Total VRAM tested in bytes.
    pub total_bytes_tested: u64,

    /// Total errors found.
    pub total_errors: u64,

    /// Number of memory controllers on this GPU.
    pub num_controllers: u32,

    /// Total scan duration in seconds.
    pub elapsed_secs: f64,
}

/// Aggregated result for one pattern across all chunks and passes.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PatternSummary {
    pub pattern: TestPattern,
    pub total_errors: u64,
    pub avg_speed_mbps: f64,
}

impl ScanProgress {
    pub fn new(num_controllers: u32) -> Self {
        Self {
            status: "Ready".to_string(),
            progress: 0.0,
            current_pass: 0,
            total_passes: 0,
            current_chunk: 0,
            total_chunks: 0,
            current_pattern_name: String::new(),
            pattern_summaries: Vec::new(),
            all_errors: Vec::new(),
            controller_errors: vec![0; num_controllers as usize],
            finished: false,
            running: false,
            error_message: None,
            total_bytes_tested: 0,
            total_errors: 0,
            num_controllers,
            elapsed_secs: 0.0,
        }
    }
}

pub type SharedProgress = Arc<Mutex<ScanProgress>>;

pub fn create_shared_progress(num_controllers: u32) -> SharedProgress {
    Arc::new(Mutex::new(ScanProgress::new(num_controllers)))
}
