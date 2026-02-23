/// Scan result reporting (JSON + console).
use crate::chip_map::{self, ChipAnalysis};
use crate::progress::ScanProgress;
use chrono::{DateTime, Utc};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct ScanReport {
    pub timestamp: DateTime<Utc>,
    pub gpu_name: String,
    pub gpu_index: i32,
    pub total_vram_bytes: u64,
    pub tested_vram_bytes: u64,
    pub chunk_size_bytes: u64,
    pub passes: u32,
    pub total_errors: u64,
    pub elapsed_seconds: f64,
    pub patterns: Vec<PatternReport>,
    pub chip_analysis: Option<ChipAnalysis>,
    pub verdict: String,
}

#[derive(Debug, Serialize)]
pub struct PatternReport {
    pub name: String,
    pub errors: u64,
    pub speed_mbps: f64,
}

impl ScanReport {
    pub fn from_progress(
        progress: &ScanProgress,
        gpu_name: &str,
        gpu_index: i32,
        total_vram: u64,
        chunk_size: u64,
    ) -> Self {
        let patterns: Vec<PatternReport> = progress
            .pattern_summaries
            .iter()
            .map(|p| PatternReport {
                name: p.pattern.name(),
                errors: p.total_errors,
                speed_mbps: p.avg_speed_mbps,
            })
            .collect();

        let chip_analysis = if !progress.all_errors.is_empty() && progress.num_controllers > 0 {
            Some(chip_map::analyze_errors(
                &progress.all_errors,
                progress.num_controllers,
            ))
        } else {
            None
        };

        let verdict = if progress.total_errors == 0 {
            "PASS — No VRAM errors detected.".to_string()
        } else {
            format!(
                "FAIL — {} VRAM errors detected!",
                progress.total_errors
            )
        };

        ScanReport {
            timestamp: Utc::now(),
            gpu_name: gpu_name.to_string(),
            gpu_index,
            total_vram_bytes: total_vram,
            tested_vram_bytes: progress.total_bytes_tested,
            chunk_size_bytes: chunk_size,
            passes: progress.total_passes,
            total_errors: progress.total_errors,
            elapsed_seconds: progress.elapsed_secs,
            patterns,
            chip_analysis,
            verdict,
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    pub fn print_console(&self) {
        println!("\n{}", "=".repeat(60));
        println!("  VRAM Diagnostics Report");
        println!("{}", "=".repeat(60));
        println!("  GPU:           {} (index {})", self.gpu_name, self.gpu_index);
        println!(
            "  Total VRAM:    {:.1} GB",
            self.total_vram_bytes as f64 / 1_073_741_824.0
        );
        println!(
            "  Tested:        {:.1} GB",
            self.tested_vram_bytes as f64 / 1_073_741_824.0
        );
        println!("  Passes:        {}", self.passes);
        println!("  Duration:      {:.1}s", self.elapsed_seconds);
        println!("{}", "-".repeat(60));

        for p in &self.patterns {
            println!(
                "  {:20} Errors: {:>8}   Speed: {:>8.1} MB/s",
                p.name, p.errors, p.speed_mbps
            );
        }

        println!("{}", "-".repeat(60));
        println!("  Total Errors:  {}", self.total_errors);
        println!("  Verdict:       {}", self.verdict);

        if let Some(ref analysis) = self.chip_analysis {
            println!("\n{}", "-".repeat(60));
            println!("  Chip Analysis (Memory Controller Mapping)");
            println!("{}", "-".repeat(60));
            for chip in &analysis.suspected_bad_chips {
                println!(
                    "  ⚠ {} — {} errors (confidence: {:.1}%)",
                    chip.chip_label, chip.error_count, chip.confidence
                );
            }
        }

        println!("{}\n", "=".repeat(60));
    }
}
