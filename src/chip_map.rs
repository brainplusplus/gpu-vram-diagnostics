/// Memory controller / VRAM chip identification via address stride analysis.
use crate::progress::ErrorEntry;
use serde::Serialize;

/// Default interleave granularity in bytes (256 bytes per controller).
const DEFAULT_INTERLEAVE_BYTES: u64 = 256;

/// Analysis result for chip identification.
#[derive(Debug, Clone, Serialize)]
pub struct ChipAnalysis {
    pub num_controllers: u32,
    pub interleave_bytes: u64,
    pub controller_results: Vec<ControllerResult>,
    pub suspected_bad_chips: Vec<SuspectedChip>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ControllerResult {
    pub controller_id: u32,
    pub error_count: u64,
    pub upper_16bit_errors: u64,
    pub lower_16bit_errors: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SuspectedChip {
    pub controller_id: u32,
    pub chip_half: ChipHalf,
    pub chip_label: String,
    pub error_count: u64,
    pub confidence: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ChipHalf {
    Lower16, // bits 0-15 → typically front-side chip
    Upper16, // bits 16-31 → typically back-side chip
    Both,
}

impl std::fmt::Display for ChipHalf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChipHalf::Lower16 => write!(f, "Front (lower 16-bit)"),
            ChipHalf::Upper16 => write!(f, "Back (upper 16-bit)"),
            ChipHalf::Both => write!(f, "Both chips"),
        }
    }
}

/// Analyze error entries to identify which memory controller(s) and chip(s) are faulty.
pub fn analyze_errors(errors: &[ErrorEntry], num_controllers: u32) -> ChipAnalysis {
    let interleave_bytes = DEFAULT_INTERLEAVE_BYTES;
    let total_stride = interleave_bytes * num_controllers as u64;

    let mut controller_results: Vec<ControllerResult> = (0..num_controllers)
        .map(|id| ControllerResult {
            controller_id: id,
            error_count: 0,
            upper_16bit_errors: 0,
            lower_16bit_errors: 0,
        })
        .collect();

    for err in errors {
        // Determine which controller this address belongs to
        let position_in_stride = err.byte_offset % total_stride;
        let controller_id = (position_in_stride / interleave_bytes) as u32;

        if (controller_id as usize) < controller_results.len() {
            let cr = &mut controller_results[controller_id as usize];
            cr.error_count += 1;

            // Analyze which bits differ to identify upper/lower chip
            let diff = err.expected ^ err.actual;
            if diff & 0x0000FFFF != 0 {
                cr.lower_16bit_errors += 1;
            }
            if diff & 0xFFFF0000 != 0 {
                cr.upper_16bit_errors += 1;
            }
        }
    }

    // Identify suspected bad chips
    let total_errors: u64 = controller_results.iter().map(|c| c.error_count).sum();
    let mut suspected_bad_chips = Vec::new();

    for cr in &controller_results {
        if cr.error_count == 0 {
            continue;
        }

        let error_ratio = cr.error_count as f64 / total_errors.max(1) as f64;

        let chip_half = if cr.lower_16bit_errors > 0 && cr.upper_16bit_errors == 0 {
            ChipHalf::Lower16
        } else if cr.upper_16bit_errors > 0 && cr.lower_16bit_errors == 0 {
            ChipHalf::Upper16
        } else {
            ChipHalf::Both
        };

        let (chip_label, _chip_index) = match chip_half {
            ChipHalf::Lower16 => (
                format!("Chip #{} (front)", cr.controller_id),
                cr.controller_id,
            ),
            ChipHalf::Upper16 => (
                format!("Chip #{} (back)", cr.controller_id + num_controllers),
                cr.controller_id + num_controllers,
            ),
            ChipHalf::Both => (
                format!("Controller {} (both chips)", cr.controller_id),
                cr.controller_id,
            ),
        };

        suspected_bad_chips.push(SuspectedChip {
            controller_id: cr.controller_id,
            chip_half,
            chip_label,
            error_count: cr.error_count,
            confidence: (error_ratio * 100.0).min(99.9),
        });
    }

    // Sort by error count descending
    suspected_bad_chips.sort_by(|a, b| b.error_count.cmp(&a.error_count));

    ChipAnalysis {
        num_controllers,
        interleave_bytes,
        controller_results,
        suspected_bad_chips,
    }
}
