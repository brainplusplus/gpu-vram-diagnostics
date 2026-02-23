/// VRAM test pattern definitions.
use rand::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum TestPattern {
    AllZero,
    AllOne,
    Walking1(u8), // bit position 0-31
    Walking0(u8), // bit position 0-31
    Checkerboard,
    InverseCheckerboard,
    Random(u32),  // seed → pattern value
}

impl TestPattern {
    /// Get the 32-bit fill value for this pattern.
    pub fn as_u32(&self) -> u32 {
        match self {
            TestPattern::AllZero => 0x00000000,
            TestPattern::AllOne => 0xFFFFFFFF,
            TestPattern::Walking1(bit) => 1u32 << (bit % 32),
            TestPattern::Walking0(bit) => !(1u32 << (bit % 32)),
            TestPattern::Checkerboard => 0x55555555,
            TestPattern::InverseCheckerboard => 0xAAAAAAAA,
            TestPattern::Random(seed) => *seed,
        }
    }

    /// Returns the basic test patterns (5 patterns for quick mode).
    pub fn basic_patterns() -> Vec<TestPattern> {
        let mut rng = rand::thread_rng();
        vec![
            TestPattern::AllZero,
            TestPattern::AllOne,
            TestPattern::Walking1(0),
            TestPattern::Walking0(0),
            TestPattern::Checkerboard,
            TestPattern::InverseCheckerboard,
            TestPattern::Random(rng.gen()),
        ]
    }

    /// Display name for the pattern.
    pub fn name(&self) -> String {
        match self {
            TestPattern::AllZero => "All Zero (0x00)".to_string(),
            TestPattern::AllOne => "All One (0xFF)".to_string(),
            TestPattern::Walking1(b) => format!("Walking 1 (bit {})", b),
            TestPattern::Walking0(b) => format!("Walking 0 (bit {})", b),
            TestPattern::Checkerboard => "Checkerboard (0x55)".to_string(),
            TestPattern::InverseCheckerboard => "Inv Checkerboard (0xAA)".to_string(),
            TestPattern::Random(v) => format!("Random (0x{:08X})", v),
        }
    }

    /// Short name for GUI grid.
    pub fn short_name(&self) -> &'static str {
        match self {
            TestPattern::AllZero => "All Zero",
            TestPattern::AllOne => "All One",
            TestPattern::Walking1(_) => "Walking 1",
            TestPattern::Walking0(_) => "Walking 0",
            TestPattern::Checkerboard => "Checkerboard",
            TestPattern::InverseCheckerboard => "Inv Checkerboard",
            TestPattern::Random(_) => "Random",
        }
    }
}
