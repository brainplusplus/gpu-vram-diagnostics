/// egui GUI — CrystalDiskMark-inspired VRAM diagnostic interface.
use crate::chip_map;
use crate::cuda::device::CudaDevice;
use crate::patterns::TestPattern;
use crate::progress::{self, SharedProgress};
use crate::report::ScanReport;
use crate::scanner;
use eframe::egui;

use std::thread;

const GREEN: egui::Color32 = egui::Color32::from_rgb(76, 175, 80);
const RED: egui::Color32 = egui::Color32::from_rgb(244, 67, 54);
const YELLOW: egui::Color32 = egui::Color32::from_rgb(255, 193, 7);
const DARK_BG: egui::Color32 = egui::Color32::from_rgb(30, 30, 30);
const PANEL_BG: egui::Color32 = egui::Color32::from_rgb(40, 40, 40);
const HEADER_BG: egui::Color32 = egui::Color32::from_rgb(46, 125, 50);

pub struct VramDiagApp {
    devices: Vec<CudaDevice>,
    selected_device: usize,
    chunk_sizes: Vec<u64>,
    selected_chunk: usize,
    passes_options: Vec<u32>,
    selected_passes: usize,
    progress: SharedProgress,
    scan_thread: Option<thread::JoinHandle<()>>,
    json_report: Option<String>,
    show_chip_map: bool,
    initialized: bool,
    init_error: Option<String>,
}

impl VramDiagApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        // Try to initialize CUDA
        let (devices, init_error) = match crate::cuda::ffi::cuda_init() {
            Ok(()) => match CudaDevice::enumerate() {
                Ok(devs) if devs.is_empty() => {
                    (vec![], Some("No NVIDIA GPUs found.".to_string()))
                }
                Ok(devs) => (devs, None),
                Err(e) => (vec![], Some(format!("GPU enumeration failed: {}", e))),
            },
            Err(e) => (vec![], Some(format!("CUDA init failed: {}", e))),
        };

        let num_controllers = devices
            .first()
            .map(|d| d.memory_controller_count() as u32)
            .unwrap_or(12);

        VramDiagApp {
            devices,
            selected_device: 0,
            chunk_sizes: vec![64, 128, 256, 512, 1024],
            selected_chunk: 3, // 512 MB default
            passes_options: vec![1, 2, 3, 5],
            selected_passes: 1, // 2 passes default
            progress: progress::create_shared_progress(num_controllers),
            scan_thread: None,
            json_report: None,
            show_chip_map: false,
            initialized: init_error.is_none(),
            init_error,
        }
    }

    fn is_running(&self) -> bool {
        self.progress.lock().unwrap().running
    }

    fn start_scan(&mut self) {
        if self.devices.is_empty() {
            return;
        }

        let device = self.devices[self.selected_device].clone();
        let chunk_size = (self.chunk_sizes[self.selected_chunk] * 1024 * 1024) as usize;
        let passes = self.passes_options[self.selected_passes];
        let micro_batch_size = 32 * 1024 * 1024; // 32 MB
        let headroom = 512 * 1024 * 1024; // 512 MB

        // Reset progress
        let num_controllers = device.memory_controller_count() as u32;
        self.progress = progress::create_shared_progress(num_controllers);
        self.json_report = None;
        self.show_chip_map = false;

        let progress = self.progress.clone();
        let handle = thread::spawn(move || {
            let _ = scanner::run_scan(
                &device,
                chunk_size,
                micro_batch_size,
                passes,
                headroom,
                progress,
            );
        });
        self.scan_thread = Some(handle);
    }

    fn stop_scan(&self) {
        let mut p = self.progress.lock().unwrap();
        p.running = false;
    }
}

impl eframe::App for VramDiagApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Request repaint while scanning for live updates
        if self.is_running() {
            ctx.request_repaint();
        }

        // Dark theme
        let mut style = (*ctx.style()).clone();
        style.visuals = egui::Visuals::dark();
        style.visuals.panel_fill = DARK_BG;
        style.visuals.window_fill = DARK_BG;
        ctx.set_style(style);

        // Top panel — controls
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.add_space(6.0);
            
            // ROW 1: GPU Selection
            ui.horizontal(|ui| {
                ui.add_space(8.0);
                ui.label(egui::RichText::new("GPU:").strong().color(egui::Color32::WHITE));
                let device_label = if self.devices.is_empty() {
                    "No GPU".to_string()
                } else {
                    self.devices[self.selected_device].display_label()
                };
                let enabled = !self.is_running() && !self.devices.is_empty();
                ui.add_enabled_ui(enabled, |ui| {
                    egui::ComboBox::from_id_salt("gpu_select")
                        .selected_text(&device_label)
                        // Make this combo fill the available width minus padding
                        .width(ui.available_width() - 16.0) 
                        .show_ui(ui, |ui| {
                            for (i, dev) in self.devices.iter().enumerate() {
                                ui.selectable_value(
                                    &mut self.selected_device,
                                    i,
                                    dev.display_label(),
                                );
                            }
                        });
                });
            });

            ui.add_space(8.0);

            // ROW 2: Config and Start Button
            ui.horizontal(|ui| {
                ui.add_space(8.0);
                
                let enabled = !self.is_running();

                // Chunk size
                ui.label(egui::RichText::new("Chunk:").color(egui::Color32::WHITE));
                ui.add_enabled_ui(enabled, |ui| {
                    egui::ComboBox::from_id_salt("chunk_select")
                        .selected_text(format!("{} MB", self.chunk_sizes[self.selected_chunk]))
                        .width(70.0)
                        .show_ui(ui, |ui| {
                            for (i, size) in self.chunk_sizes.iter().enumerate() {
                                ui.selectable_value(
                                    &mut self.selected_chunk,
                                    i,
                                    format!("{} MB", size),
                                );
                            }
                        });
                });

                ui.add_space(16.0);

                // Passes
                ui.label(egui::RichText::new("Passes:").color(egui::Color32::WHITE));
                ui.add_enabled_ui(enabled, |ui| {
                    egui::ComboBox::from_id_salt("passes_select")
                        .selected_text(format!("{}", self.passes_options[self.selected_passes]))
                        .width(50.0)
                        .show_ui(ui, |ui| {
                            for (i, &p) in self.passes_options.iter().enumerate() {
                                ui.selectable_value(
                                    &mut self.selected_passes,
                                    i,
                                    format!("{}", p),
                                );
                            }
                        });
                });

                // Start / Stop Button right-aligned
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_space(8.0);
                    
                    if self.is_running() {
                        if ui
                            .add(
                                egui::Button::new(
                                    egui::RichText::new("⏹ STOP")
                                        .size(18.0)
                                        .strong()
                                        .color(egui::Color32::WHITE),
                                )
                                .fill(RED)
                                .min_size(egui::vec2(120.0, 36.0)),
                            )
                            .clicked()
                        {
                            self.stop_scan();
                        }
                    } else if ui
                        .add(
                            egui::Button::new(
                                egui::RichText::new("▶ START")
                                    .size(18.0)
                                    .strong()
                                    .color(egui::Color32::WHITE),
                            )
                            .fill(GREEN)
                            .min_size(egui::vec2(120.0, 36.0)),
                        )
                        .clicked()
                    {
                        self.start_scan();
                    }
                });
            });
            ui.add_space(6.0);
        });

        // Bottom panel — status + progress
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.add_space(4.0);
            let p = self.progress.lock().unwrap();

            ui.horizontal(|ui| {
                ui.add_space(8.0);

                // Status text
                let status_color = if p.error_message.is_some() {
                    RED
                } else if p.total_errors > 0 {
                    YELLOW
                } else {
                    GREEN
                };
                ui.label(
                    egui::RichText::new(&p.status)
                        .color(status_color)
                        .size(13.0),
                );

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_space(8.0);
                    if p.elapsed_secs > 0.0 {
                        ui.label(
                            egui::RichText::new(format!("{:.1}s", p.elapsed_secs))
                                .color(egui::Color32::GRAY)
                                .size(12.0),
                        );
                    }
                });
            });

            // Progress bar
            ui.add_space(2.0);
            let bar_color = if p.total_errors > 0 { RED } else { GREEN };
            let progress_bar = egui::ProgressBar::new(p.progress)
                .fill(bar_color)
                .animate(p.running);
            ui.add(progress_bar);
            ui.add_space(4.0);
        });

        // Central panel — results grid + chip map
        egui::CentralPanel::default().show(ctx, |ui| {
            let p = self.progress.lock().unwrap().clone();

            // Show init error if any
            if let Some(ref err) = self.init_error {
                ui.add_space(20.0);
                ui.centered_and_justified(|ui| {
                    ui.label(
                        egui::RichText::new(format!("⚠ {}", err))
                            .color(RED)
                            .size(18.0),
                    );
                });
                return;
            }

            ui.add_space(8.0);

            // Results grid (CrystalDiskMark style)
            egui::Frame::default()
                .fill(PANEL_BG)
                .corner_radius(egui::CornerRadius::same(6))
                .inner_margin(egui::Margin::same(12))
                .show(ui, |ui| {
                    egui::Grid::new("results_grid")
                        .num_columns(3)
                        .spacing([40.0, 16.0])
                        .min_col_width(120.0)
                        .striped(true)
                        .show(ui, |ui| {
                            // Header
                            ui.label(
                                egui::RichText::new("Test Pattern")
                                    .strong()
                                    .size(14.0)
                                    .color(GREEN),
                            );
                            ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                                ui.label(
                                    egui::RichText::new("Errors")
                                        .strong()
                                        .size(14.0)
                                        .color(GREEN),
                                );
                            });
                            ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                                ui.label(
                                    egui::RichText::new("Speed (MB/s)")
                                        .strong()
                                        .size(14.0)
                                        .color(GREEN),
                                );
                            });
                            ui.end_row();

                            // Pattern rows
                            let patterns = TestPattern::basic_patterns();
                            for (i, pat) in patterns.iter().enumerate() {
                                // Pattern name
                                ui.label(
                                    egui::RichText::new(pat.short_name())
                                        .size(15.0)
                                        .color(egui::Color32::WHITE),
                                );

                                // Errors
                                let (errors, speed) = if let Some(summary) = p.pattern_summaries.get(i) {
                                    (summary.total_errors, summary.avg_speed_mbps)
                                } else {
                                    (0, 0.0)
                                };

                                let error_color = if errors > 0 { RED } else { GREEN };
                                
                                ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                                    ui.label(
                                        egui::RichText::new(format!("{}", errors))
                                            .size(20.0)
                                            .strong()
                                            .color(error_color),
                                    );
                                });

                                // Speed
                                ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                                    ui.label(
                                        egui::RichText::new(format!("{:.1}", speed))
                                            .size(20.0)
                                            .strong()
                                            .color(egui::Color32::WHITE),
                                    );
                                });
                                ui.end_row();
                            }
                        });
                });

            ui.add_space(12.0);

            // Total summary
            ui.horizontal(|ui| {
                ui.add_space(12.0);
                ui.label(
                    egui::RichText::new(format!(
                        "Total Tested: {:.1} GB",
                        p.total_bytes_tested as f64 / 1_073_741_824.0
                    ))
                    .size(13.0)
                    .color(egui::Color32::LIGHT_GRAY),
                );
                ui.separator();
                let total_color = if p.total_errors > 0 { RED } else { GREEN };
                ui.label(
                    egui::RichText::new(format!("Total Errors: {}", p.total_errors))
                        .size(13.0)
                        .strong()
                        .color(total_color),
                );

                if p.finished {
                    ui.separator();
                    if ui
                        .button(egui::RichText::new("📋 Chip Map").size(13.0))
                        .clicked()
                    {
                        self.show_chip_map = !self.show_chip_map;
                    }
                    if ui
                        .button(egui::RichText::new("💾 Save JSON").size(13.0))
                        .clicked()
                    {
                        self.save_json_report();
                    }
                }
            });

            // Chip map panel
            if self.show_chip_map && p.finished {
                ui.add_space(8.0);
                self.draw_chip_map(ui, &p);
            }
        });
    }
}

impl VramDiagApp {
    fn draw_chip_map(&self, ui: &mut egui::Ui, progress: &crate::progress::ScanProgress) {
        let num_controllers = progress.num_controllers as usize;
        if num_controllers == 0 {
            return;
        }

        egui::Frame::default()
            .fill(PANEL_BG)
            .corner_radius(egui::CornerRadius::same(6))
            .inner_margin(egui::Margin::same(12))
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new("VRAM Chip Status")
                        .strong()
                        .size(14.0)
                        .color(GREEN),
                );
                ui.add_space(6.0);

                // Analyze errors
                let analysis =
                    chip_map::analyze_errors(&progress.all_errors, progress.num_controllers);

                // Front side chips
                ui.label(
                    egui::RichText::new("Front Side:")
                        .size(12.0)
                        .color(egui::Color32::GRAY),
                );
                ui.horizontal_wrapped(|ui| {
                    for i in 0..num_controllers {
                        let errs = analysis.controller_results[i].lower_16bit_errors;
                        let color = chip_color(errs);
                        let label = format!("C{}", i);
                        let btn = egui::Button::new(
                            egui::RichText::new(&label)
                                .size(12.0)
                                .color(egui::Color32::WHITE),
                        )
                        .fill(color)
                        .min_size(egui::vec2(36.0, 28.0));
                        let response = ui.add(btn);
                        if errs > 0 {
                            response.on_hover_text(format!(
                                "Controller {} — Front chip\n{} errors (lower 16-bit)",
                                i, errs
                            ));
                        }
                    }
                });

                ui.add_space(4.0);

                // Back side chips
                ui.label(
                    egui::RichText::new("Back Side:")
                        .size(12.0)
                        .color(egui::Color32::GRAY),
                );
                ui.horizontal_wrapped(|ui| {
                    for i in 0..num_controllers {
                        let errs = analysis.controller_results[i].upper_16bit_errors;
                        let color = chip_color(errs);
                        let label = format!("C{}", i);
                        let btn = egui::Button::new(
                            egui::RichText::new(&label)
                                .size(12.0)
                                .color(egui::Color32::WHITE),
                        )
                        .fill(color)
                        .min_size(egui::vec2(36.0, 28.0));
                        let response = ui.add(btn);
                        if errs > 0 {
                            response.on_hover_text(format!(
                                "Controller {} — Back chip\n{} errors (upper 16-bit)",
                                i, errs
                            ));
                        }
                    }
                });

                // Suspected bad chips summary
                if !analysis.suspected_bad_chips.is_empty() {
                    ui.add_space(8.0);
                    ui.separator();
                    ui.label(
                        egui::RichText::new("Suspected Faulty Chips:")
                            .strong()
                            .size(13.0)
                            .color(RED),
                    );
                    for chip in &analysis.suspected_bad_chips {
                        ui.label(
                            egui::RichText::new(format!(
                                "⚠ {} — {} errors",
                                chip.chip_label, chip.error_count
                            ))
                            .size(12.0)
                            .color(YELLOW),
                        );
                    }
                } else if progress.total_errors == 0 {
                    ui.add_space(6.0);
                    ui.label(
                        egui::RichText::new("✅ All chips healthy — no errors detected")
                            .size(13.0)
                            .color(GREEN),
                    );
                }
            });
    }

    fn save_json_report(&mut self) {
        let p = self.progress.lock().unwrap().clone();
        let device = &self.devices[self.selected_device];
        let chunk_size = self.chunk_sizes[self.selected_chunk] * 1024 * 1024;
        let report = ScanReport::from_progress(
            &p,
            &device.name,
            device.ordinal,
            device.total_memory as u64,
            chunk_size,
        );

        let json = report.to_json();
        let filename = format!(
            "vram_report_{}.json",
            chrono::Local::now().format("%Y%m%d_%H%M%S")
        );

        match std::fs::write(&filename, &json) {
            Ok(()) => {
                self.json_report = Some(format!("Saved: {}", filename));
            }
            Err(e) => {
                self.json_report = Some(format!("Save failed: {}", e));
            }
        }
    }
}

fn chip_color(errors: u64) -> egui::Color32 {
    if errors == 0 {
        egui::Color32::from_rgb(46, 125, 50) // green
    } else if errors < 100 {
        egui::Color32::from_rgb(255, 152, 0) // orange
    } else {
        egui::Color32::from_rgb(211, 47, 47) // red
    }
}

pub fn run_gui() -> eframe::Result<()> {
    let icon_data = {
        let image = image::load_from_memory(include_bytes!("../assets/icon.png"))
            .expect("Failed to load window icon")
            .into_rgba8();
        let (width, height) = image.dimensions();
        egui::IconData {
            rgba: image.into_raw(),
            width,
            height,
        }
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_icon(icon_data)
            .with_inner_size([600.0, 560.0])
            .with_min_inner_size([540.0, 480.0])
            .with_title("VRAM Diagnostics v0.1.0"),
        ..Default::default()
    };

    eframe::run_native(
        "VRAM Diagnostics",
        options,
        Box::new(|cc| Ok(Box::new(VramDiagApp::new(cc)))),
    )
}
