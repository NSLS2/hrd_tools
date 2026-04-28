# FDR Design Scripts

Analysis & design scripts that produced the figures for the NSLS-II HRD
**Final Design Review**.  All scripts read their blessed beamline parameters
from a single YAML file and share a common CLI.

## Layout

| File | Purpose |
| --- | --- |
| `beamline_params.yaml` | Single source of truth for blessed FDR parameters (energy, geometry, scan, layout, real-data paths). Field names mirror the dataclasses in `hrd_tools.config`. |
| `rois_11bm.yaml` | 12-detector ROI block extracted from the 11-BM-B reduction; loaded via `multihead.config.DetectorROIs.from_yaml`. |
| `_fdr_params.py` | Shared loader + uniform CLI helpers (not installed; sits next to the scripts and imported by relative path). |

Detector geometry (pitch, sensor shape) is **not** duplicated in the YAML —
`detector.name` is a key into `hrd_tools.detector_stats.detectors` and the
loader expands it into the `DetectorConfig` fields at load time.

## Uniform CLI

Every script in this directory accepts:

```
--outdir DIR          # where to write figures (default ./figures)
--dpi N               # figure DPI (default 300)
--show / --no-show    # interactive plt.show() (default --show)
--energy-kev E        # override source.E_incident (in keV)
--params-file YAML    # override path to beamline_params.yaml
```

For automated runs:

```bash
pixi run -e xrt python design_scripts/FDR/<script>.py --no-show --outdir /tmp/fdr_figs
```

## Scripts

Sorted by topic.

### Geometry & layout

| Script | Description | Outputs |
| --- | --- | --- |
| `detector_layout.py` | Schematic of the analyzer-arm layout (banks of crystals × N) and a grid of variants. | `layout.png`, `layout_grid.png`, `angle_to_measure.png` |
| `detector_distance.py` | Sensitivity to the crystal–detector distance `Rd` and arm distance `R`. | (per-cell `*.png` saves) |
| `min_crystal_distance.py` | Minimum allowable crystal-to-crystal spacing. | per-cell figures |
| `concentricity.py` | Required mounting concentricity vs. gauge-volume width. | per-cell figures |
| `off_center.py` | Effect of off-center sample on the measured signal. | per-cell figures |
| `geometric_strip.py` | Geometric strip cross-section through the gauge volume. | per-cell figures |

### Detector coverage

| Script | Description | Outputs |
| --- | --- | --- |
| `phi_coverage.py` | Effective $\pm\phi_\text{max}$ vs. $2\theta$ for each commercial detector (simple geometry). | `phi_coverage_simple.png` |
| `phi_coverage2.py` | Same, using the full `multihead.corrections.arm_from_z`; and crystal-position sweep. | `phi_coverage_full.png`, `phi_coverage_crystal_position.png` |
| `scan_coverage.py` | Total $2\theta$ coverage swept by the arm given the blessed bank/crystal layout. | `scan_coverage.png` |
| `det_rates.py` | Per-detector count-rate estimates. | per-cell figures |

### Sensitivity studies

These wrap `hrd_tools.sensitivity` and report how reduced patterns drift
with small misalignments / parameter changes.

| Script | Description |
| --- | --- |
| `chi_sensativity.py` | Sensitivity to crystal $\chi$ (roll). |
| `theta_d_sensativity.py` | Sensitivity to $\theta_d$ (detector arm angle). |
| `mac_sensitivity.py` | Sensitivity to mass-absorption-coefficient assumptions. |
| `crystal_t_sensativity.py` | Sensitivity to crystal thickness. |
| `diff_roll.py` | Differential roll sensitivity for the design divergence. |

### Crystal & gauge volume

| Script | Description |
| --- | --- |
| `crystal_footprint.py` | Footprint of the beam on the crystal vs. $2\theta$. |
| `gague_volume.py` | Gauge volume size & shape. |
| `guage_crosssection.py` | Cross-section through the gauge volume at various depths. |
| `delta_L_angle.py` | $\Delta L$ vs. crystal angle. |
| `corrections_scales.py` | Scale of the various geometric corrections. |
| `inherent_correction_error.py` | Residual error of the analytic correction model. |

### Real-data validation (11-BM-B)

These read the real beamtime data described in the YAML's `real_data:`
block; paths are local to the developer's machine and the scripts will
refuse to run if the data isn't present.

| Script | Description | Outputs |
| --- | --- | --- |
| `real_data.py` | Through-peak frames, ROI vs. point-detector sums, kymograph + zooms, cosmic-ray example. | `through_peak.png`, `detector_sum.png`, `khymo_full.png`, `cosmic.png` |
| `live_data_varidation.py` | Streaming validation of the chunked summing pipeline. | `live_data_validation.png` |

## Running the tests

The smoke tests (`tests/test_fdr_scripts_smoke.py`) verify that every script
in this directory responds to `--help` with exit 0; the loader tests
(`tests/test_fdr_params.py`) verify the YAML round-trip and the
`_fdr_params` helpers.

```bash
pixi run -e test test
```
