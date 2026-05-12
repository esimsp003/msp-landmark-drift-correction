# MSP Landmark Drift Correction Tool
User-driven rigid body alignment for longitudinal confocal microscopy `.ims` files.

Built to replace poorly-performing automated drift correction on sparse plaque images, where full-volume phase correlation locks onto noise and produces useless results. The user clicks corresponding landmarks across timepoints; the tool fits a rigid body transform via SVD/Procrustes, runs a landmark-guided residual refinement pass, and writes a corrected `.ims` file with rebuilt resolution pyramid.

Originally developed for the NAPS-03 study (Eisai, Neurology Discovery – Translational Models) tracking amyloid plaques across D0/D7/D14/D21/D28 timepoints.

---

## Why this exists

Automated drift correction on longitudinal 2-photon plaque imaging fails reliably in two modes:

1. **Phase correlation on full volumes** locks onto repeating background structure or noise, reporting nonsense displacements (e.g. 297-voxel residuals on data that visually drifts by <30 voxels).
2. **Feature-based methods** built for cells, vessels, or dense structure don't work on the sparse, sub-resolution plaques that are the entire point of the study.

A human can pick three or four matching plaques across timepoints in under a minute. That's the design point of this tool: keep the human in the loop for the part computers are bad at (correspondence on sparse structure), and let the math handle the rest.

---

## Features

- **Interactive landmark picking** across all timepoints with XY MIP + XZ MIP per panel
- **Magnifier** synced to cursor, with toggle/minimize
- **Auto-suggest** and **auto-place** for batch starting points
- **Refine** snap to re-center landmarks on local intensity peaks
- **Iterative outlier rejection** in the rigid-body solve
- **Landmark-guided residual phase correlation** — runs per-cube around each landmark, takes the median shift; falls back to cropped full-volume PC if cubes are too dim
- **Three output canvas modes** — include all data / preserve original size / crop to common overlap
- **Live RMSE feedback** with quality color and outlier preview as you place landmarks
- **Single-file or batch mode** with elapsed time and ETA tracking
- **Skip per-timepoint** (any timepoint, including reference T0)
- **Multi-channel display cycling** for `.ims` files with ≤4 channels
- **Resolution pyramid rebuild** with correct HDF5 attribute handling so Imaris opens the result cleanly

---

## Requirements

- Windows 10/11 (developed on `ERICX-IMARIS03`)
- Python 3.8+
- Imaris `.ims` files (HDF5-based)

```
numpy
scipy
scikit-image
h5py
matplotlib
tifffile
```

The three thumbnail PNGs (`Include_entire_result.png`, `New_size_equal_to_current_size.png`, `Crop_largest_common_region.png`) must live in the same folder as the script.

---

## Usage

```bash
python landmark_drift_correction_v30.py
```

You'll be prompted to choose:

1. **Single file** — pick one `.ims` file
2. **Batch** — point at a folder; every `.ims` inside is processed in order

Then pick an output folder and reference channel. The interactive landmark UI opens for each file.

### Workflow in the UI

1. Click a plaque (or other distinctive feature) in the **D0** panel
2. The yellow instruction bar advances; click the **same** plaque in D7, D14, D21, D28
3. After 3+ landmarks are placed, the predicted RMSE appears with a color-coded quality indicator
4. Place at least 3 landmarks (4+ recommended) for a stable solve
5. Choose your output canvas mode from the three thumbnails
6. Click **Complete** (single) / **Done File** (batch)

### Keyboard shortcuts

```
PLACING LANDMARKS
  Left click          Place landmark at peak nearest click
  Shift + click       Remove and re-place that specific marker
  Ctrl + Z            Undo most recent landmark
  D                   Done — apply alignment and save

VIEW
  Scroll              Zoom (synced across timepoints)
  Right-click drag    Pan (synced across timepoints)
  R                   Reset zoom to fit
  M                   Toggle magnifier
  [ / ]               Adjust contrast
  Z                   Toggle MIP / single-slice mode
  , / .               Previous / next Z slice
  C                   Cycle display channels

WORKFLOW
  S                   Skip current timepoint
  ?                   Show keyboard shortcuts overlay
```

---

## Pipeline

```
.ims file
    │
    ▼
Read HDF5 → extract per-timepoint volumes (reference channel)
    │
    ▼
Interactive landmark picking
    │
    ▼
Rigid body fit per timepoint (SVD / Procrustes)
    └── iterative outlier rejection (residual > 2× median AND > 15 vox)
    │
    ▼
Landmark-guided residual phase correlation
    └── (25, 60, 60) vox cubes around each landmark → median shift
    │
    ▼
Compute output canvas (3 modes)
    │
    ▼
Apply transform to all channels, place on canvas
    │
    ▼
Rebuild resolution pyramid (factor-2 downsampling per level)
    │
    ▼
Write .ims with corrected HDF5 attributes:
    DataSetInfo/Image (X, Y, Z, ExtMin/Max)
    DataSet/.../Channel/ImageSize{X,Y,Z} per resolution level
    DataSetInfo/Dimension/NumberOfElements
```

---

## Output canvas modes

| Mode | Behavior | Use when |
|---|---|---|
| **Include entire result** | Expand canvas to fit all timepoints, 50% Z + 20% XY padding | You want zero data loss and don't mind black borders |
| **New size equal to current size** | Keep original dimensions, no expansion, data outside the box is clipped | Downstream pipeline assumes original geometry |
| **Crop largest common region** *(default)* | Smallest box at the intersection of all aligned timepoints | You want only valid overlapping tissue and no black anywhere |

---

## Output files

For each `input.ims`, the tool writes:

- `input_aligned.ims` — corrected hyperstack with rebuilt pyramid
- `input_aligned_report.html` — visual QC report with before/after MIPs and per-timepoint transform parameters
- `landmark_correction_log_<timestamp>.txt` — full session log (shared across all files in a batch)

---

## Known limitations

- **Rigid body only** (rotation + translation). No scaling, no affine, no deformable. Tissue compression or expansion between timepoints will not be corrected.
- **Out-of-plane rotation is capped at 2°** (`max_tilt` hardcoded). In-plane rotation cap is user-adjustable (default 25°). These reflect realistic 2-photon imaging geometry — wider caps tend to overfit on bad landmark sets.
- **Single reference channel** for alignment. Other channels are transformed with the same matrix but do not participate in the solve.
- **Sparse landmark sets cluster easily.** The tool warns if landmarks span <10% of any image dimension, but won't refuse to run.

---

## Citation

If this tool contributes to a published result, please cite the repository:

```
St. Pierre, M. (2026). MSP Landmark Drift Correction Tool.
https://github.com/MSP-003/msp-landmark-drift-correction
```

---

## Contact

Mark St. Pierre — [markstpierre.com](https://markstpierre.com)
Issues and feature requests via the GitHub issue tracker.
