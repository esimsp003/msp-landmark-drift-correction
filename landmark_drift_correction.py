#!/usr/bin/env python3
"""
MSP Landmark Drift Correction Tool
====================================
Interactive rigid body drift correction for longitudinal confocal imaging.

Features:
  - Per-landmark color coding, synchronized zoom, adjustable contrast
  - Auto-suggest landmark positions from brightest spots
  - Confidence overlay showing match quality per landmark
  - Before/after alignment preview
  - Rotation lock slider (adjustable in-app)
  - Skip timepoints, skip files, batch processing with HTML report

Requirements:
    pip install h5py numpy scipy matplotlib scikit-image

Usage:
    python landmark_drift_correction.py                    # Interactive launcher
    python landmark_drift_correction.py -i <folder> -o <folder>          # Batch
    python landmark_drift_correction.py -i <folder> -o <folder> --file m03_pos1
"""

import os, sys, shutil, argparse, traceback, re
from pathlib import Path
from datetime import datetime

try:
    import h5py
    import numpy as np
    from scipy.ndimage import affine_transform
    from skimage.registration import phase_cross_correlation
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"\nMissing: {e}\npip install h5py numpy scipy matplotlib scikit-image"); sys.exit(1)


class Logger:
    def __init__(self, path):
        self.path = path; self.lines = []
    def log(self, msg=""):
        print(msg); self.lines.append(msg)
    def save(self):
        with open(self.path, "w", encoding="utf-8") as f: f.write("\n".join(self.lines))


# ── IMS I/O ──────────────────────────────────────────────────

def get_ims_info(h5f):
    ds = h5f["DataSet/ResolutionLevel 0"]
    tp = 0
    while f"TimePoint {tp}" in ds: tp += 1
    tp0 = ds["TimePoint 0"]; ch = 0
    while f"Channel {ch}" in tp0: ch += 1
    data = tp0["Channel 0/Data"]
    n_res = 0; dset = h5f["DataSet"]
    while f"ResolutionLevel {n_res}" in dset: n_res += 1
    return {"n_tp": tp, "n_ch": ch, "shape": data.shape, "dtype": data.dtype, "n_res": n_res}

def read_vol(h5f, t, c):
    p = f"DataSet/ResolutionLevel 0/TimePoint {t}/Channel {c}/Data"
    d = h5f.get(p); return d[...] if d else None

def write_vol(h5f, t, c, data):
    p = f"DataSet/ResolutionLevel 0/TimePoint {t}/Channel {c}/Data"
    if p in h5f: del h5f[p]
    h5f.create_dataset(p, data=data, chunks=True)

def write_ims_attr(group, name, value_str):
    """Write attribute as array of single bytes (|S1) — the IMS format."""
    value_str = str(value_str)
    byte_array = np.array([c.encode('utf-8') for c in value_str], dtype='|S1')
    if name in group.attrs:
        del group.attrs[name]
    group.attrs.create(name, byte_array)


def rebuild_pyramid(h5f, n_tp, n_ch, new_shape):
    dset = h5f["DataSet"]; n_res = 0
    while f"ResolutionLevel {n_res}" in dset: n_res += 1

    nz, ny, nx = new_shape

    # Update ImageSizeX/Y/Z on ResolutionLevel 0 channels
    for t in range(n_tp):
        for c in range(n_ch):
            ch_group = h5f.get(f"DataSet/ResolutionLevel 0/TimePoint {t}/Channel {c}")
            if ch_group is not None:
                write_ims_attr(ch_group, "ImageSizeX", nx)
                write_ims_attr(ch_group, "ImageSizeY", ny)
                write_ims_attr(ch_group, "ImageSizeZ", nz)

    # Rebuild lower resolution levels
    for res in range(1, n_res):
        f = 2 ** res
        ds_z, ds_y, ds_x = nz // f, ny // f, nx // f  # Downsampled size
        for t in range(n_tp):
            for c in range(n_ch):
                fp = f"DataSet/ResolutionLevel 0/TimePoint {t}/Channel {c}/Data"
                dp = f"DataSet/ResolutionLevel {res}/TimePoint {t}/Channel {c}/Data"
                if dp in h5f: del h5f[dp]
                parent = dp.rsplit("/Data", 1)[0]
                if parent not in h5f: h5f.create_group(parent)
                h5f.create_dataset(dp, data=h5f[fp][...][::f, ::f, ::f], chunks=True)

                # Update ImageSizeX/Y/Z on pyramid channel groups too
                ch_group = h5f.get(parent)
                if ch_group is not None:
                    write_ims_attr(ch_group, "ImageSizeX", ds_x)
                    write_ims_attr(ch_group, "ImageSizeY", ds_y)
                    write_ims_attr(ch_group, "ImageSizeZ", ds_z)

def update_dims(h5f, shape, orig_shape=None, base_offset=None):
    """Update dimension metadata and physical extents for expanded canvas."""
    nz, ny, nx = shape
    img = h5f.get("DataSetInfo/Image")
    if img is None:
        return

    def get_attr_str(name):
        val = img.attrs.get(name, None)
        if val is None: return None
        if isinstance(val, np.ndarray) and val.dtype.kind == 'S':
            return b"".join(val).decode("utf-8", errors="replace")
        elif isinstance(val, bytes):
            return val.decode("utf-8", errors="replace")
        return str(val)

    def write_ims_attr(name, value_str):
        """Write attribute as array of single bytes (|S1) — the IMS format."""
        value_str = str(value_str)
        byte_array = np.array([c.encode('utf-8') for c in value_str], dtype='|S1')
        if name in img.attrs:
            del img.attrs[name]
        img.attrs.create(name, byte_array)

    # Read original metadata dimensions BEFORE overwriting
    orig_meta = {}
    for key in ["X", "Y", "Z"]:
        val = get_attr_str(key)
        if val:
            try: orig_meta[key] = int(val)
            except: pass

    # Write X, Y, Z in correct format
    write_ims_attr("X", nx)
    write_ims_attr("Y", ny)
    write_ims_attr("Z", nz)

    # Update BOTH ExtMin and ExtMax
    # base_offset tells us how many pixels of padding were added on the low side
    # We shift ExtMin back by that amount so the physical coordinates stay consistent
    dim_map = [
        ("Z", "ExtMin2", "ExtMax2", 0),
        ("Y", "ExtMin1", "ExtMax1", 1),
        ("X", "ExtMin0", "ExtMax0", 2),
    ]
    for dim_key, ext_min_key, ext_max_key, dim_idx in dim_map:
        min_str = get_attr_str(ext_min_key)
        max_str = get_attr_str(ext_max_key)
        orig_n = orig_meta.get(dim_key)
        new_n = shape[dim_idx]

        if min_str and max_str and orig_n and orig_n > 0:
            try:
                ext_min = float(min_str)
                ext_max = float(max_str)
                voxel_size = (ext_max - ext_min) / orig_n

                offset_pixels = base_offset[dim_idx] if base_offset is not None else 0
                new_min = ext_min - offset_pixels * voxel_size
                new_max = new_min + new_n * voxel_size

                write_ims_attr(ext_min_key, f"{new_min:.6f}")
                write_ims_attr(ext_max_key, f"{new_max:.6f}")
            except (ValueError, ZeroDivisionError):
                pass

    # Also update DataSetInfo/Dimension X/Y/Z NumberOfElements
    for dim_name, dim_val in [("X", nx), ("Y", ny), ("Z", nz)]:
        dim_group = h5f.get(f"DataSetInfo/Dimension {dim_name}")
        if dim_group is not None and "NumberOfElements" in dim_group.attrs:
            val_str = str(dim_val)
            byte_array = np.array([c.encode('utf-8') for c in val_str], dtype='|S1')
            del dim_group.attrs["NumberOfElements"]
            dim_group.attrs.create("NumberOfElements", byte_array)


# ── AUTO-SUGGEST LANDMARKS ────────────────────────────────────

def auto_suggest_landmarks(mip, n=5, min_dist=40):
    """Find N brightest isolated spots in a MIP for suggested landmark positions."""
    from scipy.ndimage import maximum_filter, label
    # Smooth slightly to reduce noise
    from scipy.ndimage import uniform_filter
    smooth = uniform_filter(mip.astype(np.float64), size=5)
    # Find local maxima
    local_max = maximum_filter(smooth, size=max(15, min_dist // 2))
    peaks = (smooth == local_max) & (smooth > np.percentile(smooth, 95))
    labeled, n_found = label(peaks)
    
    candidates = []
    for i in range(1, n_found + 1):
        ys, xs = np.where(labeled == i)
        cy, cx = int(ys.mean()), int(xs.mean())
        val = smooth[cy, cx]
        candidates.append((val, cy, cx))
    
    candidates.sort(reverse=True)
    
    # Filter by minimum distance
    selected = []
    for val, cy, cx in candidates:
        if len(selected) >= n:
            break
        too_close = False
        for _, sy, sx in selected:
            if np.sqrt((cy - sy)**2 + (cx - sx)**2) < min_dist:
                too_close = True; break
        if not too_close:
            selected.append((val, cy, cx))
    
    return [(y, x) for _, y, x in selected]


def auto_place_landmarks(volumes, n=5, min_dist=40, search_radius=100):
    """
    Automatically detect and match landmarks across ALL timepoints.
    
    Algorithm:
    1. Find many peaks in each timepoint's MIP
    2. Estimate global drift between T0 and each Tn using peak matching
    3. For each selected T0 peak, find the best matching peak in Tn
       near the drift-corrected expected position
    4. Get Z positions from the 3D volumes
    """
    from scipy.ndimage import maximum_filter, label, uniform_filter
    
    n_tp = len(volumes)
    mips = [v.max(axis=0) for v in volumes]
    
    # Find target peaks in T0 (the ones we want to track)
    target_peaks = auto_suggest_landmarks(mips[0], n=n, min_dist=min_dist)
    if not target_peaks:
        return []
    
    # Find many candidate peaks in ALL timepoints (for drift estimation + matching)
    all_tp_peaks = []
    for t in range(n_tp):
        candidates = auto_suggest_landmarks(mips[t], n=max(n * 4, 20), min_dist=15)
        all_tp_peaks.append(np.array(candidates, dtype=np.float64) if candidates else np.empty((0, 2)))
    
    t0_all = all_tp_peaks[0]
    
    all_landmarks = []
    for sy, sx in target_peaks:
        # T0: get Z position from local maximum in volume
        sr = 3
        yl, yh = max(0, sy-sr), min(volumes[0].shape[1], sy+sr+1)
        xl, xh = max(0, sx-sr), min(volumes[0].shape[2], sx+sr+1)
        col = volumes[0][:, yl:yh, xl:xh]
        sz = int(np.argmax(col.max(axis=(1, 2))))
        
        round_pos = [(sz, sy, sx)]
        
        for t in range(1, n_tp):
            tn_peaks = all_tp_peaks[t]
            if len(tn_peaks) == 0 or len(t0_all) == 0:
                round_pos.append(None)
                continue
            
            # Estimate global drift: for each T0 peak, find nearest Tn peak
            # Then take median shift
            if len(tn_peaks) >= 3 and len(t0_all) >= 3:
                # Brute-force nearest neighbor (no scipy.spatial needed)
                shifts = []
                for t0_y, t0_x in t0_all[:min(len(t0_all), 20)]:
                    dists = np.sqrt((tn_peaks[:, 0] - t0_y)**2 + (tn_peaks[:, 1] - t0_x)**2)
                    best_idx = np.argmin(dists)
                    if dists[best_idx] < search_radius:
                        shifts.append(tn_peaks[best_idx] - np.array([t0_y, t0_x]))
                
                if len(shifts) >= 3:
                    global_drift = np.median(np.array(shifts), axis=0)
                else:
                    global_drift = np.array([0.0, 0.0])
            else:
                global_drift = np.array([0.0, 0.0])
            
            # Find best match for this specific peak near expected position
            expected_y = sy + global_drift[0]
            expected_x = sx + global_drift[1]
            
            dists_to_expected = np.sqrt(
                (tn_peaks[:, 0] - expected_y)**2 + (tn_peaks[:, 1] - expected_x)**2
            )
            best_idx = np.argmin(dists_to_expected)
            
            if dists_to_expected[best_idx] < min_dist * 1.5:
                ty, tx = int(round(tn_peaks[best_idx, 0])), int(round(tn_peaks[best_idx, 1]))
            else:
                # No good peak match — use drift-adjusted position
                ty = int(round(max(0, min(expected_y, mips[t].shape[0] - 1))))
                tx = int(round(max(0, min(expected_x, mips[t].shape[1] - 1))))
            
            # Get Z from volume
            yl, yh = max(0, ty-sr), min(volumes[t].shape[1], ty+sr+1)
            xl, xh = max(0, tx-sr), min(volumes[t].shape[2], tx+sr+1)
            col_t = volumes[t][:, yl:yh, xl:xh]
            tz = int(np.argmax(col_t.max(axis=(1, 2))))
            
            round_pos.append((tz, ty, tx))
        
        all_landmarks.append(round_pos)
    
    return all_landmarks


# ── CONFIDENCE OVERLAY ────────────────────────────────────────

def compute_landmark_confidence(all_lm, volumes, radius=(10, 30, 30)):
    """Compute phase correlation confidence for each landmark at each timepoint."""
    confidences = []
    for rr in all_lm:
        conf_round = []
        for t in range(len(rr)):
            if t == 0 or rr[t] is None or rr[0] is None:
                conf_round.append(1.0 if rr[t] is not None else 0.0)
                continue
            try:
                cube_ref = extract_local_cube(volumes[0], 
                    [int(round(c)) for c in rr[0]], radius).astype(np.float32)
                cube_mov = extract_local_cube(volumes[t],
                    [int(round(c)) for c in rr[t]], radius).astype(np.float32)
                min_shape = tuple(min(a, b) for a, b in zip(cube_ref.shape, cube_mov.shape))
                cube_ref = cube_ref[:min_shape[0], :min_shape[1], :min_shape[2]]
                cube_mov = cube_mov[:min_shape[0], :min_shape[1], :min_shape[2]]
                ref_std = cube_ref.std()
                mov_std = cube_mov.std()
                if ref_std < 1.0 or mov_std < 1.0:
                    conf_round.append(0.2)
                    continue
                # Normalized cross-correlation as confidence
                norm_ref = (cube_ref - cube_ref.mean()) / (ref_std + 1e-8)
                norm_mov = (cube_mov - cube_mov.mean()) / (mov_std + 1e-8)
                ncc = np.mean(norm_ref * norm_mov)
                conf_round.append(max(0.0, min(1.0, ncc)))
            except Exception:
                conf_round.append(0.0)
        confidences.append(conf_round)
    return confidences


def show_confidence_overlay(all_lm, confidences, mips_xy, days, lm_colors_func):
    """Show a brief confidence overlay on the MIPs."""
    n_tp = len(mips_xy)
    fig, axes = plt.subplots(1, n_tp, figsize=(6 * n_tp, 6))
    if n_tp == 1: axes = [axes]
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle("Landmark Confidence  ·  Green = strong match, Red = weak",
                 fontsize=13, fontweight='bold', color='white')
    
    for i in range(n_tp):
        vmin, vmax = np.percentile(mips_xy[i], 1), np.percentile(mips_xy[i], 99.5)
        axes[i].imshow(mips_xy[i], cmap='gray', vmin=vmin, vmax=vmax)
        axes[i].set_title(days[i], fontsize=12, fontweight='bold', color='white')
        axes[i].tick_params(colors='#666', labelsize=6)
        for spine in axes[i].spines.values(): spine.set_color('#333')
        
        for ri, (rr, conf) in enumerate(zip(all_lm, confidences)):
            if i < len(rr) and rr[i] is not None:
                z, y, x = rr[i]
                c_val = conf[i] if i < len(conf) else 0.5
                # Red (low) -> Yellow (mid) -> Green (high)
                r_c = max(0, 1.0 - c_val * 2)
                g_c = min(1.0, c_val * 2)
                color = (r_c, g_c, 0.2)
                size = 200 + c_val * 300
                axes[i].scatter(x, y, s=size, c=[color], marker='o', alpha=0.7, 
                               edgecolors=lm_colors_func(ri), linewidths=2, zorder=10)
                axes[i].annotate(f"L{ri+1}\n{c_val:.0%}", (x, y), color='white',
                               fontsize=8, fontweight='bold', ha='center', va='top',
                               xytext=(0, 18), textcoords='offset points',
                               bbox=dict(fc='#000', alpha=0.7, ec='none', boxstyle='round,pad=0.2'))
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(3)
    plt.close(fig)


# ── BEFORE/AFTER PREVIEW ─────────────────────────────────────

def show_before_after_preview(vols_orig, transforms, orig_shape, new_shape, base_off, 
                               days, skipped_tps, valid_tps):
    """Show aligned MIP preview with Accept/Redo choice."""
    n_orig = len(vols_orig)
    n_valid = len(valid_tps)
    
    # Compute aligned MIPs
    aligned_mips = []
    for new_t, orig_t in enumerate(valid_tps):
        R, tr, _ = transforms[orig_t]
        vol = vols_orig[orig_t]
        if orig_t == 0:
            expanded = place_t0(vol, new_shape, base_off)
        else:
            expanded = apply_correction(vol, R, tr, orig_shape, new_shape, base_off)
        aligned_mips.append(expanded.max(axis=0))
    
    orig_mips = [v.max(axis=0) for v in vols_orig]
    
    fig, axes = plt.subplots(2, max(n_orig, n_valid), figsize=(6 * max(n_orig, n_valid), 10))
    if max(n_orig, n_valid) == 1:
        axes = axes.reshape(2, 1)
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle("Before (top)  vs  After Alignment (bottom)  ·  Close window to accept",
                 fontsize=14, fontweight='bold', color='white')
    
    for i in range(n_orig):
        vmin, vmax = np.percentile(orig_mips[i], 1), np.percentile(orig_mips[i], 99.5)
        axes[0, i].imshow(orig_mips[i], cmap='gray', vmin=vmin, vmax=max(vmax, 1))
        label = f"{days[i]}" + (" [SKIP]" if i in skipped_tps else "")
        axes[0, i].set_title(label, fontsize=11, fontweight='bold', color='white')
        for spine in axes[0, i].spines.values(): spine.set_color('#333')
        axes[0, i].tick_params(colors='#555', labelsize=5)
    
    for i in range(n_valid):
        vmax = np.percentile(aligned_mips[i], 99.5)
        axes[1, i].imshow(aligned_mips[i], cmap='gray', vmin=0, vmax=max(vmax, 1))
        axes[1, i].set_title(f"{days[valid_tps[i]]} aligned", fontsize=11, 
                            fontweight='bold', color='#00f5d4')
        for spine in axes[1, i].spines.values(): spine.set_color('#333')
        axes[1, i].tick_params(colors='#555', labelsize=5)
    
    # Hide unused axes
    for i in range(n_valid, max(n_orig, n_valid)):
        if i < axes.shape[1]: axes[1, i].set_visible(False)
    for i in range(n_orig, max(n_orig, n_valid)):
        if i < axes.shape[1]: axes[0, i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    plt.rcdefaults()
    return True  # Accept (Redo would require re-running picker)


# ── HTML DASHBOARD ────────────────────────────────────────────

def generate_html_dashboard(output_dir, file_results):
    """Generate an HTML summary report for a batch run."""
    html_path = output_dir / f"alignment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    
    rows = ""
    for res in file_results:
        status_color = "#4DFF88" if res["status"] == "done" else "#FFD74D" if res["status"] == "skip" else "#FF4D4D"
        skipped_str = ", ".join(res.get("skipped_days", [])) if res.get("skipped_days") else "—"
        
        tp_cells = ""
        for tp_info in res.get("timepoints", []):
            tp_cells += f"""<td style="text-align:center;padding:6px;">
                <div style="font-size:11px;color:#aaa;">{tp_info['day']}</div>
                <div>T={tp_info['translation']:.1f}vox</div>
                <div>R={tp_info['rotation']:.1f}°</div>
                <div style="color:{'#4DFF88' if tp_info['rmse']<5 else '#FFD74D' if tp_info['rmse']<10 else '#FF4D4D'};">
                    RMSE={tp_info['rmse']:.1f}</div>
            </td>"""
        
        rows += f"""
        <tr style="border-bottom:1px solid #333;">
            <td style="padding:8px;font-weight:bold;">{res['filename']}</td>
            <td style="padding:8px;color:{status_color};font-weight:bold;">{res['status'].upper()}</td>
            <td style="padding:8px;">{res.get('n_landmarks', '—')}</td>
            <td style="padding:8px;">{res.get('n_timepoints', '—')}</td>
            <td style="padding:8px;color:#888;">{skipped_str}</td>
            {tp_cells}
        </tr>"""
    
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>MSP Landmark Drift Correction — Batch Report</title>
<style>
    body {{ background:#1a1a2e; color:#e0e0e0; font-family:'Segoe UI',sans-serif; padding:30px; }}
    h1 {{ color:#00b4d8; border-bottom:2px solid #00b4d8; padding-bottom:10px; }}
    h2 {{ color:#00f5d4; margin-top:30px; }}
    table {{ border-collapse:collapse; width:100%; background:#16213e; border-radius:8px; overflow:hidden; }}
    th {{ background:#0f3460; padding:10px; text-align:left; color:#00b4d8; font-size:12px; text-transform:uppercase; }}
    td {{ padding:8px; font-size:13px; }}
    tr:hover {{ background:#1a3a5c; }}
    .meta {{ color:#888; font-size:13px; margin:8px 0; }}
    .stat {{ display:inline-block; background:#0f3460; padding:8px 16px; border-radius:6px; margin:4px; }}
    .stat-val {{ font-size:20px; font-weight:bold; color:#00f5d4; }}
    .stat-label {{ font-size:11px; color:#888; text-transform:uppercase; }}
</style></head><body>
<h1>MSP Landmark Drift Correction Tool</h1>
<p class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · Output: {output_dir}</p>

<div style="margin:20px 0;">
    <div class="stat"><div class="stat-val">{sum(1 for r in file_results if r['status']=='done')}</div><div class="stat-label">Corrected</div></div>
    <div class="stat"><div class="stat-val">{sum(1 for r in file_results if r['status']=='skip')}</div><div class="stat-label">Skipped</div></div>
    <div class="stat"><div class="stat-val">{len(file_results)}</div><div class="stat-label">Total Files</div></div>
</div>

<h2>File Results</h2>
<table>
<tr><th>File</th><th>Status</th><th>Landmarks</th><th>Timepoints</th><th>Skipped TPs</th></tr>
{rows}
</table>
</body></html>"""
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


# ── LANDMARK PICKING ─────────────────────────────────────────

def pick_landmarks(volumes, filename, day_labels, max_rot_default=25.0,
                    ref_ch=0, all_channel_volumes=None, batch_info=None,
                    is_batch_mode=True):
    """
    Interactive landmark picker — polished UI with per-landmark colors.
    """
    from matplotlib.widgets import Button, CheckButtons, Slider
    n_tp = len(volumes)
    mips_xy = [v.max(axis=0) for v in volumes]
    mips_xz = [v.max(axis=1) for v in volumes]
    img_h, img_w = mips_xy[0].shape

    # Multi-channel state
    n_ch_total = len(all_channel_volumes[0]) if all_channel_volumes else 1

    def reload_channel(ch_idx):
        """Switch the displayed volumes to a different channel."""
        if all_channel_volumes is None or ch_idx >= n_ch_total:
            return
        for t in range(n_tp):
            volumes[t] = all_channel_volumes[t][ch_idx]
            mips_xy[t] = volumes[t].max(axis=0)
            mips_xz[t] = volumes[t].max(axis=1)
        # Update display
        for i in range(n_tp):
            vmin = np.percentile(mips_xy[i], 1)
            vmax = np.percentile(mips_xy[i], 99.5)
            if axes_xy[i].images:
                axes_xy[i].images[0].set_data(mips_xy[i])
                axes_xy[i].images[0].set_clim(vmin, max(vmax, 1))
            vmin_z = np.percentile(mips_xz[i], 1)
            vmax_z = np.percentile(mips_xz[i], 99.5)
            if axes_xz[i].images:
                axes_xz[i].images[0].set_data(mips_xz[i])
                axes_xz[i].images[0].set_clim(vmin_z, max(vmax_z, 1))
        fig.canvas.draw_idle()

    n_cols = min(n_tp, 3)
    n_rows = (n_tp + n_cols - 1) // n_cols

    all_lm = []; cur = []; cur_tp = [0]; rnd = [1]
    result = [None, "done"]
    closed = [False]
    skipped_tps = [set()]
    contrast_pct = [99.5]  # adjustable contrast percentile
    max_rot = [max_rot_default]

    # ── Color palette for landmarks ──
    LM_COLORS = [
        '#FF4D4D',  # L1 red
        '#4DA6FF',  # L2 blue
        '#4DFF88',  # L3 green
        '#FFD74D',  # L4 gold
        '#D94DFF',  # L5 purple
        '#FF944D',  # L6 orange
        '#4DFFF0',  # L7 cyan
        '#FF4DA6',  # L8 pink
        '#A6FF4D',  # L9 lime
        '#4D6AFF',  # L10 indigo
    ]

    def lm_color(idx):
        return LM_COLORS[idx % len(LM_COLORS)]

    # ── Dark theme ──
    BG_COLOR = '#1a1a2e'
    PANEL_BG = '#16213e'
    TEXT_COLOR = '#e0e0e0'
    ACCENT = '#00b4d8'
    BORDER_ACTIVE = '#00f5d4'
    BORDER_INACTIVE = '#333355'

    plt.rcParams.update({
        'figure.facecolor': BG_COLOR,
        'axes.facecolor': '#0a0a1a',
        'axes.edgecolor': BORDER_INACTIVE,
        'axes.labelcolor': TEXT_COLOR,
        'xtick.color': '#888',
        'ytick.color': '#888',
        'text.color': TEXT_COLOR,
        'keymap.save': [],        # Disable 's' = save figure (we use S for skip)
        'keymap.fullscreen': [],  # Disable 'f' conflicts
        'keymap.quit': [],        # Disable 'q' conflicts
    })

    col_width = 8
    fig = plt.figure(figsize=(col_width * n_cols, 7 * n_rows + 2.2))
    fig.patch.set_facecolor(BG_COLOR)
    # Maximize window on startup
    try:
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')  # Windows/TkAgg maximize
    except Exception:
        try: mng.resize(*mng.window.maxsize())
        except Exception: pass

    row_heights = []
    for _ in range(n_rows):
        row_heights.extend([5, 1])

    gs = fig.add_gridspec(n_rows * 2, n_cols,
                          left=0.03, right=0.97, top=0.86, bottom=0.16,
                          hspace=0.22, wspace=0.08,
                          height_ratios=row_heights)

    axes_xy = []; axes_xz = []
    for r in range(n_rows):
        for ci in range(n_cols):
            idx = r * n_cols + ci
            if idx < n_tp:
                axes_xy.append(fig.add_subplot(gs[r*2, ci]))
                axes_xz.append(fig.add_subplot(gs[r*2+1, ci]))

    # ── Magnifier placement ──
    # If the grid has an unfilled slot (e.g., 5 timepoints in a 2×3 grid), put magnifier there.
    # Otherwise (6 TPs filling a 2×3 grid), float in the corner of the last subplot.
    n_grid_slots = n_rows * n_cols
    has_free_slot = (n_grid_slots > n_tp)
    mag_minimized = [False]   # toggleable

    if has_free_slot:
        # Place in the empty slot's XY area
        last_idx = n_tp  # first empty slot index
        r = last_idx // n_cols
        ci = last_idx % n_cols
        # Compute pixel-space position of that grid cell
        gs_pos_xy = gs[r*2, ci].get_position(fig)  # XY row
        mag_x0, mag_y0 = gs_pos_xy.x0, gs_pos_xy.y0
        mag_w, mag_h = gs_pos_xy.width, gs_pos_xy.height
        ax_mag = fig.add_axes([mag_x0, mag_y0, mag_w, mag_h])
    else:
        # Float in lower-right corner of the last subplot (overlay)
        gs_pos_last = gs[(n_rows-1)*2, n_cols-1].get_position(fig)
        overlay_w = gs_pos_last.width * 0.40
        overlay_h = gs_pos_last.height * 0.40
        mag_x0 = gs_pos_last.x1 - overlay_w - 0.005
        mag_y0 = gs_pos_last.y0 + 0.005
        ax_mag = fig.add_axes([mag_x0, mag_y0, overlay_w, overlay_h])

    ax_mag.set_title("MAGNIFIER", fontsize=8, fontweight='bold', color=ACCENT, pad=4)
    ax_mag.set_xticks([]); ax_mag.set_yticks([])
    for spine in ax_mag.spines.values(): spine.set_color(ACCENT); spine.set_linewidth(1.5)
    mag_img = ax_mag.imshow(np.zeros((60, 60)), cmap='gray', aspect='equal')
    mag_vline = ax_mag.axvline(x=0.5, color=ACCENT, linewidth=0.8, alpha=0.8)
    mag_hline = ax_mag.axhline(y=0.5, color=ACCENT, linewidth=0.8, alpha=0.8)
    mag_visible = [True]

    # Magnifier minimize toggle button (small icon in the top-right corner of the magnifier)
    mag_pos = ax_mag.get_position()
    ax_mag_min = fig.add_axes([mag_pos.x1 - 0.020, mag_pos.y1 - 0.022, 0.020, 0.022])
    btn_mag_min = Button(ax_mag_min, '–', color='#0a1424', hovercolor=ACCENT)
    btn_mag_min.label.set_color(ACCENT); btn_mag_min.label.set_fontweight('bold'); btn_mag_min.label.set_fontsize(11)
    for s in ax_mag_min.spines.values(): s.set_color(ACCENT); s.set_linewidth(0.8)

    # Floating mini-icon shown when minimized (click to restore)
    ax_mag_restore = fig.add_axes([mag_pos.x1 - 0.030, mag_pos.y1 - 0.025, 0.030, 0.025])
    btn_mag_restore = Button(ax_mag_restore, 'Mag', color='#0a1424', hovercolor=ACCENT)
    btn_mag_restore.label.set_color(ACCENT); btn_mag_restore.label.set_fontweight('bold'); btn_mag_restore.label.set_fontsize(8)
    for s in ax_mag_restore.spines.values(): s.set_color(ACCENT); s.set_linewidth(0.8)
    ax_mag_restore.set_visible(False)

    def toggle_mag_min():
        mag_minimized[0] = not mag_minimized[0]
        ax_mag.set_visible(not mag_minimized[0])
        ax_mag_min.set_visible(not mag_minimized[0])
        ax_mag_restore.set_visible(mag_minimized[0])
        fig.canvas.draw_idle()

    btn_mag_min.on_clicked(lambda ev: toggle_mag_min())
    btn_mag_restore.on_clicked(lambda ev: toggle_mag_min())

    orig_xlim = (0, img_w)
    orig_ylim = (img_h, 0)
    zoom_level = [1.0]

    # Display MIPs
    for i in range(n_tp):
        vmin, vmax = np.percentile(mips_xy[i], 1), np.percentile(mips_xy[i], 99.5)
        axes_xy[i].imshow(mips_xy[i], cmap='gray', aspect='equal', vmin=vmin, vmax=vmax)
        axes_xy[i].set_title(day_labels[i], fontsize=14, fontweight='bold', color='white', pad=6)
        axes_xy[i].tick_params(labelsize=6, colors='#666')
        for spine in axes_xy[i].spines.values(): spine.set_color(BORDER_INACTIVE); spine.set_linewidth(1.5)

        vmin_z, vmax_z = np.percentile(mips_xz[i], 1), np.percentile(mips_xz[i], 99.5)
        axes_xz[i].imshow(mips_xz[i], cmap='gray', aspect='auto', vmin=vmin_z, vmax=vmax_z)
        axes_xz[i].set_ylabel("Z", fontsize=7, color='#888')
        axes_xz[i].tick_params(labelsize=5, colors='#555')
        for spine in axes_xz[i].spines.values(): spine.set_color(BORDER_INACTIVE); spine.set_linewidth(1)

    txts = [[] for _ in range(n_tp)]
    plot_markers = [[] for _ in range(n_tp)]  # Track plot artists for cleanup

    bords_xy = []; bords_xz = []
    for i in range(n_tp):
        r1 = plt.Rectangle((0,0),1,1,transform=axes_xy[i].transAxes,fill=False,
                             edgecolor=BORDER_ACTIVE,linewidth=3,visible=(i==0))
        r2 = plt.Rectangle((0,0),1,1,transform=axes_xz[i].transAxes,fill=False,
                             edgecolor=BORDER_ACTIVE,linewidth=2,visible=(i==0))
        axes_xy[i].add_patch(r1); axes_xz[i].add_patch(r2)
        bords_xy.append(r1); bords_xz.append(r2)

    skip_markers = []
    for i in range(n_tp):
        txt = axes_xy[i].text(0.5, 0.5, "SKIPPED", transform=axes_xy[i].transAxes,
                               ha='center', va='center', fontsize=22, fontweight='bold',
                               color='#FF3333', alpha=0.8, visible=False,
                               bbox=dict(fc='#000000', alpha=0.6, ec='#FF3333', lw=2,
                                         boxstyle='round,pad=0.3'))
        skip_markers.append(txt)

    xz_vlines = [axes_xz[i].axvline(x=-1, color=BORDER_ACTIVE, linewidth=0.8, alpha=0) for i in range(n_tp)]
    xz_hlines = [axes_xz[i].axhline(y=-1, color=BORDER_ACTIVE, linewidth=0.8, alpha=0) for i in range(n_tp)]

    # Header
    title_text = f"MSP Landmark Drift Correction Tool  ·  {filename}"
    if batch_info and is_batch_mode:
        idx, total, elapsed, eta = batch_info
        eta_str = f"ETA {eta:.1f} min" if eta else "—"
        title_text = (f"MSP Landmark Drift Correction Tool  ·  "
                      f"File {idx}/{total}  ·  {filename}  ·  "
                      f"elapsed {elapsed:.1f} min  ·  {eta_str}")
    fig.text(0.5, 0.98, title_text,
             ha='center', va='top', fontsize=11, color='#888', fontstyle='italic',
             fontfamily='monospace')
    instr = fig.text(0.5, 0.940, "", ha='center', va='top', fontsize=12, fontweight='bold',
                     color='white',
                     bbox=dict(boxstyle='round,pad=0.5', fc=PANEL_BG, ec='#FFD700',
                               alpha=0.95, lw=2))
    stat = fig.text(0.5, 0.012, "", ha='center', va='center', fontsize=8,
                    color='#888',
                    bbox=dict(boxstyle='round,pad=0.4', fc=PANEL_BG, ec='#333',
                              alpha=0.85, lw=0.5))

    # ── BOTTOM CONTROL BAR ──
    # Three zones:
    #   Settings (left, 4 cols × 2 rows)
    #   Output canvas (middle, 3 horizontal thumbs + "Output canvas" label centered below)
    #   Action buttons (right, vertically stacked, context-aware)

    # Row baselines (lowered to give plaque images more headroom)
    row_top = 0.090
    row_mid = 0.052
    row_bot = 0.014
    row_h   = 0.030

    # ─── Settings zone (left) ───
    set_col_w = 0.090
    set_col_x = [0.030, 0.130, 0.230, 0.330]
    set_h = 0.026

    # Row 1: Max Rot slider, N slider, Suggest, Auto-place
    ax_slider = fig.add_axes([set_col_x[0] + 0.022, row_top + 0.005, set_col_w - 0.025, 0.014])
    ax_slider.set_facecolor(PANEL_BG)
    slider_rot = Slider(ax_slider, 'Max Rot°', 0.5, 45.0, valinit=max_rot[0],
                        valstep=0.5, color=ACCENT)
    slider_rot.label.set_color(TEXT_COLOR); slider_rot.label.set_fontsize(7)
    slider_rot.valtext.set_color(TEXT_COLOR); slider_rot.valtext.set_fontsize(7)
    def on_slider(val): max_rot[0] = val
    slider_rot.on_changed(on_slider)

    n_auto = [4]
    ax_nslider = fig.add_axes([set_col_x[1] + 0.012, row_top + 0.005, set_col_w - 0.018, 0.014])
    ax_nslider.set_facecolor(PANEL_BG)
    slider_n = Slider(ax_nslider, 'N', 3, 12, valinit=4, valstep=1, color='#00f5d4')
    slider_n.label.set_color(TEXT_COLOR); slider_n.label.set_fontsize(7)
    slider_n.valtext.set_color(TEXT_COLOR); slider_n.valtext.set_fontsize(7)
    def on_n_slider(val): n_auto[0] = int(val)
    slider_n.on_changed(on_n_slider)

    ax_auto = fig.add_axes([set_col_x[2], row_top, set_col_w, set_h])
    btn_auto = Button(ax_auto, 'Suggest', color='#1a3a5a', hovercolor='#2a5a7a')
    btn_auto.label.set_color(ACCENT); btn_auto.label.set_fontweight('bold'); btn_auto.label.set_fontsize(8)

    ax_autoplace = fig.add_axes([set_col_x[3], row_top, set_col_w, set_h])
    btn_autoplace = Button(ax_autoplace, 'Auto-place', color='#15401e', hovercolor='#256030')
    btn_autoplace.label.set_color('#88FF88'); btn_autoplace.label.set_fontweight('bold'); btn_autoplace.label.set_fontsize(8)

    # Row 2: Refine, Delete all, Reset view, Preview
    ax_refine = fig.add_axes([set_col_x[0], row_mid, set_col_w, set_h])
    btn_refine = Button(ax_refine, 'Refine', color='#3a2a5a', hovercolor='#5a4a7a')
    btn_refine.label.set_color('#C9A8FF'); btn_refine.label.set_fontweight('bold'); btn_refine.label.set_fontsize(8)

    ax_delall = fig.add_axes([set_col_x[1], row_mid, set_col_w, set_h])
    btn_delall = Button(ax_delall, 'Delete all', color='#3a1818', hovercolor='#5a2828')
    btn_delall.label.set_color('#FF8888'); btn_delall.label.set_fontweight('bold'); btn_delall.label.set_fontsize(8)

    ax_resetview = fig.add_axes([set_col_x[2], row_mid, set_col_w, set_h])
    btn_resetview = Button(ax_resetview, 'Reset view', color='#1a3a5a', hovercolor='#2a5a7a')
    btn_resetview.label.set_color(ACCENT); btn_resetview.label.set_fontweight('bold'); btn_resetview.label.set_fontsize(8)

    show_preview = [False]
    ax_chk = fig.add_axes([set_col_x[3], row_mid, set_col_w, set_h])
    for spine in ax_chk.spines.values(): spine.set_linewidth(0)
    btn_preview = Button(ax_chk, '\u2610  Preview', color='#1a3a5a', hovercolor='#2a5a7a')
    btn_preview.label.set_color(ACCENT); btn_preview.label.set_fontweight('bold'); btn_preview.label.set_fontsize(8)
    def on_chk(ev):
        show_preview[0] = not show_preview[0]
        btn_preview.label.set_text('\u2611  Preview' if show_preview[0] else '\u2610  Preview')
        fig.canvas.draw_idle()
    btn_preview.on_clicked(on_chk)

    # Help icon (top-right)
    ax_help = fig.add_axes([0.965, 0.96, 0.025, 0.030])
    btn_help = Button(ax_help, '?', color='#2a3a5a', hovercolor='#3a5a7a')
    btn_help.label.set_color(ACCENT); btn_help.label.set_fontweight('bold'); btn_help.label.set_fontsize(11)

    # ─── Output canvas zone (middle): 3 horizontal thumbnails + "Output canvas" label below ───
    canvas_mode = ['crop_overlap']
    cm_thumb_w = 0.060
    cm_thumb_h = 0.060
    cm_thumb_gap = 0.010
    cm_total_w = 3 * cm_thumb_w + 2 * cm_thumb_gap   # 0.200
    cm_x_start = 0.560   # left edge of first thumbnail
    cm_thumbs_y = 0.060  # thumbnail row (spans 0.060 to 0.120 — top aligns with action buttons)
    cm_label_y = 0.048   # label position below thumbnails (clears the bottom shortcut bar at y=0.012)

    # Centered "Output canvas" label below the thumbnails
    fig.text(cm_x_start + cm_total_w / 2, cm_label_y,
             "Output canvas",
             ha='center', va='top', fontsize=9, color=ACCENT, fontweight='bold')

    # Load PNGs
    import matplotlib.image as mpimg
    script_dir = Path(__file__).parent if '__file__' in globals() else Path('.')
    png_paths = {
        'include_all':  script_dir / 'Include_entire_result.png',
        'same_size':    script_dir / 'New_size_equal_to_current_size.png',
        'crop_overlap': script_dir / 'Crop_largest_common_region.png',
    }
    cm_images = {}
    for mode_id, path in png_paths.items():
        try:
            cm_images[mode_id] = mpimg.imread(str(path)) if path.exists() else None
        except Exception:
            cm_images[mode_id] = None

    # Tooltips for the canvas modes
    cm_specs = [
        ('include_all',  'Include entire result',          0),
        ('same_size',    'New size equal to current size', 1),
        ('crop_overlap', 'Crop largest common region',     2),
    ]
    cm_thumb_axes = {}
    cm_buttons = {}
    cm_tooltip_for = {}
    for mode_id, tooltip, idx in cm_specs:
        x = cm_x_start + idx * (cm_thumb_w + cm_thumb_gap)
        ax_thumb = fig.add_axes([x, cm_thumbs_y, cm_thumb_w, cm_thumb_h])
        ax_thumb.set_xticks([]); ax_thumb.set_yticks([])
        ax_thumb.set_facecolor('#FFFFFF')
        is_active = (mode_id == 'crop_overlap')
        for s in ax_thumb.spines.values():
            s.set_color(ACCENT if is_active else '#444')
            s.set_linewidth(2.0 if is_active else 0.5)
        if cm_images.get(mode_id) is not None:
            ax_thumb.imshow(cm_images[mode_id], aspect='auto', interpolation='bilinear')
        else:
            ax_thumb.set_facecolor('#0a0a1a')
            ax_thumb.text(0.5, 0.5, '?', ha='center', va='center',
                          color='#666', fontsize=14, transform=ax_thumb.transAxes)
        cm_thumb_axes[mode_id] = ax_thumb
        cm_tooltip_for[ax_thumb] = tooltip
        # Make the entire thumbnail axis clickable using button_press_event (handled in onclick)
        # We store mode mapping for that lookup
        cm_buttons[mode_id] = ax_thumb  # reuse name for compatibility

    # Map axis → mode for click handling
    cm_ax_to_mode = {ax: mid for mid, ax in cm_thumb_axes.items()}

    def set_canvas_mode(mode):
        canvas_mode[0] = mode
        for mid, ax in cm_thumb_axes.items():
            active = (mid == mode)
            for s in ax.spines.values():
                s.set_color(ACCENT if active else '#444')
                s.set_linewidth(2.0 if active else 0.5)
        fig.canvas.draw_idle()
        print(f"  Output canvas mode: {mode}")

    set_canvas_mode('crop_overlap')

    # ── ACTION BUTTONS (right side, vertically stacked, context-aware) ──
    act_x = 0.795; act_w = 0.180

    if is_batch_mode:
        ax_skip = fig.add_axes([act_x, row_top, act_w, row_h])
        ax_done = fig.add_axes([act_x, row_mid, act_w, row_h])
        ax_finish = fig.add_axes([act_x, row_bot, act_w, row_h])
        for ax_btn in [ax_skip, ax_done, ax_finish]:
            for spine in ax_btn.spines.values(): spine.set_linewidth(0)
        btn_skip = Button(ax_skip, 'Skip File', color='#3a1818', hovercolor='#5a2828')
        btn_done = Button(ax_done, 'Done File', color='#15401e', hovercolor='#256030')
        btn_finish = Button(ax_finish, 'Finish', color='#3a2a08', hovercolor='#5a4818')
        btn_skip.label.set_color('#FF8888'); btn_skip.label.set_fontweight('bold'); btn_skip.label.set_fontsize(10)
        btn_done.label.set_color('#88FF88'); btn_done.label.set_fontweight('bold'); btn_done.label.set_fontsize(10)
        btn_finish.label.set_color('#FFD700'); btn_finish.label.set_fontweight('bold'); btn_finish.label.set_fontsize(10)
    else:
        ax_skip = fig.add_axes([act_x, row_top, act_w, row_h])
        ax_done = fig.add_axes([act_x, row_mid, act_w, row_h])
        for ax_btn in [ax_skip, ax_done]:
            for spine in ax_btn.spines.values(): spine.set_linewidth(0)
        btn_skip = Button(ax_skip, 'Cancel', color='#3a1818', hovercolor='#5a2828')
        btn_done = Button(ax_done, 'Complete', color='#15401e', hovercolor='#256030')
        btn_skip.label.set_color('#FF8888'); btn_skip.label.set_fontweight('bold'); btn_skip.label.set_fontsize(10)
        btn_done.label.set_color('#88FF88'); btn_done.label.set_fontweight('bold'); btn_done.label.set_fontsize(11)
        btn_finish = None

    # Auto-suggest markers (for cleanup)
    suggest_markers = []

    def on_auto_suggest(ev):
        if closed[0]: return
        # Clear old suggestions
        for m in suggest_markers: m.remove()
        suggest_markers.clear()
        suggestions = auto_suggest_landmarks(mips_xy[0], n=n_auto[0], min_dist=40)
        if not suggestions:
            print("  No landmarks auto-detected"); return
        print(f"  Auto-suggested {len(suggestions)} landmark positions on T0")
        for sy, sx in suggestions:
            ln, = axes_xy[0].plot(sx, sy, 'o', color='#00f5d4', markersize=18,
                                  markeredgewidth=1.2, fillstyle='none', zorder=9, alpha=0.6)
            suggest_markers.append(ln)
        fig.canvas.draw_idle()

    btn_auto.on_clicked(on_auto_suggest)

    def on_auto_place(ev):
        if closed[0]: return
        # Clear old suggestions
        for m in suggest_markers: m.remove()
        suggest_markers.clear()

        print(f"  Auto-placing {n_auto[0]} landmarks across all timepoints...")
        placed = auto_place_landmarks(volumes, n=n_auto[0], min_dist=40, search_radius=100)
        if not placed:
            print("  Auto-placement failed — no peaks detected"); return

        # Add as completed landmarks
        for rr in placed:
            all_lm.append(rr)
            rnd[0] += 1

        print(f"  Auto-placed {len(placed)} landmarks ({len(placed) * n_tp} positions)")
        print(f"  Inspect results — Shift+click on any marker to remove & re-place")
        cur.clear(); cur_tp[0] = 0
        # Skip past skipped TPs
        while cur_tp[0] < n_tp and cur_tp[0] in skipped_tps[0]:
            cur.append(None); cur_tp[0] += 1
        refresh()

    btn_autoplace.on_clicked(on_auto_place)

    # Refine: re-center each placed landmark on its nearest bright peak
    def on_refine(ev):
        if closed[0]: return
        if not all_lm:
            print("  No landmarks to refine"); return

        print(f"  Refining {len(all_lm)} landmark(s)...")
        # Window: ~20 vox in YX, ~8 vox in Z — large enough to find center if user clicked off,
        # small enough not to jump to a different plaque
        ry_xy = 20; ry_z = 8
        moved_count = 0; total_dist = 0.0
        for ri, rr in enumerate(all_lm):
            for ti in range(len(rr)):
                if rr[ti] is None: continue
                z, y, x = rr[ti]
                zi, yi, xi = int(round(z)), int(round(y)), int(round(x))
                vol = volumes[ti]
                zl = max(0, zi - ry_z); zh = min(vol.shape[0], zi + ry_z + 1)
                yl = max(0, yi - ry_xy); yh = min(vol.shape[1], yi + ry_xy + 1)
                xl = max(0, xi - ry_xy); xh = min(vol.shape[2], xi + ry_xy + 1)
                local = vol[zl:zh, yl:yh, xl:xh]
                if local.size == 0: continue
                pk = np.unravel_index(local.argmax(), local.shape)
                new_z = zl + pk[0]; new_y = yl + pk[1]; new_x = xl + pk[2]
                d = np.sqrt((new_z-zi)**2 + (new_y-yi)**2 + (new_x-xi)**2)
                if d > 0.5:
                    rr[ti] = (new_z, new_y, new_x)
                    moved_count += 1
                    total_dist += d
        if moved_count > 0:
            print(f"  Refined {moved_count} positions (avg shift {total_dist/moved_count:.1f} vox)")
        else:
            print(f"  All landmarks already centered on peaks")
        refresh()

    btn_refine.on_clicked(on_refine)

    # Delete all: clear every landmark
    def on_delete_all(ev):
        if closed[0]: return
        if not all_lm and not cur:
            print("  Nothing to delete"); return
        n_cleared = len(all_lm) + (1 if cur else 0)
        all_lm.clear()
        cur.clear()
        cur_tp[0] = 0
        rnd[0] = 1
        # Clear suggestion markers too
        for m in suggest_markers: m.remove()
        suggest_markers.clear()
        # Skip past any skipped TPs
        while cur_tp[0] < n_tp and cur_tp[0] in skipped_tps[0]:
            cur.append(None); cur_tp[0] += 1
        for h in xz_hlines: h.set_alpha(0)
        print(f"  Cleared all landmarks ({n_cleared} round(s))")
        refresh()

    btn_delall.on_clicked(on_delete_all)

    # Image objects for contrast updates
    img_objects_xy = []
    img_objects_xz = []

    def close_with_action(action):
        if closed[0]: return
        if action == "done" and not all_lm:
            print("  Place at least 1 landmark first!"); return
        cur.clear()
        result[0] = list(all_lm) if all_lm else None
        result[1] = action
        closed[0] = True
        # Reset rcParams
        plt.rcdefaults()
        plt.close(fig)

    btn_skip.on_clicked(lambda ev: close_with_action("skip"))
    btn_done.on_clicked(lambda ev: close_with_action("done"))
    if btn_finish is not None:
        btn_finish.on_clicked(lambda ev: close_with_action("finish"))

    def advance_to_next_valid_tp():
        while cur_tp[0] < n_tp and cur_tp[0] in skipped_tps[0]:
            cur.append(None); cur_tp[0] += 1
        if cur_tp[0] >= n_tp:
            all_lm.append(list(cur)); cur.clear(); cur_tp[0] = 0
            rnd[0] += 1
            print(f"  Landmark {rnd[0]-1} complete.")
            for h in xz_hlines: h.set_alpha(0)
            while cur_tp[0] < n_tp and cur_tp[0] in skipped_tps[0]:
                cur.append(None); cur_tp[0] += 1

    def refresh():
        if closed[0]:
            instr.set_text("  Processing...  "); stat.set_text(""); return

        active_tp = cur_tp[0] if cur_tp[0] < n_tp else 0
        cur_color = lm_color(rnd[0] - 1)

        # Compute live alignment quality
        compute_quality()
        quality_str = ""
        if quality_info[0] is not None:
            rmse_avg, outliers, spread_ok = quality_info[0]
            if rmse_avg is not None:
                if rmse_avg < 5: q_lbl, q_col = "good", '#88FF88'
                elif rmse_avg < 10: q_lbl, q_col = "marginal", '#FFD700'
                else: q_lbl, q_col = "poor", '#FF8888'
                quality_str = f"  ·  Predicted RMSE: {rmse_avg:.1f} vox · {q_lbl}"
            if not spread_ok:
                quality_str += "  ·  ⚠ landmarks too clustered"
            if outliers:
                quality_str += f"  ·  ⚠ L{','.join(f'{i+1}' for i in sorted(outliers))} look like outlier(s)"

        if edit_mode[0] is not None:
            ri, ti = edit_mode[0]
            instr.set_text(f"  CORRECTING L{ri+1}  ·  Click new position in  [ {day_labels[ti]} ]  ")
        else:
            base = f"  LANDMARK {rnd[0]}  ·  Click plaque in  [ {day_labels[active_tp]} ]  "
            instr.set_text(base + quality_str)

        stat.set_text(f"Click=place  ·  Right-drag=pan  ·  Shift+click=correct  ·  Ctrl+Z=undo  ·  "
                      f"Scroll=zoom  ·  R=reset  ·  S=skip  ·  Z=slice  ·  C=channel  ·  ?=help"
                      f"    ·  {len(all_lm)} landmarks")

        outliers_set = quality_info[0][1] if quality_info[0] is not None else set()

        for i in range(n_tp):
            if edit_mode[0] is not None:
                ri, ti = edit_mode[0]
                is_active = (i == ti and not closed[0])
            else:
                is_active = (i == active_tp and not closed[0])
            bords_xy[i].set_visible(is_active)
            bords_xz[i].set_visible(is_active)
            if is_active:
                for spine in axes_xy[i].spines.values():
                    spine.set_color(BORDER_ACTIVE); spine.set_linewidth(2.5)
            else:
                for spine in axes_xy[i].spines.values():
                    spine.set_color(BORDER_INACTIVE); spine.set_linewidth(1.5)
            skip_markers[i].set_visible(i in skipped_tps[0])

        # Redraw all landmarks with per-round colors
        for ti in range(n_tp):
            for t in txts[ti]: t.remove()
            txts[ti].clear()
            for m in plot_markers[ti]: m.remove()
            plot_markers[ti].clear()

            for ri, rr in enumerate(all_lm):
                if ti < len(rr) and rr[ti] is not None:
                    z, y, x = rr[ti]
                    c = lm_color(ri)
                    is_outlier = (ri in outliers_set)
                    # Outliers get a red ring around the marker
                    if is_outlier:
                        ring, = axes_xy[ti].plot(x, y, 'o', color='#FF3333', markersize=20,
                                                  markeredgewidth=1.5, fillstyle='none',
                                                  zorder=9, alpha=0.9)
                        plot_markers[ti].append(ring)
                    ln, = axes_xy[ti].plot(x, y, '+', color=c, markersize=12, markeredgewidth=1.5, zorder=10)
                    plot_markers[ti].append(ln)
                    ln2, = axes_xz[ti].plot(x, z, '+', color=c, markersize=8, markeredgewidth=1.0, zorder=10)
                    plot_markers[ti].append(ln2)
                    label_txt = f"L{ri+1}" + ("⚠" if is_outlier else "")
                    label_color = '#FF3333' if is_outlier else c
                    txts[ti].append(axes_xy[ti].annotate(
                        label_txt, (x, y), color=label_color, fontsize=9, fontweight='bold',
                        ha='left', xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.2', fc='#000000', alpha=0.7, ec=label_color, lw=1)))

            if ti < len(cur) and cur[ti] is not None:
                z, y, x = cur[ti]
                c = lm_color(rnd[0] - 1)
                ln, = axes_xy[ti].plot(x, y, '+', color=c, markersize=14, markeredgewidth=2, zorder=11)
                plot_markers[ti].append(ln)
                ln2, = axes_xz[ti].plot(x, z, '+', color=c, markersize=9, markeredgewidth=1.2, zorder=11)
                plot_markers[ti].append(ln2)
                txts[ti].append(axes_xy[ti].annotate(
                    f"L{rnd[0]}", (x, y), color=c, fontsize=9, fontweight='bold',
                    ha='left', xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.2', fc='#000000', alpha=0.7, ec=c, lw=1.5)))

        fig.canvas.draw_idle()

    def sync_zoom(center_x, center_y, factor):
        zoom_level[0] *= factor
        zoom_level[0] = max(0.5, min(zoom_level[0], 10.0))
        half_w = img_w / (2.0 * zoom_level[0])
        half_h = img_h / (2.0 * zoom_level[0])
        cx = max(half_w, min(center_x, img_w - half_w))
        cy = max(half_h, min(center_y, img_h - half_h))
        for ax in axes_xy:
            ax.set_xlim(cx - half_w, cx + half_w)
            ax.set_ylim(cy + half_h, cy - half_h)
        fig.canvas.draw_idle()

    def update_magnifier(xdata, ydata, tp_idx):
        if not mag_visible[0] or xdata is None or ydata is None: return
        mip = mips_xy[tp_idx]
        r = 25
        xi, yi = int(round(xdata)), int(round(ydata))
        y0, y1 = max(0, yi-r), min(mip.shape[0], yi+r)
        x0, x1 = max(0, xi-r), min(mip.shape[1], xi+r)
        crop = mip[y0:y1, x0:x1]
        if crop.size > 0:
            vmin, vmax = np.percentile(mips_xy[tp_idx], 1), np.percentile(mips_xy[tp_idx], 99.5)
            mag_img.set_data(crop); mag_img.set_clim(vmin, vmax)
            mag_img.set_extent([x0, x1, y1, y0])
            ax_mag.set_xlim(x0, x1); ax_mag.set_ylim(y1, y0)
            mag_vline.set_xdata([(x0+x1)/2]); mag_hline.set_ydata([(y0+y1)/2])
            fig.canvas.draw_idle()

    # ── Z-slice / display state ──
    z_slice_mode = [False]      # False = MIP, True = single Z slice
    current_z = [0]             # current Z slice (used when z_slice_mode is True)
    display_channel = [ref_ch]  # which channel is currently displayed
    show_help = [False]         # help overlay visibility
    pan_state = [None]          # (xdata_start, ydata_start, xlim_start, ylim_start, axis) when panning

    # Map from axis to tooltip text
    tooltip_text = {
        ax_slider:    "Maximum allowed in-plane rotation (default 25°)",
        ax_nslider:   "Number of landmarks for Suggest / Auto-place",
        ax_auto:      "Detect candidate landmark positions in T0",
        ax_autoplace: "Auto-detect AND place landmarks across all timepoints",
        ax_refine:    "Re-center placed landmarks on nearest bright peak",
        ax_delall:    "Clear all placed landmarks",
        ax_resetview: "Reset zoom to fit full image (R)",
        ax_chk:       "Show side-by-side preview after Done File",
        ax_help:      "Show keyboard shortcuts (?)",
        ax_mag_min:   "Minimize magnifier",
        ax_mag_restore: "Restore magnifier",
    }
    # Canvas mode thumbnails get just the mode name as tooltip
    for ax, mid in cm_ax_to_mode.items():
        labels = {'include_all': 'Include entire result',
                  'same_size': 'New size equal to current size',
                  'crop_overlap': 'Crop largest common region'}
        tooltip_text[ax] = labels[mid]

    if is_batch_mode:
        tooltip_text[ax_skip]   = "Skip this file without writing output"
        tooltip_text[ax_done]   = "Apply alignment and save corrected .ims file (D)"
        tooltip_text[ax_finish] = "Finish batch — skip remaining files"
    else:
        tooltip_text[ax_skip]   = "Cancel without writing output"
        tooltip_text[ax_done]   = "Apply alignment and save corrected .ims file (D)"
    tooltip_annot = [None]  # current tooltip annotation object

    def show_tooltip(text, x, y):
        # Remove previous tooltip
        if tooltip_annot[0] is not None:
            try: tooltip_annot[0].remove()
            except Exception: pass
            tooltip_annot[0] = None
        if text:
            tooltip_annot[0] = fig.text(x, y, text, fontsize=8, color='#FFFFFF',
                                          ha='center', va='bottom',
                                          bbox=dict(boxstyle='round,pad=0.4',
                                                    fc='#0a1a2e', ec=ACCENT,
                                                    alpha=0.95, lw=0.8))
            fig.canvas.draw_idle()

    def hide_tooltip():
        if tooltip_annot[0] is not None:
            try: tooltip_annot[0].remove()
            except Exception: pass
            tooltip_annot[0] = None
            fig.canvas.draw_idle()

    # ── Help overlay ──
    help_overlay_axes = [None]

    def toggle_help():
        if show_help[0]:
            # Hide
            if help_overlay_axes[0] is not None:
                try: help_overlay_axes[0].remove()
                except Exception: pass
                help_overlay_axes[0] = None
            show_help[0] = False
        else:
            # Show
            ax_overlay = fig.add_axes([0.15, 0.20, 0.70, 0.65])
            ax_overlay.set_facecolor('#0a1424')
            for s in ax_overlay.spines.values(): s.set_color(ACCENT); s.set_linewidth(2)
            ax_overlay.set_xticks([]); ax_overlay.set_yticks([])
            help_text = """KEYBOARD SHORTCUTS
═════════════════════════════════════════════════

PLACING LANDMARKS
  Left click .................... Place landmark at peak nearest click
  Shift + click on marker ....... Remove and re-place that specific marker
  Ctrl + Z ...................... Undo most recent landmark placement
  D ............................. Done — apply alignment and save

VIEW
  Scroll wheel .................. Zoom in / out (synced across timepoints)
  Right-click + drag ............ Pan (when zoomed in)
  R ............................. Reset zoom to full view
  M ............................. Toggle magnifier panel
  [ / ] ......................... Adjust display contrast
  Z ............................. Toggle MIP / single Z-slice mode
  , / .  (comma/period) ......... Previous / next Z slice (slice mode only)
  C ............................. Cycle through display channels

WORKFLOW
  S ............................. Skip current timepoint
  ? ............................. Toggle this help overlay
  Click outside this panel ...... Close help

BUTTONS
  Suggest ....................... Auto-detect candidate positions on T0
  Auto-place .................... Auto-detect AND match across all timepoints
  Refine ........................ Re-center existing landmarks on bright peaks
  Delete all .................... Clear all landmarks
  Reset view .................... Reset zoom to full view
  Skip File / Done File / Finish .. End-of-file actions"""
            ax_overlay.text(0.05, 0.95, help_text, transform=ax_overlay.transAxes,
                            fontsize=9, fontfamily='monospace', color='#e0e0e0',
                            va='top', ha='left')
            help_overlay_axes[0] = ax_overlay
            show_help[0] = True
        fig.canvas.draw_idle()

    btn_help.on_clicked(lambda ev: toggle_help())

    # ── Reset View handler ──
    def reset_view():
        zoom_level[0] = 1.0
        for ax in axes_xy:
            ax.set_xlim(orig_xlim); ax.set_ylim(orig_ylim)
        fig.canvas.draw_idle()

    btn_resetview.on_clicked(lambda ev: reset_view())

    # ── Z-slice rendering ──
    def render_displays():
        """Re-render all MIP/slice axes based on current display mode."""
        z_idx = max(0, min(current_z[0], volumes[0].shape[0] - 1))
        for i in range(n_tp):
            vol = volumes[i]
            if z_slice_mode[0]:
                clipped_z = max(0, min(z_idx, vol.shape[0] - 1))
                disp_xy = vol[clipped_z]
                disp_xz = vol.max(axis=1)  # XZ stays as MIP
            else:
                disp_xy = mips_xy[i]
                disp_xz = mips_xz[i]
            vmin = np.percentile(disp_xy, max(0.5, 100 - contrast_pct[0]))
            vmax = np.percentile(disp_xy, contrast_pct[0])
            if axes_xy[i].images:
                axes_xy[i].images[0].set_data(disp_xy)
                axes_xy[i].images[0].set_clim(vmin, max(vmax, 1))
        fig.canvas.draw_idle()

    # ── Live RMSE / outlier / spread analysis ──
    quality_info = [None]  # (rmse_avg, outlier_set, spread_ok)

    def compute_quality():
        """Compute predicted RMSE, identify outlier landmarks, check spread."""
        if len(all_lm) < 3:
            quality_info[0] = None
            return
        # Use lowest non-skipped TP as reference
        ref_idx = next((t for t in range(n_tp) if t not in skipped_tps[0]), 0)
        outliers = set()
        rmses = []
        for t in range(n_tp):
            if t == ref_idx or t in skipped_tps[0]: continue
            P0, Pt, lm_idx = [], [], []
            for ri, rr in enumerate(all_lm):
                if t < len(rr) and ref_idx < len(rr) and rr[t] is not None and rr[ref_idx] is not None:
                    P0.append(rr[ref_idx]); Pt.append(rr[t]); lm_idx.append(ri)
            if len(P0) < 3: continue
            try:
                P0_arr = np.array(P0, dtype=np.float64)
                Pt_arr = np.array(Pt, dtype=np.float64)
                R, tr, _ = fit_rigid(P0_arr, Pt_arr)
                pred = (P0_arr @ R.T) + tr
                resid = np.linalg.norm(Pt_arr - pred, axis=1)
                rmses.append(np.sqrt((resid**2).mean()))
                if len(resid) > 0:
                    med = np.median(resid)
                    for j, r in enumerate(resid):
                        if r > 2 * med and r > 15:
                            outliers.add(lm_idx[j])
            except Exception:
                pass
        # Spread check (reference TP positions)
        spread_ok = True
        ref_pts = [rr[ref_idx] for rr in all_lm if ref_idx < len(rr) and rr[ref_idx] is not None]
        if len(ref_pts) >= 3:
            ref_arr = np.array(ref_pts, dtype=np.float64)
            stds = ref_arr[:, 1:].std(axis=0)
            img_size = np.array([img_h, img_w])
            spread_ratio = stds / img_size
            if np.any(spread_ratio < 0.1):
                spread_ok = False
        quality_info[0] = (np.mean(rmses) if rmses else None, outliers, spread_ok)

    # ── Updated event handlers ──
    def onmove(ev):
        if closed[0]: return

        # Tooltip handling — show when hovering over labeled buttons
        if ev.inaxes in tooltip_text:
            ax = ev.inaxes
            bbox = ax.get_position()
            tx = bbox.x0 + bbox.width / 2
            ty = bbox.y1 + 0.005
            show_tooltip(tooltip_text[ax], tx, ty)
        else:
            if tooltip_annot[0] is not None:
                hide_tooltip()

        # Right-button drag = pan
        if pan_state[0] is not None and ev.xdata is not None and ev.ydata is not None:
            xs, ys, xl0, yl0, ax_pan = pan_state[0]
            if ev.inaxes is ax_pan:
                dx = ev.xdata - xs
                dy = ev.ydata - ys
                new_xlim = (xl0[0] - dx, xl0[1] - dx)
                new_ylim = (yl0[0] - dy, yl0[1] - dy)
                # Apply same pan to all XY axes for sync
                for ax in axes_xy:
                    ax.set_xlim(new_xlim)
                    ax.set_ylim(new_ylim)
                fig.canvas.draw_idle()
                return

        if ev.inaxes is None: return
        for i in range(n_tp):
            if ev.inaxes == axes_xy[i]:
                update_magnifier(ev.xdata, ev.ydata, i)
                xz_vlines[i].set_xdata([ev.xdata]); xz_vlines[i].set_alpha(0.8)
                fig.canvas.draw_idle(); break

    def onscroll(ev):
        if closed[0] or ev.inaxes is None: return
        for i in range(n_tp):
            if ev.inaxes == axes_xy[i]:
                factor = 1.25 if ev.button == 'up' else 0.8
                sync_zoom(ev.xdata, ev.ydata, factor)
                break

    edit_mode = [None]  # (round_idx, timepoint_idx) when correcting a specific position

    def do_undo():
        """Undo most recent landmark placement."""
        if cur:
            while cur and cur[-1] is None: cur.pop()
            if cur: cur.pop()
            while cur and cur[-1] is None: cur.pop()
            cur_tp[0] = len(cur)
            while cur_tp[0] < n_tp and cur_tp[0] in skipped_tps[0]:
                cur.append(None); cur_tp[0] += 1
        elif all_lm:
            last = all_lm.pop()
            rnd[0] -= 1
            while last and last[-1] is None: last.pop()
            if last: last.pop()
            while last and last[-1] is None: last.pop()
            cur.clear()
            cur.extend(last)
            cur_tp[0] = len(cur)
            while cur_tp[0] < n_tp and cur_tp[0] in skipped_tps[0]:
                cur.append(None); cur_tp[0] += 1
        refresh()

    def onclick(ev):
        if closed[0]: return

        # Help overlay open: any click outside it dismisses
        if show_help[0]:
            if ev.inaxes is not help_overlay_axes[0]:
                toggle_help()
            return

        # Canvas mode thumbnail click
        if ev.button == 1 and ev.inaxes in cm_ax_to_mode:
            set_canvas_mode(cm_ax_to_mode[ev.inaxes])
            return

        # Right-button press: start panning if in an XY axis
        if ev.button == 3 and ev.inaxes in axes_xy:
            pan_state[0] = (ev.xdata, ev.ydata,
                            ev.inaxes.get_xlim(), ev.inaxes.get_ylim(),
                            ev.inaxes)
            return

        ctp = None
        for i in range(n_tp):
            if ev.inaxes == axes_xy[i]: ctp = i; break
        if ctp is None: return

        shift_held = hasattr(ev, 'guiEvent') and ev.guiEvent and (ev.guiEvent.state & 0x1)
        if ev.button == 1 and shift_held:
            # Shift+click = remove nearest marker, enter edit mode
            click_y, click_x = ev.ydata, ev.xdata
            best_ri, best_dist = None, float('inf')
            for ri, rr in enumerate(all_lm):
                if ctp < len(rr) and rr[ctp] is not None:
                    z, y, x = rr[ctp]
                    dist = np.sqrt((y - click_y)**2 + (x - click_x)**2)
                    if dist < best_dist and dist < 25:
                        best_ri, best_dist = ri, dist
            if best_ri is not None:
                all_lm[best_ri][ctp] = None
                edit_mode[0] = (best_ri, ctp)
                print(f"  Removed L{best_ri+1} at {day_labels[ctp]} — click to re-place")
                refresh()
            else:
                print(f"  No marker found near click")
            return

        if ev.button == 1:
            if edit_mode[0] is not None:
                ri, ti = edit_mode[0]
                if ctp != ti:
                    print(f"  → Click in {day_labels[ti]} to replace L{ri+1}"); return
                xc, yc = int(round(ev.xdata)), int(round(ev.ydata))
                vol = volumes[ctp]
                xc = max(0, min(xc, vol.shape[2]-1)); yc = max(0, min(yc, vol.shape[1]-1))
                sr = 5
                yl, yh = max(0, yc-sr), min(vol.shape[1], yc+sr+1)
                xl, xh = max(0, xc-sr), min(vol.shape[2], xc+sr+1)
                local = vol[:, yl:yh, xl:xh]
                pk = np.unravel_index(local.argmax(), local.shape)
                zc, yc, xc = pk[0], yl+pk[1], xl+pk[2]
                all_lm[ri][ti] = (zc, yc, xc)
                edit_mode[0] = None
                print(f"  L{ri+1} {day_labels[ti]}: corrected → z={zc}, y={yc}, x={xc}")
                refresh()
                return

            # Normal placement
            if ctp != cur_tp[0]:
                print(f"  → Click in {day_labels[cur_tp[0]]}"); return
            if ctp in skipped_tps[0]:
                print(f"  → {day_labels[ctp]} is skipped"); return
            xc, yc = int(round(ev.xdata)), int(round(ev.ydata))
            vol = volumes[ctp]
            xc = max(0, min(xc, vol.shape[2]-1)); yc = max(0, min(yc, vol.shape[1]-1))
            sr = 5
            yl, yh = max(0, yc-sr), min(vol.shape[1], yc+sr+1)
            xl, xh = max(0, xc-sr), min(vol.shape[2], xc+sr+1)
            local = vol[:, yl:yh, xl:xh]
            pk = np.unravel_index(local.argmax(), local.shape)
            zc, yc, xc = pk[0], yl+pk[1], xl+pk[2]
            cur.append((zc, yc, xc))
            print(f"  L{rnd[0]} {day_labels[ctp]}: z={zc}, y={yc}, x={xc}")
            xz_hlines[ctp].set_ydata([zc]); xz_hlines[ctp].set_alpha(0.8)
            cur_tp[0] += 1
            advance_to_next_valid_tp()
            refresh()

    def on_release(ev):
        # End panning on right-button release
        if ev.button == 3 and pan_state[0] is not None:
            pan_state[0] = None

    def onkey(ev):
        if show_help[0] and ev.key not in ('?',):
            toggle_help(); return

        if ev.key in ('d', 'D'):
            close_with_action("done")
        elif ev.key == 'ctrl+z':
            do_undo()
        elif ev.key in ('m', 'M'):
            mag_visible[0] = not mag_visible[0]
            ax_mag.set_visible(mag_visible[0])
            fig.canvas.draw_idle()
        elif ev.key in ('r', 'R'):
            reset_view()
        elif ev.key == '?':
            toggle_help()
        elif ev.key in ('z', 'Z'):
            z_slice_mode[0] = not z_slice_mode[0]
            if z_slice_mode[0]:
                current_z[0] = volumes[0].shape[0] // 2  # start at middle
                print(f"  Z-slice mode ON  ·  z={current_z[0]}/{volumes[0].shape[0]-1}  ·  use , and . to scrub")
            else:
                print(f"  MIP mode ON")
            render_displays()
        elif ev.key == ',' and z_slice_mode[0]:
            current_z[0] = max(0, current_z[0] - 1)
            print(f"  Z = {current_z[0]} / {volumes[0].shape[0]-1}")
            render_displays()
        elif ev.key == '.' and z_slice_mode[0]:
            current_z[0] = min(volumes[0].shape[0] - 1, current_z[0] + 1)
            print(f"  Z = {current_z[0]} / {volumes[0].shape[0]-1}")
            render_displays()
        elif ev.key in ('c', 'C'):
            display_channel[0] = (display_channel[0] + 1) % n_ch_total
            print(f"  Display channel: {display_channel[0] + 1} / {n_ch_total}")
            reload_channel(display_channel[0])
        elif ev.key == ']':
            contrast_pct[0] = max(90.0, contrast_pct[0] - 2.0)
            for i in range(n_tp):
                vmax = np.percentile(mips_xy[i], contrast_pct[0])
                vmin = np.percentile(mips_xy[i], 100 - contrast_pct[0])
                if axes_xy[i].images: axes_xy[i].images[0].set_clim(vmin, max(vmax, 1))
                vmax_z = np.percentile(mips_xz[i], contrast_pct[0])
                if axes_xz[i].images: axes_xz[i].images[0].set_clim(0, max(vmax_z, 1))
            print(f"  Contrast: {contrast_pct[0]:.0f}th percentile")
            fig.canvas.draw_idle()
        elif ev.key == '[':
            contrast_pct[0] = min(100.0, contrast_pct[0] + 2.0)
            for i in range(n_tp):
                vmax = np.percentile(mips_xy[i], contrast_pct[0])
                vmin = np.percentile(mips_xy[i], max(0.5, 100 - contrast_pct[0]))
                if axes_xy[i].images: axes_xy[i].images[0].set_clim(vmin, max(vmax, 1))
                vmax_z = np.percentile(mips_xz[i], contrast_pct[0])
                if axes_xz[i].images: axes_xz[i].images[0].set_clim(0, max(vmax_z, 1))
            print(f"  Contrast: {contrast_pct[0]:.0f}th percentile")
            fig.canvas.draw_idle()
        elif ev.key in ('s', 'S'):
            tp = cur_tp[0]
            if tp >= n_tp: return
            # Don't allow ALL timepoints to be skipped
            n_after_skip = n_tp - len(skipped_tps[0]) - (1 if tp not in skipped_tps[0] else -1)
            if tp not in skipped_tps[0] and n_after_skip < 2:
                print("  At least 2 timepoints required for alignment"); return
            if tp in skipped_tps[0]:
                skipped_tps[0].discard(tp)
                print(f"  {day_labels[tp]}: UN-SKIPPED")
            else:
                skipped_tps[0].add(tp)
                print(f"  {day_labels[tp]}: SKIPPED")
                # Clear any landmarks placed at this TP
                for rr in all_lm:
                    if tp < len(rr):
                        rr[tp] = None
                if len(cur) == tp:
                    cur.append(None); cur_tp[0] += 1
                    advance_to_next_valid_tp()
            refresh()

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('key_press_event', onkey)
    fig.canvas.mpl_connect('motion_notify_event', onmove)
    fig.canvas.mpl_connect('scroll_event', onscroll)
    refresh(); plt.show()

    # Reset matplotlib defaults after closing
    plt.rcdefaults()
    return result[0], result[1], skipped_tps[0], max_rot[0], show_preview[0], canvas_mode[0]


# ── LOCAL REFINEMENT (Sub-voxel via Phase Correlation) ───────

def extract_local_cube(vol, center_zyx, radius_zyx=(10, 30, 30)):
    """Extract a local 3D cube, handling edge boundaries."""
    z, y, x = center_zyx
    rz, ry, rx = radius_zyx
    nz, ny, nx = vol.shape

    z0, z1 = max(0, z - rz), min(nz, z + rz + 1)
    y0, y1 = max(0, y - ry), min(ny, y + ry + 1)
    x0, x1 = max(0, x - rx), min(nx, x + rx + 1)

    return vol[z0:z1, y0:y1, x0:x1].astype(np.float32)


def refine_position(vol_ref, vol_mov, pos_ref, pos_mov,
                    radius=(10, 30, 30), upsample=10, max_move=15.0):
    """
    Refine the clicked position in vol_mov using local phase correlation.

    Only accepts the refinement if:
    - The local cubes have enough signal (std > 1.0)
    - The detected shift is < max_move voxels (prevents noise-driven jumps)
    """
    cube_ref = extract_local_cube(vol_ref, pos_ref, radius)
    cube_mov = extract_local_cube(vol_mov, pos_mov, radius)

    # Ensure cubes are the same size
    min_shape = tuple(min(a, b) for a, b in zip(cube_ref.shape, cube_mov.shape))
    cube_ref = cube_ref[:min_shape[0], :min_shape[1], :min_shape[2]]
    cube_mov = cube_mov[:min_shape[0], :min_shape[1], :min_shape[2]]

    if cube_ref.size < 100 or cube_mov.size < 100:
        return pos_mov, 0.0, "too_small"

    # Skip if either cube has very little signal
    ref_std = cube_ref.std()
    mov_std = cube_mov.std()
    if ref_std < 1.0 or mov_std < 1.0:
        return pos_mov, 0.0, "low_signal"

    try:
        shift, error, diffphase = phase_cross_correlation(
            cube_ref, cube_mov,
            upsample_factor=upsample,
            space="real"
        )

        move_mag = np.sqrt(shift[0]**2 + shift[1]**2 + shift[2]**2)

        # Reject if the refinement wants to move too far (noise)
        if move_mag > max_move:
            return pos_mov, 0.0, f"rejected(move={move_mag:.1f}>{max_move})"

        refined_z = pos_mov[0] - shift[0]
        refined_y = pos_mov[1] - shift[1]
        refined_x = pos_mov[2] - shift[2]

        # Clamp to volume bounds
        refined_z = max(0, min(refined_z, vol_mov.shape[0] - 1))
        refined_y = max(0, min(refined_y, vol_mov.shape[1] - 1))
        refined_x = max(0, min(refined_x, vol_mov.shape[2] - 1))

        return (refined_z, refined_y, refined_x), move_mag, "ok"

    except Exception:
        return pos_mov, 0.0, "error"


def refine_all_landmarks(all_landmarks, volumes, log, days,
                          radius=(10, 30, 30)):
    """
    Refine all landmark positions using local phase correlation.
    Only applies refinement when the local correlation is reliable.
    """
    n_tp = len(volumes)
    refined = []

    log.log(f"\n  Refining landmarks (local phase correlation, radius={radius}):")

    for r, rnd in enumerate(all_landmarks):
        refined_rnd = [rnd[0]]  # T0 stays as-is

        for t in range(1, len(rnd)):
            pos_ref = rnd[0]
            pos_clicked = rnd[t]

            # Skip None entries (skipped timepoints)
            if pos_clicked is None or pos_ref is None:
                refined_rnd.append(None)
                continue

            refined_pos, move_mag, status = refine_position(
                volumes[0], volumes[t],
                pos_ref, pos_clicked,
                radius=radius
            )

            refined_rnd.append(refined_pos)

            if status == "ok" and move_mag > 0.5:
                log.log(f"    L{r+1} {days[t]}: refined {move_mag:.1f}vox "
                        f"({pos_clicked[0]:.0f},{pos_clicked[1]:.0f},{pos_clicked[2]:.0f}) -> "
                        f"({refined_pos[0]:.1f},{refined_pos[1]:.1f},{refined_pos[2]:.1f})")
            elif status == "ok":
                log.log(f"    L{r+1} {days[t]}: no change needed")
            else:
                log.log(f"    L{r+1} {days[t]}: kept raw click ({status})")

        refined.append(refined_rnd)

    return refined


# ── RIGID BODY TRANSFORM WITH OUTLIER REJECTION ──────────────

def fit_rigid(pts_src, pts_tgt):
    """SVD Procrustes: target ≈ R @ source + t"""
    src, tgt = np.array(pts_src, dtype=np.float64), np.array(pts_tgt, dtype=np.float64)
    cs, ct = src.mean(0), tgt.mean(0)
    if len(src) >= 3:
        H = (src-cs).T @ (tgt-ct)
        U,S,Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        R = Vt.T @ np.diag([1,1,np.sign(d)]) @ U.T
    else:
        R = np.eye(3)
    t = ct - R @ cs
    xformed = (R @ src.T).T + t
    errs = np.sqrt(np.sum((xformed - tgt)**2, axis=1))
    return R, t, np.sqrt(np.mean(errs**2)), errs


def cap_rotation(R, max_deg=3.0):
    """
    Cap rotation to protect thin Z stacks.

    In array coordinates (0=Z, 1=Y, 2=X):
    - rx = rotation around axis 0 (Z axis): only moves Y,X → SAFE, allow up to max_deg
    - ry = rotation around axis 1 (Y axis): tips Z vs X → MUST CAP to protect Z
    - rz = rotation around axis 2 (X axis): tips Z vs Y → MUST CAP to protect Z

    sin(0.5°) × 256 = 2.2 voxels max Z displacement (acceptable for 112-slice stack)
    sin(3°) × 256 = 13 voxels Z displacement (destroys 24% of Z stack!)
    """
    rx, ry, rz = decompose_rot(R)

    max_tilt = 2.0   # Max ry/rz (out-of-plane tilt) — sin(2°)×N/2 vox Z displacement at edges
    max_spin = max_deg  # Max rx (in-plane spin, safe for Z)

    capped = False
    # rx is SAFE — rotation in XY plane
    if abs(rx) > max_spin:
        rx = max_spin * np.sign(rx); capped = True
    # ry and rz are DANGEROUS — they tip the thin slab
    if abs(ry) > max_tilt:
        ry = max_tilt * np.sign(ry); capped = True
    if abs(rz) > max_tilt:
        rz = max_tilt * np.sign(rz); capped = True

    if not capped:
        return R, False

    rx_r, ry_r, rz_r = np.radians(rx), np.radians(ry), np.radians(rz)
    Rx = np.array([[1,0,0],[0,np.cos(rx_r),-np.sin(rx_r)],[0,np.sin(rx_r),np.cos(rx_r)]])
    Ry = np.array([[np.cos(ry_r),0,np.sin(ry_r)],[0,1,0],[-np.sin(ry_r),0,np.cos(ry_r)]])
    Rz = np.array([[np.cos(rz_r),-np.sin(rz_r),0],[np.sin(rz_r),np.cos(rz_r),0],[0,0,1]])
    return Rz @ Ry @ Rx, True


def fit_rigid_robust(pts_src, pts_tgt, log, label="", max_rot_deg=3.0):
    """
    Rigid body fit with outlier rejection and rotation capping.
    1. Fit rigid body from all landmarks
    2. Remove outliers iteratively
    3. Cap rotation to max_rot_deg per axis
    4. Recompute translation after capping
    """
    src = list(pts_src)
    tgt = list(pts_tgt)
    indices = list(range(len(src)))
    rejected = []

    while True:
        if len(src) >= 3:
            R, t, rmse, errs = fit_rigid(src, tgt)
        else:
            # Translation only fallback
            shifts = np.array(tgt) - np.array(src)
            t = np.median(shifts, axis=0)
            R = np.eye(3)
            errs = np.sqrt(np.sum((shifts - t)**2, axis=1))
            rmse = np.sqrt(np.mean(errs**2))

        if len(src) <= 2 or rmse < 10:
            break

        # Check for outliers
        median_err = np.median(errs)
        worst_idx = np.argmax(errs)
        worst_err = errs[worst_idx]

        if worst_err > max(2 * median_err, 15) and len(src) > 2:
            rejected.append((indices[worst_idx], worst_err))
            log.log(f"      Rejected L{indices[worst_idx]+1} (residual={worst_err:.1f} vox, "
                    f"threshold={max(2*median_err, 15):.1f})")
            del src[worst_idx]
            del tgt[worst_idx]
            del indices[worst_idx]
        else:
            break

    mode = "rigid body" if len(src) >= 3 else "translation only"

    # Cap rotation to prevent noisy estimates from destroying the image
    if len(src) >= 3 and not np.allclose(R, np.eye(3)):
        R_capped, was_capped = cap_rotation(R, max_rot_deg)
        if was_capped:
            rx_old = decompose_rot(R)
            rx_new = decompose_rot(R_capped)
            log.log(f"      Rotation capped: rx(XY-spin)={rx_old[0]:+.1f}->{rx_new[0]:+.1f}, "
                    f"ry(tilt)={rx_old[1]:+.1f}->{rx_new[1]:+.1f}, "
                    f"rz(tilt)={rx_old[2]:+.1f}->{rx_new[2]:+.1f} deg")
            R = R_capped
            # Recompute translation with capped rotation
            src_arr, tgt_arr = np.array(src), np.array(tgt)
            t = tgt_arr.mean(0) - R @ src_arr.mean(0)
            # Recompute residuals
            xformed = (R @ src_arr.T).T + t
            errs = np.sqrt(np.sum((xformed - tgt_arr)**2, axis=1))
            rmse = np.sqrt(np.mean(errs**2))
            mode = "rigid body (rotation capped)"

    return R, t, rmse, errs, indices, rejected, mode


def decompose_rot(R):
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        return (np.degrees(np.arctan2(R[2,1],R[2,2])),
                np.degrees(np.arctan2(-R[2,0],sy)),
                np.degrees(np.arctan2(R[1,0],R[0,0])))
    return (np.degrees(np.arctan2(-R[1,2],R[1,1])),
            np.degrees(np.arctan2(-R[2,0],sy)), 0)


# ── CANVAS EXPANSION ─────────────────────────────────────────

def compute_canvas(orig_shape, transforms, canvas_mode="include_all"):
    """
    canvas_mode:
      "include_all"  - expand canvas so all data fits with padding (default, no data lost)
      "same_size"    - keep original dimensions; data shifted past edges is clipped
      "crop_overlap" - smallest box containing region all timepoints overlap
    """
    nz, ny, nx = orig_shape
    corners = np.array([[0,0,0],[0,0,nx],[0,ny,0],[0,ny,nx],
                         [nz,0,0],[nz,0,nx],[nz,ny,0],[nz,ny,nx]], dtype=np.float64)
    
    tp_mins, tp_maxs = [], []
    for R, t, *_ in transforms:
        if np.allclose(R, np.eye(3)):
            smin, smax = -t, np.array(orig_shape, dtype=np.float64) - t
        else:
            center = np.array(orig_shape, dtype=np.float64) / 2.0
            R_inv = R.T; off = center - R_inv@center - R_inv@t
            tc = (R_inv @ corners.T).T + off
            smin, smax = tc.min(0), tc.max(0)
        tp_mins.append(smin); tp_maxs.append(smax)
    tp_mins = np.array(tp_mins); tp_maxs = np.array(tp_maxs)
    
    if canvas_mode == "same_size":
        return orig_shape, np.zeros(3)
    
    if canvas_mode == "crop_overlap":
        overlap_min = tp_mins.max(axis=0)
        overlap_max = tp_maxs.min(axis=0)
        overlap_min = np.maximum(overlap_min, np.zeros(3))
        overlap_max = np.minimum(overlap_max, np.array(orig_shape, dtype=np.float64))
        if np.any(overlap_max - overlap_min <= 0):
            print("  Warning: timepoints don't overlap; falling back to same-size mode")
            return orig_shape, np.zeros(3)
        new_shape = tuple(int(np.floor(mx-mn)) for mn,mx in zip(overlap_min, overlap_max))
        return new_shape, -overlap_min
    
    # include_all (default)
    all_min = tp_mins.min(axis=0); all_max = tp_maxs.max(axis=0)
    all_min = np.minimum(all_min, np.zeros(3))
    all_max = np.maximum(all_max, np.array(orig_shape, dtype=np.float64))
    new_shape = tuple(int(np.ceil(mx-mn))+2 for mn,mx in zip(all_min, all_max))
    pad_z = max(56, int(orig_shape[0] * 0.5))
    pad_y = max(100, int(orig_shape[1] * 0.2))
    pad_x = max(100, int(orig_shape[2] * 0.2))
    new_shape = (new_shape[0] + pad_z, new_shape[1] + pad_y, new_shape[2] + pad_x)
    base_offset = -all_min
    base_offset[0] += pad_z // 2
    base_offset[1] += pad_y // 2
    base_offset[2] += pad_x // 2
    return new_shape, base_offset


def apply_correction(vol, R, t, orig_shape, new_shape, base_offset, order=1):
    center = np.array(orig_shape, dtype=np.float64) / 2.0
    combined_offset = center - R@center + t - R@base_offset
    return affine_transform(vol.astype(np.float64), R, offset=combined_offset,
                            output_shape=new_shape, order=order, mode='constant', cval=0.0
                            ).astype(vol.dtype)


def place_t0(vol, new_shape, base_offset):
    out = np.zeros(new_shape, dtype=vol.dtype)
    oz,oy,ox = [int(round(x)) for x in base_offset]
    nz,ny,nx = vol.shape
    z0,z1 = max(0,oz), min(new_shape[0],oz+nz)
    y0,y1 = max(0,oy), min(new_shape[1],oy+ny)
    x0,x1 = max(0,ox), min(new_shape[2],ox+nx)
    out[z0:z1,y0:y1,x0:x1] = vol[z0-oz:z1-oz, y0-oy:y1-oy, x0-ox:x1-ox]
    return out


def refine_residual_drift(vols, transforms, orig_shape, new_shape, base_offset,
                           skipped_tps, log, days, all_landmarks=None, ref_tp_idx=0):
    """
    Two-pass refinement using landmark-guided phase correlation.
    
    Instead of full-volume phase correlation (which fails on sparse images),
    extracts large cubes around each landmark in the coarse-aligned volumes
    and takes the median shift. This focuses on regions with actual signal
    and is robust even when the coarse correction left large residuals.
    """
    log.log(f"\n  Residual drift refinement (landmark-guided phase correlation):")

    # Apply coarse correction to get roughly-aligned reference channel
    aligned_vols = []
    for t in range(len(vols)):
        if t in skipped_tps:
            aligned_vols.append(None)
            continue
        R, tr, _ = transforms[t]
        if t == ref_tp_idx:
            aligned_vols.append(place_t0(vols[t], new_shape, base_offset))
        else:
            aligned_vols.append(apply_correction(vols[t], R, tr, orig_shape, new_shape, base_offset))

    ref_vol = aligned_vols[ref_tp_idx].astype(np.float32)

    # Compute landmark positions in the expanded canvas (ref TP positions + base_offset)
    lm_positions = []
    if all_landmarks:
        for rr in all_landmarks:
            if ref_tp_idx < len(rr) and rr[ref_tp_idx] is not None:
                z, y, x = rr[ref_tp_idx]
                lm_positions.append((
                    int(round(z + base_offset[0])),
                    int(round(y + base_offset[1])),
                    int(round(x + base_offset[2]))
                ))

    refined_transforms = list(transforms)  # start with originals; we'll patch non-ref non-skipped TPs

    for t in range(len(vols)):
        if t == ref_tp_idx: continue
        R, tr, rmse = transforms[t]

        if t in skipped_tps:
            continue
        
        mov_vol = aligned_vols[t].astype(np.float32)
        
        # Strategy 1: Landmark-guided (preferred — works on sparse images)
        if lm_positions and len(lm_positions) >= 2:
            cube_radius = (25, 60, 60)  # Large cubes for robust correlation
            shifts_per_lm = []
            
            for li, (lz, ly, lx) in enumerate(lm_positions):
                try:
                    cube_ref = extract_local_cube(ref_vol, (lz, ly, lx), cube_radius)
                    cube_mov = extract_local_cube(mov_vol, (lz, ly, lx), cube_radius)
                    
                    min_shape = tuple(min(a, b) for a, b in zip(cube_ref.shape, cube_mov.shape))
                    cube_ref = cube_ref[:min_shape[0], :min_shape[1], :min_shape[2]]
                    cube_mov = cube_mov[:min_shape[0], :min_shape[1], :min_shape[2]]
                    
                    if cube_ref.std() < 0.5 or cube_mov.std() < 0.5:
                        continue
                    
                    shift, error, _ = phase_cross_correlation(
                        cube_ref, cube_mov, upsample_factor=10, space="real"
                    )
                    
                    shift_mag = np.sqrt(shift[0]**2 + shift[1]**2 + shift[2]**2)
                    if shift_mag < 80:  # Reject obvious outliers
                        shifts_per_lm.append(shift)
                except Exception:
                    continue
            
            if shifts_per_lm:
                # Take median shift across all landmarks (robust to outliers)
                shifts_arr = np.array(shifts_per_lm)
                median_shift = np.median(shifts_arr, axis=0)
                residual_mag = np.sqrt(np.sum(median_shift**2))
                
                # Also compute spread to assess reliability
                spread = np.std(np.sqrt(np.sum(shifts_arr**2, axis=1)))
                
                if 0.5 < residual_mag < 80:
                    tr_refined = tr - R @ median_shift
                    
                    log.log(f"    {days[t]}: residual dz={median_shift[0]:+.1f}, "
                            f"dy={median_shift[1]:+.1f}, dx={median_shift[2]:+.1f}  "
                            f"({residual_mag:.1f} vox, {len(shifts_per_lm)} landmarks, "
                            f"spread={spread:.1f}) → corrected")
                    refined_transforms[t] = (R, tr_refined, rmse)
                    continue
                elif residual_mag <= 0.5:
                    log.log(f"    {days[t]}: residual <0.5 vox — already well-aligned "
                            f"({len(shifts_per_lm)} landmarks)")
                    continue
                    continue
        
        # Strategy 2: Fallback to full-volume phase correlation
        # (only if landmark-guided didn't produce a result)
        try:
            # Crop to data-containing region to avoid noise in padding
            ref_mask = ref_vol > 0
            mov_mask = mov_vol > 0
            overlap = ref_mask | mov_mask
            
            if overlap.sum() < 1000:
                log.log(f"    {days[t]}: insufficient data for refinement")
                continue
                continue
            
            # Find bounding box of data
            nz_idx = np.where(overlap.any(axis=(1,2)))[0]
            ny_idx = np.where(overlap.any(axis=(0,2)))[0]
            nx_idx = np.where(overlap.any(axis=(0,1)))[0]
            
            if len(nz_idx) == 0:
                continue
                continue
            
            z0, z1 = nz_idx[0], nz_idx[-1] + 1
            y0, y1 = ny_idx[0], ny_idx[-1] + 1
            x0, x1 = nx_idx[0], nx_idx[-1] + 1
            
            ref_crop = ref_vol[z0:z1, y0:y1, x0:x1]
            mov_crop = mov_vol[z0:z1, y0:y1, x0:x1]
            
            shift, error, _ = phase_cross_correlation(
                ref_crop, mov_crop, upsample_factor=10, space="real"
            )
            
            residual_mag = np.sqrt(shift[0]**2 + shift[1]**2 + shift[2]**2)
            
            if 0.5 < residual_mag < 80:
                tr_refined = tr - R @ shift
                log.log(f"    {days[t]}: residual dz={shift[0]:+.1f}, dy={shift[1]:+.1f}, "
                        f"dx={shift[2]:+.1f}  ({residual_mag:.1f} vox, full-volume fallback) → corrected")
                refined_transforms[t] = (R, tr_refined, rmse)
            else:
                if residual_mag <= 0.5:
                    log.log(f"    {days[t]}: residual <0.5 vox — already well-aligned")
                else:
                    log.log(f"    {days[t]}: residual {residual_mag:.1f} vox — unreliable, skipping")
                continue
                
        except Exception as e:
            log.log(f"    {days[t]}: refinement failed ({e}), keeping original")
            continue
    
    return refined_transforms


# ── FILE PROCESSING ──────────────────────────────────────────

def get_days(fn):
    m = re.search(r'_d(\d+(?:\+d\d+)*)', fn, re.IGNORECASE)
    return [d.upper() for d in m.group(0)[1:].split('+')] if m else None


def process_file(ims_path, output_path, log, ref_ch=0, trans_only=False, max_rot=25.0,
                  batch_info=None, is_batch_mode=True):
    log.log(f"  Opening: {ims_path.name}")
    with h5py.File(str(ims_path), "r") as h5:
        info = get_ims_info(h5)
        n_tp, n_ch, orig_shape = info["n_tp"], info["n_ch"], info["shape"]
        log.log(f"  Dims: {n_tp}T x {n_ch}C x {orig_shape[0]}Z x {orig_shape[1]}Y x {orig_shape[2]}X")
        if n_tp < 2: shutil.copy2(str(ims_path), str(output_path)); return True
        log.log(f"  Reading channel {ref_ch+1}...")
        vols = [read_vol(h5, t, ref_ch) for t in range(n_tp)]
        if any(v is None for v in vols): return False
        # Also read other channels for the channel-cycling display feature
        # (skip if memory is a concern — can be disabled)
        all_channel_volumes = None
        if n_ch > 1 and n_ch <= 4:
            try:
                log.log(f"  Reading {n_ch} channels for display cycling...")
                all_channel_volumes = []
                for t in range(n_tp):
                    ch_vols = []
                    for c in range(n_ch):
                        if c == ref_ch:
                            ch_vols.append(vols[t])  # reuse already-loaded
                        else:
                            ch_vols.append(read_vol(h5, t, c))
                    all_channel_volumes.append(ch_vols)
            except Exception as e:
                log.log(f"  (Could not load all channels — channel cycling disabled: {e})")
                all_channel_volumes = None
        days = get_days(ims_path.name)
        if not days or len(days)!=n_tp: days=[f"T{t}" for t in range(n_tp)]
        log.log(f"  Timepoints: {', '.join(days)}")

    all_lm, action, skipped_tps, max_rot, do_preview, canvas_mode = pick_landmarks(
        vols, ims_path.name, days, max_rot_default=max_rot,
        ref_ch=ref_ch, all_channel_volumes=all_channel_volumes,
        batch_info=batch_info, is_batch_mode=is_batch_mode)

    if action in ("skip", "finish"):
        log.log(f"  {'Skipped by user' if action=='skip' else 'Batch finished by user'}")
        return action

    if not all_lm:
        log.log(f"  No landmarks."); return False

    if skipped_tps:
        log.log(f"\n  Skipped timepoints: {', '.join(days[t] for t in sorted(skipped_tps))}")

    log.log(f"\n  Placed {len(all_lm)} landmark(s) (raw clicks):")
    for r,rr in enumerate(all_lm):
        parts = []
        for t,(pos) in enumerate(rr):
            if pos is None:
                parts.append(f"{days[t]}: SKIP")
            else:
                z,y,x = pos
                parts.append(f"{days[t]}: z={z},y={y},x={x}")
        log.log(f"    L{r+1}: " + "  |  ".join(parts))

    # Refine landmarks (skip None entries)
    all_lm = refine_all_landmarks(all_lm, vols, log, days, radius=(15, 50, 50))

    log.log(f"\n  Refined {len(all_lm)} landmark(s):")
    for r,rr in enumerate(all_lm):
        parts = []
        for t, pos in enumerate(rr):
            if pos is None:
                parts.append(f"{days[t]}: SKIP")
            else:
                z,y,x = pos
                parts.append(f"{days[t]}: z={z:.1f},y={y:.1f},x={x:.1f}")
        log.log(f"    L{r+1}: " + "  |  ".join(parts))

    # Show confidence overlay
    LM_COLORS = ['#FF4D4D','#4DA6FF','#4DFF88','#FFD74D','#D94DFF',
                 '#FF944D','#4DFFF0','#FF4DA6','#A6FF4D','#4D6AFF']
    mips_xy = [v.max(axis=0) for v in vols]
    confidences = compute_landmark_confidence(all_lm, vols)
    log.log(f"\n  Landmark confidence:")
    for ri, (rr, conf) in enumerate(zip(all_lm, confidences)):
        parts = []
        for t in range(len(conf)):
            if rr[t] is not None:
                parts.append(f"{days[t]}={conf[t]:.0%}")
        log.log(f"    L{ri+1}: {', '.join(parts)}")
    show_confidence_overlay(all_lm, confidences, mips_xy, days, 
                            lambda idx: LM_COLORS[idx % len(LM_COLORS)])

    mode = "Translation only" if trans_only else "Rigid body with outlier rejection"
    log.log(f"\n  Mode: {mode}")

    # Determine reference timepoint (lowest non-skipped index)
    ref_tp_idx = next((t for t in range(n_tp) if t not in skipped_tps), 0)
    if ref_tp_idx != 0:
        log.log(f"  Reference timepoint: {days[ref_tp_idx]} (T0 was skipped)")
    log.log(f"  Computing transforms:")

    transforms = [(np.eye(3), np.zeros(3), 0.0)] * n_tp  # initialize all as identity
    transforms[ref_tp_idx] = (np.eye(3), np.zeros(3), 0.0)

    for t in range(n_tp):
        if t == ref_tp_idx: continue
        if t in skipped_tps:
            log.log(f"    {days[t]}: SKIPPED")
            continue

        pts_ref = [rr[ref_tp_idx] for rr in all_lm
                   if ref_tp_idx<len(rr) and t<len(rr)
                   and rr[ref_tp_idx] is not None and rr[t] is not None]
        ptst    = [rr[t]          for rr in all_lm
                   if ref_tp_idx<len(rr) and t<len(rr)
                   and rr[ref_tp_idx] is not None and rr[t] is not None]
        if not pts_ref:
            log.log(f"    {days[t]}: no matched landmarks, identity transform")
            continue

        log.log(f"    {days[t]}:")
        if trans_only:
            shifts = np.array(ptst, dtype=np.float64) - np.array(pts_ref, dtype=np.float64)
            tr = np.median(shifts, axis=0)
            R = np.eye(3); errs = np.sqrt(np.sum((shifts-tr)**2,axis=1))
            rmse = np.sqrt(np.mean(errs**2))
            used_idx = list(range(len(pts_ref))); rejected = []; fit_mode = "translation only"
        else:
            R, tr, rmse, errs, used_idx, rejected, fit_mode = fit_rigid_robust(pts_ref, ptst, log, days[t], max_rot)

        transforms[t] = (R, tr, rmse)
        rx,ry,rz = decompose_rot(R)
        mag_t = np.linalg.norm(tr)
        mag_r = np.sqrt(rx**2+ry**2+rz**2)

        log.log(f"      Mode: {fit_mode} ({len(used_idx)} landmarks used" +
                (f", {len(rejected)} rejected)" if rejected else ")"))
        log.log(f"      Translation: dz={tr[0]:+.1f}, dy={tr[1]:+.1f}, dx={tr[2]:+.1f}  ({mag_t:.1f} vox)")
        if mag_r > 0.01:
            log.log(f"      Rotation: rx={rx:+.2f}, ry={ry:+.2f}, rz={rz:+.2f}  ({mag_r:.2f} deg)")
        log.log(f"      RMSE: {rmse:.1f} vox")
        for j, e in zip(used_idx, errs):
            log.log(f"        L{j+1}: {e:.1f} vox")

    # Only compute canvas for non-skipped timepoints
    valid_transforms = [transforms[t] for t in range(n_tp) if t not in skipped_tps]
    new_shape, base_off = compute_canvas(orig_shape, valid_transforms, canvas_mode=canvas_mode)
    canvas_label = {'include_all': 'Include entire result',
                    'same_size': 'Same size as input',
                    'crop_overlap': 'Crop overlap region'}.get(canvas_mode, canvas_mode)
    log.log(f"\n  Output canvas mode: {canvas_label}")
    log.log(f"  Canvas: {orig_shape} -> {new_shape}")

    # Two-pass residual refinement using phase correlation on coarse-aligned volumes
    transforms = refine_residual_drift(vols, transforms, orig_shape, new_shape, base_off,
                                        skipped_tps, log, days, all_lm, ref_tp_idx=ref_tp_idx)

    # Build list of valid (non-skipped) timepoint indices
    valid_tps = [t for t in range(n_tp) if t not in skipped_tps]
    n_valid = len(valid_tps)

    log.log(f"\n  Summary ({n_valid} timepoints, {len(skipped_tps)} skipped):")
    for t in range(n_tp):
        if t in skipped_tps:
            log.log(f"    {days[t]}: SKIPPED")
        else:
            R,tr,rmse = transforms[t]
            rx,ry,rz = decompose_rot(R)
            log.log(f"    {days[t]}: T={np.linalg.norm(tr):.1f}vox  R={np.sqrt(rx**2+ry**2+rz**2):.1f}deg  RMSE={rmse:.1f}")

    # Before/after preview (only if user enabled checkbox)
    if do_preview:
        log.log(f"\n  Showing alignment preview...")
        show_before_after_preview(vols, transforms, orig_shape, new_shape, base_off,
                                  days, skipped_tps, valid_tps)

    log.log(f"\n  Applying corrections...")
    shutil.copy2(str(ims_path), str(output_path))
    with h5py.File(str(output_path), "a") as h5o:
        # If timepoints were skipped, we need to rewrite with fewer TimePoints
        if skipped_tps:
            # Read all valid data first
            all_data = {}  # (new_t, c) -> expanded_vol
            for new_t, orig_t in enumerate(valid_tps):
                R, tr, _ = transforms[orig_t]
                for c in range(n_ch):
                    vol = read_vol(h5o, orig_t, c)
                    if vol is None: continue
                    if orig_t == ref_tp_idx:
                        all_data[(new_t, c)] = place_t0(vol, new_shape, base_off)
                    else:
                        all_data[(new_t, c)] = apply_correction(vol, R, tr, orig_shape, new_shape, base_off)

            # Delete all old TimePoint groups at all resolution levels
            dataset = h5o["DataSet"]
            n_res = 0
            while f"ResolutionLevel {n_res}" in dataset: n_res += 1
            for res in range(n_res):
                res_group = dataset[f"ResolutionLevel {res}"]
                for t in range(n_tp):
                    tp_key = f"TimePoint {t}"
                    if tp_key in res_group:
                        del res_group[tp_key]

            # Write valid timepoints with new indices (ResolutionLevel 0 only)
            for new_t in range(n_valid):
                for c in range(n_ch):
                    key = (new_t, c)
                    if key in all_data:
                        write_vol(h5o, new_t, c, all_data[key])
                lbl = 'placed' if valid_tps[new_t]==ref_tp_idx else 'corrected'
                log.log(f"    {days[valid_tps[new_t]]}: {lbl} -> T{new_t}")

        else:
            # No skipped timepoints — simpler path
            for t in range(n_tp):
                R, tr, _ = transforms[t]
                for c in range(n_ch):
                    vol = read_vol(h5o, t, c)
                    if vol is None: continue
                    if t == ref_tp_idx:
                        expanded = place_t0(vol, new_shape, base_off)
                    else:
                        expanded = apply_correction(vol, R, tr, orig_shape, new_shape, base_off)
                    write_vol(h5o, t, c, expanded)
                log.log(f"    {days[t]}: {'placed' if t==ref_tp_idx else 'corrected'}")

        update_dims(h5o, new_shape, orig_shape, base_off)
        log.log(f"  Rebuilding pyramid...")
        rebuild_pyramid(h5o, n_valid, n_ch, new_shape)

    log.log(f"  Saved: {output_path.name} ({n_valid}T x {new_shape[0]}Z x {new_shape[1]}Y x {new_shape[2]}X)")

    # Build result info for dashboard
    result_info = {
        "filename": ims_path.name, "status": "done",
        "n_landmarks": len(all_lm), "n_timepoints": n_valid,
        "skipped_days": [days[t] for t in sorted(skipped_tps)],
        "timepoints": []
    }
    for t in range(n_tp):
        if t in skipped_tps: continue
        R, tr, rmse = transforms[t]
        rx, ry, rz = decompose_rot(R)
        result_info["timepoints"].append({
            "day": days[t],
            "translation": float(np.linalg.norm(tr)),
            "rotation": float(np.sqrt(rx**2 + ry**2 + rz**2)),
            "rmse": float(rmse)
        })
    return result_info


def interactive_launcher():
    """Interactive launcher with file/folder picker dialogs."""
    print()
    print("=" * 60)
    print("  MSP Landmark Drift Correction Tool")
    print("  Rigid body alignment with local refinement")
    print("=" * 60)
    print()
    print("  [1] Process a single file")
    print("  [2] Batch process an entire folder")
    print("  [Q] Quit")
    print()

    choice = input("  Select mode (1/2/Q): ").strip().lower()

    if choice == 'q':
        print("  Exiting."); return

    # Get channel
    ch_str = input("  Reference channel (1 or 2, default=1): ").strip()
    ref_channel = int(ch_str) - 1 if ch_str and ch_str.isdigit() else 0

    # NOW init tkinter for file dialogs (after input() so focus isn't stolen)
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)  # Ensure dialogs appear on top

    if choice == '1':
        print("\n  Select the .ims file to process...")
        input_file = filedialog.askopenfilename(
            title="Select .ims file to align",
            filetypes=[("Imaris files", "*.ims"), ("All files", "*.*")]
        )
        if not input_file:
            print("  No file selected."); root.destroy(); return
        input_file = Path(input_file)

        print(f"\n  Select output folder...")
        output_dir = filedialog.askdirectory(title="Select output folder")
        if not output_dir:
            output_dir = input_file.parent / "landmark_aligned"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        files = [input_file]

    elif choice == '2':
        print("\n  Select the INPUT folder containing .ims files...")
        input_dir = filedialog.askdirectory(title="Select input folder with .ims files")
        if not input_dir:
            print("  No folder selected."); root.destroy(); return
        input_dir = Path(input_dir)

        print(f"\n  Select OUTPUT folder for aligned files...")
        output_dir = filedialog.askdirectory(title="Select output folder")
        if not output_dir:
            output_dir = input_dir.parent / (input_dir.name + "_aligned")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(input_dir.glob("*.ims"))
        if not files:
            print(f"  No .ims files found in {input_dir}"); root.destroy(); return

        print(f"\n  Found {len(files)} .ims files:")
        for f in files:
            print(f"    {f.name}")
        print()
    else:
        print("  Invalid choice."); return

    root.destroy()

    # Run processing
    is_batch = (choice == '2')
    run_processing(files, output_dir, ref_channel, is_batch_mode=is_batch)


def run_processing(files, output_dir, ref_channel=0, trans_only=False, max_rot=25.0,
                    is_batch_mode=True):
    """Process a list of files with landmark-based drift correction."""
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = Logger(output_dir / f"landmark_correction_log_{ts}.txt")
    log.log(f"{'='*70}")
    log.log(f"  MSP Landmark Drift Correction Tool")
    log.log(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.log(f"  Output: {output_dir}  |  Files: {len(files)}")
    log.log(f"  Channel: {ref_channel+1}  |  Mode: {'translation' if trans_only else 'rigid body'}")
    log.log(f"{'='*70}")

    ok = 0
    total = len(files)
    file_results = []
    import time
    batch_start = time.time()
    file_durations = []  # for ETA estimation

    for i, f in enumerate(files, 1):
        op = output_dir / (f.stem + "_aligned.ims")
        if op.exists():
            log.log(f"\n[{i}/{total}] SKIP (exists): {f.name}")
            file_results.append({"filename": f.name, "status": "exists"})
            continue

        # Timing info for batch
        elapsed_min = (time.time() - batch_start) / 60.0
        if file_durations:
            avg_per_file = sum(file_durations) / len(file_durations)
            remaining_files = total - i + 1
            eta_min = (avg_per_file * remaining_files) / 60.0
        else:
            eta_min = None
        batch_info = (i, total, elapsed_min, eta_min)

        log.log(f"\n{'='*70}")
        log.log(f"[{i}/{total}] {f.name}  (elapsed {elapsed_min:.1f} min"
                + (f", ETA {eta_min:.1f} min" if eta_min else "") + ")")
        log.log(f"{'='*70}")

        file_start = time.time()
        try:
            result = process_file(f, op, log, ref_channel, trans_only, max_rot,
                                   batch_info=batch_info, is_batch_mode=is_batch_mode)
            file_durations.append(time.time() - file_start)
            if isinstance(result, dict):
                ok += 1
                file_results.append(result)
                log.log(f"  ✓ Complete ({ok} done, {total - i} remaining)")
            elif result == "skip":
                log.log(f"  → Skipped")
                file_results.append({"filename": f.name, "status": "skip"})
            elif result == "finish":
                log.log(f"  → Batch finished by user")
                file_results.append({"filename": f.name, "status": "finish"})
                break
            else:
                file_results.append({"filename": f.name, "status": "error"})
        except Exception as e:
            log.log(f"  ERROR: {e}"); traceback.print_exc()
            if op.exists(): op.unlink()
            file_results.append({"filename": f.name, "status": "error"})

    log.log(f"\n{'='*70}")
    log.log(f"  Finished! {ok}/{total} files corrected.")
    log.log(f"  Total batch time: {(time.time()-batch_start)/60.0:.1f} min")
    log.log(f"  Output: {output_dir}")
    log.log(f"{'='*70}")
    log.save()
    print(f"\n  Log saved: {log.path}")

    # Generate HTML dashboard
    if file_results:
        html_path = generate_html_dashboard(output_dir, file_results)
        print(f"  Report saved: {html_path}")
        log.log(f"  Report: {html_path}")
        log.save()


def main():
    ap = argparse.ArgumentParser(description="MSP Landmark Drift Correction Tool")
    ap.add_argument("--input", "-i", default=None,
                    help="Input folder or file. If omitted, opens file picker.")
    ap.add_argument("--output", "-o", default=None,
                    help="Output folder. If omitted, opens folder picker.")
    ap.add_argument("--file", "-f", default=None,
                    help="Process only files containing this string")
    ap.add_argument("--channel", type=int, default=1,
                    help="Reference channel (1-based, default: 1)")
    ap.add_argument("--translation-only", action="store_true")
    ap.add_argument("--max-rotation", type=float, default=25.0)
    args = ap.parse_args()

    # If no input/output specified, launch interactive mode
    if args.input is None or args.output is None:
        interactive_launcher()
        return

    # Command-line mode (backwards compatible)
    ind = Path(args.input)
    outd = Path(args.output)
    outd.mkdir(parents=True, exist_ok=True)

    if ind.is_file():
        files = [ind]
    else:
        files = sorted(ind.glob("*.ims"))
        if args.file:
            files = [f for f in files if args.file.lower() in f.name.lower()]

    if not files:
        print("No .ims files found."); sys.exit(1)

    run_processing(files, outd, args.channel - 1, args.translation_only, args.max_rotation)

if __name__ == "__main__":
    main()
