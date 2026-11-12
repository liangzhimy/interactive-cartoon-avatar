#!/usr/bin/env python3
"""
Universal Single-Image Cartoon Avatar Rigging & 6-Layer Skeletal Segmentation Pipeline
(`server/tools/universal_rig.py`)

Turns ANY standing / A-Pose cartoon character illustration (men, women, teachers,
doctors, students, chibi or adult proportions, pants/shorts or skirts/coats, glasses
or bare eyes, any skin tone, transparent or solid/gradient studio background) into
a standardized `AvatarBundle` (`version: "1.2.0"`):
  - `source.png`        : Clean copy of original input image
  - `full_rgba.png`     : Universal alpha-matted full character (border + pocket matting)
  - `head.png`          : Decoupled head & hair layer with curved jaw/hair cut
  - `body.png`          : Full decoupled lower body with upward-extended neck (fallback)
  - `torso.png`         : Skeletal torso + pelvis/shorts OR intact skirt/coat layer
  - `left_arm.png`      : Stage-left arm with local-color circular shoulder capsule cap
  - `right_arm.png`     : Stage-right arm with local-color circular shoulder capsule cap
  - `left_leg.png`      : Stage-left leg with circular hip socket dome behind pelvis/skirt
  - `right_leg.png`     : Stage-right leg with circular hip socket dome behind pelvis/skirt
  - `patches/*.png`     : 7 skin-matched, glasses-safe feathered expression & viseme patches
  - `manifest.json`     : Complete 6-layer specs, rig metadata, and 22 anatomical landmarks
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
    median_filter,
)

from server.tools.build_avatar_bundle import align_variant_to_base, extract_feathered_patch

_KERNEL_3X3 = np.ones((3, 3), dtype=np.uint8)


def save_rgba_png_clean(arr: np.ndarray, path: Path) -> None:
    """
    Save RGBA PNG with fast 2px RGB edge-bleed dilation into alpha==0 pixels
    so WebGL bilinear texture sampling never produces dark fringes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    out = arr.copy()
    rgb = out[..., :3]
    alpha = out[..., 3]
    for _ in range(2):
        zero_a = alpha == 0
        if not np.any(zero_a):
            break
        dil_rgb = cv2.dilate(rgb, _KERNEL_3X3, iterations=1)
        dil_a = cv2.dilate(alpha, _KERNEL_3X3, iterations=1)
        fill = zero_a & (dil_a > 0)
        rgb[fill] = dil_rgb[fill]
        alpha = np.where(fill, 1, alpha)
    out[..., :3] = rgb
    Image.fromarray(out, mode="RGBA").save(path, format="PNG")


def keep_largest_components(rgba: np.ndarray, alpha_thresh: int = 12, min_ratio: float = 0.08) -> np.ndarray:
    """Keep the primary connected component (and any large secondary part >= min_ratio of primary)."""
    out = rgba.copy()
    mask = (out[..., 3] > alpha_thresh).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return out
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area = int(areas.max())
    keep_ids = [1 + i for i, a in enumerate(areas) if a >= max(16, int(max_area * min_ratio))]
    drop = (~np.isin(labels, keep_ids)) & (labels != 0)
    out[drop] = 0
    return out


def keep_component_near_anchor(rgba: np.ndarray, anchor_x: float, anchor_y: float, alpha_thresh: int = 10) -> np.ndarray:
    """
    Strictly keep ONLY the single connected foreground component containing (or closest to)
    `(anchor_x, anchor_y)`. Guarantees limbs never include disconnected hip/skirt islands.
    """
    out = rgba.copy()
    mask = (out[..., 3] > alpha_thresh).astype(np.uint8)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return out

    h, w = mask.shape
    ax = int(np.clip(round(anchor_x), 0, w - 1))
    ay = int(np.clip(round(anchor_y), 0, h - 1))
    chosen = int(labels[ay, ax])
    if chosen == 0:
        best_score = -1e9
        chosen = 1
        for lid in range(1, num):
            area = float(stats[lid, cv2.CC_STAT_AREA])
            if area < 25:
                continue
            cx, cy = float(centroids[lid, 0]), float(centroids[lid, 1])
            dist = np.hypot(cx - anchor_x, cy - anchor_y)
            score = area - 18.0 * dist
            if score > best_score:
                best_score = score
                chosen = lid

    out[(labels != chosen) & (labels != 0)] = 0
    return out


def crop_layer_spec(
    full_layer: np.ndarray,
    offset_x: int,
    offset_y: int,
    pad: int,
    cols: int,
    rows: int,
    tex_name: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Crop non-empty RGBA bounding box with padding and return `(cropped_rgba, LayerSpec)`."""
    h, w, _ = full_layer.shape
    ys, xs = np.where(full_layer[..., 3] > 3)
    if len(xs) == 0:
        cx, cy = w // 2, h // 2
        x0, y0, x1, y1 = max(0, cx - 8), max(0, cy - 8), min(w, cx + 8), min(h, cy + 8)
    else:
        x0 = max(0, int(xs.min()) - pad)
        y0 = max(0, int(ys.min()) - pad)
        x1 = min(w, int(xs.max()) + pad + 1)
        y1 = min(h, int(ys.max()) + pad + 1)
    cropped = full_layer[y0:y1, x0:x1].copy()
    spec = {
        "texture": tex_name,
        "x": int(offset_x + x0),
        "y": int(offset_y + y0),
        "width": int(x1 - x0),
        "height": int(y1 - y0),
        "meshCols": cols,
        "meshRows": rows,
    }
    return cropped, spec


# ==============================================================================
# 1. Universal Background Removal (Transparent / White / Colored / Gradient + Pockets)
# ==============================================================================

def remove_universal_background(pil_img: Image.Image) -> np.ndarray:
    """
    Convert any input PIL Image into clean matted RGBA (`HxWx4` uint8):
      1. Preserves existing alpha transparency if border is already transparent.
      2. Models solid or smooth gradient studio backgrounds from the 4 image borders in Lab+RGB.
      3. Flood-fills exterior background (without Sobel choking narrow leg/shoe gaps!) AND
         removes enclosed background pockets under arms / between legs / beside neck while
         protecting white/cream shirts, lab coats, and eye sclera.
      4. Un-premultiplies background color from anti-aliased silhouette edges.
    """
    rgba_in = np.array(pil_img.convert("RGBA"))
    h, w, _ = rgba_in.shape
    rgb = rgba_in[..., :3].astype(np.float32)
    alpha_in = rgba_in[..., 3].astype(np.float32)

    margin = max(3, min(h, w) // 90)
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:margin, :] = True
    border_mask[-margin:, :] = True
    border_mask[:, :margin] = True
    border_mask[:, -margin:] = True

    if np.mean(alpha_in[border_mask] < 24) > 0.35:
        out = rgba_in.copy()
        out[out[..., 3] < 6] = 0
        return out

    lab = cv2.cvtColor(rgba_in[..., :3], cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(rgba_in[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32)

    patch_sz = max(6, min(h, w) // 24)
    corner_coords = [
        (0, patch_sz, 0, patch_sz),
        (0, patch_sz, w - patch_sz, w),
        (h - patch_sz, h, 0, patch_sz),
        (h - patch_sz, h, w - patch_sz, w),
        (0, patch_sz, w // 2 - patch_sz // 2, w // 2 + patch_sz // 2),
        (h // 2 - patch_sz // 2, h // 2 + patch_sz // 2, 0, patch_sz),
        (h // 2 - patch_sz // 2, h // 2 + patch_sz // 2, w - patch_sz, w),
    ]
    sample_labs = [np.median(lab[y0:y1, x0:x1].reshape(-1, 3), axis=0) for (y0, y1, x0, x1) in corner_coords]
    global_bg_lab = np.median(np.stack(sample_labs, axis=0), axis=0)

    border_dist_global = np.linalg.norm(lab - global_bg_lab[None, None, :], axis=2)
    valid_bg_border = border_mask & (border_dist_global < 28.0)
    if np.sum(valid_bg_border) < 40:
        valid_bg_border = border_mask

    small_h, small_w = 32, 32
    weight_full = valid_bg_border.astype(np.float32)
    w_small = cv2.resize(weight_full, (small_w, small_h), interpolation=cv2.INTER_AREA)
    lab_small = cv2.resize(lab * weight_full[..., None], (small_w, small_h), interpolation=cv2.INTER_AREA)
    rgb_small = cv2.resize(rgb * weight_full[..., None], (small_w, small_h), interpolation=cv2.INTER_AREA)

    for _ in range(48):
        lab_blur = cv2.GaussianBlur(lab_small, (5, 5), 1.2)
        rgb_blur = cv2.GaussianBlur(rgb_small, (5, 5), 1.2)
        w_blur = cv2.GaussianBlur(w_small, (5, 5), 1.2)
        missing = w_small < 1e-4
        lab_small[missing] = lab_blur[missing]
        rgb_small[missing] = rgb_blur[missing]
        w_small[missing] = w_blur[missing]

    denom = np.maximum(w_small[..., None], 1e-5)
    bg_lab_field = cv2.resize(lab_small / denom, (w, h), interpolation=cv2.INTER_CUBIC)
    bg_rgb_field = np.clip(cv2.resize(rgb_small / denom, (w, h), interpolation=cv2.INTER_CUBIC), 0.0, 255.0)

    lab_dist = np.linalg.norm(lab - bg_lab_field, axis=2)
    rgb_dist = np.max(np.abs(rgb - bg_rgb_field), axis=2)

    border_noise = float(np.percentile(lab_dist[valid_bg_border], 92)) if np.any(valid_bg_border) else 6.0
    ext_lab_thresh = float(np.clip(max(15.0, border_noise * 2.5), 15.0, 25.0))
    ext_rgb_thresh = float(np.clip(ext_lab_thresh * 1.7, 24.0, 42.0))

    # 4-connectivity flood-fill (Jordan curve duality on Z^2): floods open channels between shoes/ankles
    # cleanly while NEVER jumping diagonally across thin 1px dark outlines into light shirts or sclera!
    bg_candidate = ((lab_dist < ext_lab_thresh) & (rgb_dist < ext_rgb_thresh)).astype(np.uint8)
    _, labels = cv2.connectedComponents(bg_candidate, connectivity=4)
    border_labels = (
        set(labels[0, :])
        | set(labels[h - 1, :])
        | set(labels[:, 0])
        | set(labels[:, w - 1])
    )
    border_labels.discard(0)
    exterior_bg = np.isin(labels, list(border_labels))

    # Remove enclosed background pockets (between legs, under arms, or tiny hair-neck triangles)
    fg_prelim = ~exterior_bg
    ys_fg, xs_fg = np.where(fg_prelim)
    if len(xs_fg) > 100:
        cx0, cx1 = int(xs_fg.min()), int(xs_fg.max())
        cy0, cy1 = int(ys_fg.min()), int(ys_fg.max())
        cw = max(1, cx1 - cx0)
        ch = max(1, cy1 - cy0)
        char_cx = 0.5 * (cx0 + cx1)
        char_area = float(np.sum(fg_prelim))

        pocket_cand = (
            (lab_dist < ext_lab_thresh * 0.94)
            & (rgb_dist < ext_rgb_thresh * 0.94)
            & (~exterior_bg)
        ).astype(np.uint8)
        p_num, p_labels, p_stats, p_centroids = cv2.connectedComponentsWithStats(pocket_cand, connectivity=8)
        pocket_ids: List[int] = []
        for pid in range(1, p_num):
            area = float(p_stats[pid, cv2.CC_STAT_AREA])
            if area < 10 or area > char_area * 0.09:
                continue
            px, py = float(p_centroids[pid, 0]), float(p_centroids[pid, 1])
            rel_y = (py - cy0) / float(ch)
            rel_dx = abs(px - char_cx) / float(cw)
            is_arm_pocket = (0.20 <= rel_y <= 0.78) and (rel_dx >= 0.085)
            is_leg_pocket = (0.44 <= rel_y <= 0.99) and (rel_dx <= 0.16)
            if is_arm_pocket or is_leg_pocket:
                comp_mask = p_labels == pid
                if float(np.std(gray[comp_mask])) < 14.0:
                    pocket_ids.append(pid)
        if pocket_ids:
            exterior_bg = exterior_bg | np.isin(p_labels, pocket_ids)

    fg_clean = (~exterior_bg).astype(np.uint8)
    c_num, c_labels, c_stats, _ = cv2.connectedComponentsWithStats(fg_clean, connectivity=8)
    if c_num > 1:
        max_fg = int(c_stats[1:, cv2.CC_STAT_AREA].max())
        valid_fg = [1 + i for i, a in enumerate(c_stats[1:, cv2.CC_STAT_AREA]) if a >= max(36, int(max_fg * 0.015))]
        exterior_bg = ~np.isin(c_labels, valid_fg)

    dilated_bg = binary_dilation(exterior_bg, iterations=2)
    eroded_fg = binary_erosion(~exterior_bg, iterations=1)
    edge_band = dilated_bg & (~exterior_bg) & (~eroded_fg)

    alpha = (~exterior_bg).astype(np.float32)
    edge_cov = np.clip((rgb_dist - 4.0) / 145.0, 0.0, 1.0)
    alpha[edge_band] = np.minimum(alpha[edge_band], edge_cov[edge_band])

    clean_rgb = rgb.copy()
    semi = (alpha > 0.03) & (alpha < 0.98)
    for c in range(3):
        clean_rgb[semi, c] = np.clip(
            (clean_rgb[semi, c] - (1.0 - alpha[semi]) * bg_rgb_field[semi, c]) / np.maximum(alpha[semi], 0.10),
            0.0,
            255.0,
        )

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = np.round(clean_rgb).astype(np.uint8)
    rgba[..., 3] = np.round(alpha * 255.0).astype(np.uint8)
    rgba[rgba[..., 3] < 5] = 0
    return rgba


# ==============================================================================
# 2. Adaptive Head / Neck / Hair Split + Clean Skin-Matched Throat Extrusion
# ==============================================================================

def find_adaptive_head_neck_split(
    rgba: np.ndarray,
    override_neck_pivot: Optional[Tuple[float, float]] = None,
) -> Tuple[int, int, int, np.ndarray]:
    """
    Locate optimal neck waist `(neck_y, neck_xl, neck_xr)` across both chibi and adult
    proportions, trace the quadratic chin U-curve inside the neck columns, and curve the
    side cut gently above the shoulders so bob/medium hair stays intact on `head.png`.
    """
    h, w, _ = rgba.shape
    fg = rgba[..., 3] > 25
    ys_fg, xs_fg = np.where(fg)
    char_top, char_bot = int(ys_fg.min()), int(ys_fg.max())
    char_left, char_right = int(xs_fg.min()), int(xs_fg.max())
    char_h = max(1, char_bot - char_top)
    char_w = max(1, char_right - char_left)
    char_cx = 0.5 * (char_left + char_right)

    rgb = rgba[..., :3].astype(np.float32)
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    if override_neck_pivot is not None:
        best_y = int(np.clip(round(override_neck_pivot[1]) + 4, char_top + 20, char_bot - 40))
    else:
        search_y0 = char_top + int(char_h * 0.11)
        search_y1 = char_top + int(char_h * 0.36)

        widths = np.full(h, float(w), dtype=np.float32)
        spans: Dict[int, Tuple[int, int]] = {}
        for y in range(search_y0, search_y1):
            row = fg[y]
            num_r, r_lbl = cv2.connectedComponents(row[None, :].astype(np.uint8), connectivity=8)
            best_seg = None
            best_dist = 1e9
            for lid in range(1, num_r):
                seg_xs = np.where(r_lbl[0] == lid)[0]
                if len(seg_xs) < 10:
                    continue
                seg_cx = 0.5 * (seg_xs[0] + seg_xs[-1])
                d = abs(seg_cx - char_cx)
                if d < best_dist:
                    best_dist = d
                    best_seg = (int(seg_xs[0]), int(seg_xs[-1]))
            if best_seg is not None and best_dist < char_w * 0.22:
                widths[y] = float(best_seg[1] - best_seg[0])
                spans[y] = best_seg

        smooth_w = gaussian_filter(widths, sigma=2.0)
        best_score = 1e9
        best_y = char_top + int(char_h * 0.22)
        step_dy = max(8, int(char_h * 0.035))
        for y in range(search_y0 + 4, search_y1 - step_dy):
            if y not in spans:
                continue
            shoulder_flare = smooth_w[min(search_y1 - 1, y + step_dy)] - smooth_w[y]
            score = smooth_w[y] - 0.55 * max(0.0, shoulder_flare)
            if score < best_score:
                best_score = score
                best_y = y

    xs_row = np.where(fg[best_y])[0]
    if len(xs_row) >= 12:
        raw_xl, raw_xr = int(xs_row[0]), int(xs_row[-1])
    else:
        raw_xl, raw_xr = int(char_cx - char_w * 0.08), int(char_cx + char_w * 0.08)

    max_neck_half = int(char_w * 0.14)
    if (raw_xr - raw_xl) > max_neck_half * 2:
        c_int = int(round(char_cx))
        l_search = luma[best_y, max(raw_xl, c_int - max_neck_half) : max(raw_xl + 4, c_int - int(char_w * 0.035))]
        r_search = luma[best_y, min(raw_xr - 4, c_int + int(char_w * 0.035)) : min(raw_xr, c_int + max_neck_half)]
        if len(l_search) > 3 and len(r_search) > 3:
            neck_xl = max(raw_xl, c_int - max_neck_half) + int(np.argmin(l_search))
            neck_xr = min(raw_xr - 4, c_int + int(char_w * 0.035)) + int(np.argmin(r_search))
        else:
            neck_xl = c_int - int(char_w * 0.08)
            neck_xr = c_int + int(char_w * 0.08)
    else:
        neck_xl, neck_xr = raw_xl, raw_xr

    chin_y_per_x = np.full(w, best_y - 2, dtype=np.int32)
    scan_top = max(char_top + 12, best_y - int(char_h * 0.055))
    scan_bot = best_y + 3
    margin_x0 = max(0, neck_xl - 4)
    margin_x1 = min(w - 1, neck_xr + 4)

    found_xs: List[int] = []
    found_ys: List[int] = []
    for x in range(margin_x0, margin_x1 + 1):
        col = luma[scan_top:scan_bot, x]
        dark_idx = np.where(col < 125)[0]
        if len(dark_idx) > 0:
            cy = scan_top + int(dark_idx[-1])
            found_xs.append(x)
            found_ys.append(cy)

    if len(found_xs) >= 5:
        poly = np.poly1d(np.polyfit(found_xs, found_ys, deg=2))
        for x in range(margin_x0, margin_x1 + 1):
            chin_y_per_x[x] = int(np.clip(np.round(poly(x)), scan_top, best_y + 2))

    cutoff_1d = np.full(w, best_y, dtype=np.int32)
    cutoff_1d[margin_x0 : margin_x1 + 1] = chin_y_per_x[margin_x0 : margin_x1 + 1]

    # Curve side cutoff gently ONLY where hair actually hangs down past `best_y - 5`
    # (keeps bob/medium hair tips on `head.png` while never grazing shirt collars on short hair!)
    hair_probe_y = max(char_top + 6, best_y - 5)
    side_drop_max = int(char_h * 0.024)
    for x in range(0, margin_x0):
        if fg[hair_probe_y, x]:
            dist_out = (margin_x0 - x) / max(1.0, float(char_w * 0.18))
            cutoff_1d[x] = best_y + int(round(min(1.0, dist_out) * side_drop_max))
        else:
            cutoff_1d[x] = best_y - 4
    for x in range(margin_x1 + 1, w):
        if fg[hair_probe_y, x]:
            dist_out = (x - margin_x1) / max(1.0, float(char_w * 0.18))
            cutoff_1d[x] = best_y + int(round(min(1.0, dist_out) * side_drop_max))
        else:
            cutoff_1d[x] = best_y - 4

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    head_zone = yy <= cutoff_1d[xx]
    return best_y, neck_xl, neck_xr, head_zone


def split_head_and_body_universal(
    rgba: np.ndarray, neck_y: int, neck_xl: int, neck_xr: int, head_zone: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decouple `head_rgba` and `body_rgba` cleanly and synthesize a narrow, skin-matched
    throat cylinder behind the chin (avoiding extruding shirt collars into vertical pillars!).
    """
    h, w, _ = rgba.shape

    head_rgba = rgba.copy()
    head_float = head_zone.astype(np.float32)
    head_soft = cv2.GaussianBlur(head_float, (3, 3), 0.6)
    head_rgba[..., 3] = np.round(
        rgba[..., 3].astype(np.float32) * np.clip(head_soft * 1.15, 0.0, 1.0)
    ).astype(np.uint8)
    head_rgba[head_soft < 0.05] = 0
    head_rgba = keep_largest_components(head_rgba, alpha_thresh=10, min_ratio=0.04)

    body_rgba = rgba.copy()
    body_rgba[head_zone] = 0

    # Sample throat/lower-face skin color near center above/around `neck_y` so collared shirts
    # never extrude blue/white collar tips upward behind the ears!
    neck_cx = 0.5 * (neck_xl + neck_xr)
    neck_span = max(14, neck_xr - neck_xl)
    s_x0 = max(0, int(round(neck_cx - neck_span * 0.22)))
    s_x1 = min(w, int(round(neck_cx + neck_span * 0.22)))
    s_y0 = max(0, neck_y - int(neck_span * 0.55))
    s_y1 = min(h, neck_y + int(neck_span * 0.20))

    throat_crop = rgba[s_y0:s_y1, s_x0:s_x1]
    t_rgb = throat_crop[..., :3].astype(np.float32)
    t_luma = 0.299 * t_rgb[..., 0] + 0.587 * t_rgb[..., 1] + 0.114 * t_rgb[..., 2]
    valid_skin = (throat_crop[..., 3] > 200) & (t_luma > 105.0)
    if np.sum(valid_skin) >= 12:
        throat_rgb = np.median(t_rgb[valid_skin], axis=0)
    else:
        throat_rgb = np.array([238.0, 194.0, 162.0], dtype=np.float32)

    dark_px = t_rgb[(throat_crop[..., 3] > 200) & (t_luma < 95.0)]
    outline_rgb = np.median(dark_px, axis=0) if len(dark_px) >= 6 else np.array([52.0, 32.0, 26.0], dtype=np.float32)

    # Narrow throat cylinder (`~0.68 * neck_span`, safely inside jawline!)
    cyl_half_w = max(8.0, neck_span * 0.34)
    ext_height = max(28, int(neck_span * 0.68))
    start_fill_y = min(h - 1, neck_y + 3)

    for step in range(0, ext_height + 4):
        y = start_fill_y - step
        if y < 0:
            break
        dy_above = max(0, neck_y - y)
        t = dy_above / float(ext_height)
        cur_half = cyl_half_w * (1.0 - 0.18 * t)
        if cur_half <= 4.0:
            break

        c_xl = max(0, int(round(neck_cx - cur_half)))
        c_xr = min(w - 1, int(round(neck_cx + cur_half)))
        xs_line = np.arange(c_xl, c_xr + 1, dtype=np.float32)
        edge_d = np.minimum(xs_line - c_xl, c_xr - xs_line)

        fade = 1.0 if t < 0.72 else max(0.0, (1.0 - t) / 0.28)
        row_a = np.round(np.clip((edge_d + 0.8) / 1.8, 0.0, 1.0) * fade * 255.0).astype(np.uint8)

        # Subtle cast chin shadow + crisp 1.5px side outline so neck looks naturally drawn
        shade = 0.93 - 0.05 * min(1.0, dy_above / 16.0)
        base_col = np.clip(throat_rgb * shade, 0.0, 255.0)
        is_rim = (edge_d < 1.8) & (t < 0.65)
        row_rgb = np.where(is_rim[:, None], outline_rgb[None, :], base_col[None, :]).astype(np.uint8)

        # Only paint where body_rgba is currently empty or right along the cut line (`y <= neck_y + 2`)
        if y <= neck_y + 1:
            body_rgba[y, max(0, neck_xl - 2) : min(w, neck_xr + 3)] = 0
        existing_a = body_rgba[y, c_xl : c_xr + 1, 3]
        upd = row_a > existing_a
        body_rgba[y, c_xl : c_xr + 1, :3][upd] = row_rgb[upd]
        body_rgba[y, c_xl : c_xr + 1, 3][upd] = row_a[upd]

    body_rgba = keep_largest_components(body_rgba, alpha_thresh=10, min_ratio=0.02)
    return head_rgba, body_rgba


# ==============================================================================
# 3. Universal Skin-Tone, Glasses-Safe Face Mask & Eye/Mouth Detector + Patch Synth
# ==============================================================================

def detect_face_and_features(
    head_crop_rgba: np.ndarray,
    head_bbox: Tuple[int, int, int, int],
    landmark_overrides: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Any]:
    hh, hw, _ = head_crop_rgba.shape
    hx0, hy0, _, _ = head_bbox
    rgb = head_crop_rgba[..., :3].astype(np.float32)
    alpha = head_crop_rgba[..., 3] > 200
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    lab = cv2.cvtColor(head_crop_rgba[..., :3], cv2.COLOR_RGB2LAB).astype(np.float32)

    cy0, cy1 = int(hh * 0.48), int(hh * 0.88)
    cx0, cx1 = int(hw * 0.33), int(hw * 0.67)
    core_alpha = alpha[cy0:cy1, cx0:cx1]
    core_luma = luma[cy0:cy1, cx0:cx1]
    core_lab = lab[cy0:cy1, cx0:cx1]
    core_rgb = rgb[cy0:cy1, cx0:cx1]

    valid_core = core_alpha
    if np.sum(valid_core) > 40:
        p30 = float(np.percentile(core_luma[valid_core], 32))
        p96 = float(np.percentile(core_luma[valid_core], 96))
        skin_seed = valid_core & (core_luma >= p30) & (core_luma <= max(p30 + 10.0, p96))
        if np.sum(skin_seed) < 20:
            skin_seed = valid_core
        skin_lab = np.median(core_lab[skin_seed], axis=0)
        skin_rgb = np.median(core_rgb[skin_seed], axis=0)
    else:
        skin_lab = np.array([210.0, 142.0, 148.0], dtype=np.float32)
        skin_rgb = np.array([246.0, 198.0, 164.0], dtype=np.float32)

    d_lab = np.linalg.norm(lab - skin_lab[None, None, :], axis=2)
    skin_like = alpha & (d_lab < 29.0)

    close_k = max(5, int(round(min(hh, hw) * 0.032)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    bridged = cv2.morphologyEx(skin_like.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0

    num_l, lbls, stats, _ = cv2.connectedComponentsWithStats(bridged.astype(np.uint8), connectivity=8)
    face_comp = np.zeros((hh, hw), dtype=bool)
    min_comp_area = max(30, int(hh * hw * 0.008))
    for lid in range(1, num_l):
        if stats[lid, cv2.CC_STAT_AREA] < min_comp_area:
            continue
        bx = stats[lid, cv2.CC_STAT_LEFT]
        by = stats[lid, cv2.CC_STAT_TOP]
        bw = stats[lid, cv2.CC_STAT_WIDTH]
        bh = stats[lid, cv2.CC_STAT_HEIGHT]
        if (bx < hw * 0.68) and (bx + bw > hw * 0.32) and (by < hh * 0.88) and (by + bh > hh * 0.42):
            face_comp |= lbls == lid

    if not np.any(face_comp):
        face_comp = alpha

    inv = (~face_comp).astype(np.uint8)
    _, inv_lbls = cv2.connectedComponents(inv, connectivity=8)
    border_inv = (
        set(inv_lbls[0, :])
        | set(inv_lbls[hh - 1, :])
        | set(inv_lbls[:, 0])
        | set(inv_lbls[:, hw - 1])
    )
    filled_face = (~np.isin(inv_lbls, list(border_inv))) & alpha

    eroded = binary_erosion(filled_face, iterations=1)
    dist_in = distance_transform_edt(eroded)
    face_interior_mask = np.clip(dist_in / 4.5, 0.0, 1.0).astype(np.float32)

    fys, fxs = np.where(filled_face)
    if len(fxs) > 40:
        fx0, fx1 = int(fxs.min()), int(fxs.max())
        fy0, fy1 = int(fys.min()), int(fys.max())
    else:
        fx0, fx1 = int(hw * 0.22), int(hw * 0.78)
        fy0, fy1 = int(hh * 0.36), int(hh * 0.95)
    fw = max(24, fx1 - fx0)
    fh = max(24, fy1 - fy0)
    face_cx = 0.5 * (fx0 + fx1)

    skin_luma = float(0.299 * skin_rgb[0] + 0.587 * skin_rgb[1] + 0.114 * skin_rgb[2])
    dark_feat = (face_interior_mask > 0.28) & ((luma < skin_luma * 0.68) | (d_lab > 26.0))

    lex = face_cx - fw * 0.205
    rex = face_cx + fw * 0.205
    ley = fy0 + fh * 0.44
    rey = ley
    ew = max(10, int(round(fw * 0.115)))
    eh = max(8, int(round(fh * 0.085)))

    eye_band = np.zeros((hh, hw), dtype=bool)
    ey_min = max(0, int(fy0 + fh * 0.18))
    ey_max = min(hh, int(fy0 + fh * 0.66))
    eye_band[ey_min:ey_max, max(0, int(fx0 + fw * 0.06)) : min(hw, int(fx1 - fw * 0.06))] = True

    e_mask = (dark_feat & eye_band).astype(np.uint8)
    en, _, estats, ecent = cv2.connectedComponentsWithStats(e_mask, connectivity=8)
    left_cands = []
    right_cands = []
    for eid in range(1, en):
        area = int(estats[eid, cv2.CC_STAT_AREA])
        bw = int(estats[eid, cv2.CC_STAT_WIDTH])
        bh = int(estats[eid, cv2.CC_STAT_HEIGHT])
        cx_e, cy_e = float(ecent[eid, 0]), float(ecent[eid, 1])
        if area < 12 or bw > fw * 0.42 or bh > fh * 0.36:
            continue
        if bw > 3.4 * max(1, bh) and cy_e < fy0 + fh * 0.36:
            continue
        item = (cx_e, cy_e, bw, bh, area)
        if face_cx - fw * 0.38 <= cx_e <= face_cx - fw * 0.06:
            left_cands.append(item)
        elif face_cx + fw * 0.06 <= cx_e <= face_cx + fw * 0.38:
            right_cands.append(item)

    best_pair = None
    best_pair_score = -1e9
    for lc in left_cands:
        for rc in right_cands:
            dy = abs(lc[1] - rc[1])
            if dy > fh * 0.10:
                continue
            sym_err = abs((face_cx - lc[0]) - (rc[0] - face_cx))
            if sym_err > fw * 0.14:
                continue
            score = (lc[4] + rc[4]) - 8.0 * dy - 5.0 * sym_err
            if score > best_pair_score:
                best_pair_score = score
                best_pair = (lc, rc)

    if best_pair is not None:
        lc, rc = best_pair
        lex, ley = lc[0], lc[1]
        rex, rey = rc[0], rc[1]
        ew = int(np.clip(round(0.52 * max(lc[2], rc[2])), int(fw * 0.075), int(fw * 0.145)))
        eh = int(np.clip(round(0.55 * max(lc[3], rc[3])), int(fh * 0.055), int(fh * 0.125)))
    else:
        row_dark = np.sum(
            (luma[ey_min:ey_max, int(face_cx - fw * 0.32) : int(face_cx + fw * 0.32)] < skin_luma * 0.62),
            axis=1,
        )
        if len(row_dark) > 4 and np.max(row_dark) > 4:
            ley = rey = float(ey_min + int(np.argmax(gaussian_filter(row_dark.astype(np.float32), sigma=2.0))))

    my_min = min(hh - 6, max(int(max(ley, rey) + eh + fh * 0.12), int(fy0 + fh * 0.62)))
    my_max = min(hh - 3, int(fy0 + fh * 0.92))
    mx_min = max(0, int(face_cx - fw * 0.20))
    mx_max = min(hw, int(face_cx + fw * 0.20))

    mx = face_cx
    my = fy0 + fh * 0.78
    mw = int(np.clip(round(fw * 0.11), 10, int(hw * 0.13)))

    if my_max > my_min + 4 and mx_max > mx_min + 6:
        m_patch = (
            (face_interior_mask[my_min:my_max, mx_min:mx_max] > 0.35)
            & (
                (luma[my_min:my_max, mx_min:mx_max] < skin_luma * 0.80)
                | (d_lab[my_min:my_max, mx_min:mx_max] > 16.0)
            )
        )
        mys, mxs = np.where(m_patch)
        if len(mxs) >= 8:
            mx = float(mx_min + np.median(mxs))
            my = float(my_min + np.median(mys))
            span_w = int(np.percentile(mxs, 92) - np.percentile(mxs, 8))
            mw = int(np.clip(round(span_w * 0.58), int(fw * 0.075), int(fw * 0.15)))

    dark_pixels = rgb[alpha & (luma < min(105.0, skin_luma * 0.52))]
    line_color = np.median(dark_pixels, axis=0).astype(np.uint8).tolist() if len(dark_pixels) > 25 else [54, 28, 22]

    ring_r0 = max(ew + 2, int(ew * 1.15))
    ring_r1 = max(ring_r0 + 3, int(ew * 1.65))
    yy_h, xx_h = np.meshgrid(np.arange(hh, dtype=np.float32), np.arange(hw, dtype=np.float32), indexing="ij")
    l_ring = (np.hypot(xx_h - lex, yy_h - ley) >= ring_r0) & (np.hypot(xx_h - lex, yy_h - ley) <= ring_r1)
    has_glasses = bool(np.mean(luma[l_ring & alpha] < skin_luma * 0.62) > 0.22) if np.any(l_ring & alpha) else False

    if landmark_overrides:
        if "leftEye" in landmark_overrides and len(landmark_overrides["leftEye"]) == 2:
            lex = float(landmark_overrides["leftEye"][0] - hx0)
            ley = float(landmark_overrides["leftEye"][1] - hy0)
        if "rightEye" in landmark_overrides and len(landmark_overrides["rightEye"]) == 2:
            rex = float(landmark_overrides["rightEye"][0] - hx0)
            rey = float(landmark_overrides["rightEye"][1] - hy0)
        if "mouth" in landmark_overrides and len(landmark_overrides["mouth"]) == 2:
            mx = float(landmark_overrides["mouth"][0] - hx0)
            my = float(landmark_overrides["mouth"][1] - hy0)

    return {
        "face_interior_mask": face_interior_mask,
        "skin_color": np.round(skin_rgb).astype(np.uint8).tolist(),
        "line_color": line_color,
        "left_eye": (lex, ley),
        "right_eye": (rex, rey),
        "eye_radius": (ew, eh),
        "mouth": (mx, my),
        "mouth_half_w": mw,
        "face_box": (fx0, fy0, fx1, fy1),
        "has_glasses": has_glasses,
    }


def synthesize_universal_patches(
    head_crop_rgba: np.ndarray,
    feat: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    hh, hw, _ = head_crop_rgba.shape
    face_interior_mask = feat["face_interior_mask"]
    skin_color: List[int] = feat["skin_color"]
    line_color: List[int] = feat["line_color"]
    lex, ley = int(round(feat["left_eye"][0])), int(round(feat["left_eye"][1]))
    rex, rey = int(round(feat["right_eye"][0])), int(round(feat["right_eye"][1]))
    ew, eh = int(feat["eye_radius"][0]), int(feat["eye_radius"][1])
    mx, my = int(round(feat["mouth"][0])), int(round(feat["mouth"][1]))
    mw = int(feat["mouth_half_w"])
    has_glasses: bool = bool(feat.get("has_glasses", False))

    rgb = head_crop_rgba[..., :3].astype(np.float32)
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    skin_luma = float(0.299 * skin_color[0] + 0.587 * skin_color[1] + 0.114 * skin_color[2])

    pad_e = 1 if has_glasses else 4
    line_th = max(2, int(round(min(hw, hh) / 115.0)))

    def paint_eye_pair(mode: str) -> np.ndarray:
        canvas = head_crop_rgba.copy()
        for cx, cy in ((lex, ley), (rex, rey)):
            cv2.ellipse(
                canvas,
                (cx, cy),
                (ew + pad_e, eh + pad_e),
                0,
                0,
                360,
                skin_color + [255],
                -1,
                cv2.LINE_AA,
            )
            if mode == "closed":
                cv2.ellipse(
                    canvas,
                    (cx, cy - max(1, eh // 5)),
                    (ew, max(3, int(eh * 0.45))),
                    0,
                    15,
                    165,
                    line_color + [255],
                    line_th,
                    cv2.LINE_AA,
                )
            else:
                cv2.ellipse(
                    canvas,
                    (cx, cy + max(2, eh // 4)),
                    (ew, max(4, int(eh * 0.58))),
                    0,
                    195,
                    345,
                    line_color + [255],
                    line_th,
                    cv2.LINE_AA,
                )
        if has_glasses:
            yy_g, xx_g = np.meshgrid(np.arange(hh), np.arange(hw), indexing="ij")
            rim_l = np.hypot((xx_g - lex) / max(1.0, ew + 1.5), (yy_g - ley) / max(1.0, eh + 1.5)) > 0.88
            rim_r = np.hypot((xx_g - rex) / max(1.0, ew + 1.5), (yy_g - rey) / max(1.0, eh + 1.5)) > 0.88
            frame_px = rim_l & rim_r & (luma < skin_luma * 0.55)
            canvas[frame_px] = head_crop_rgba[frame_px]
        return canvas

    c_blink = paint_eye_pair("closed")
    c_happy = paint_eye_pair("happy")

    mh = max(6, int(round(mw * 0.62)))
    cover_rx = mw + 7
    cover_ry = mh + 6

    c_smile = head_crop_rgba.copy()
    cv2.ellipse(c_smile, (mx, my), (cover_rx, cover_ry), 0, 0, 360, skin_color + [255], -1, cv2.LINE_AA)
    cv2.ellipse(c_smile, (mx, my - max(2, mh // 3)), (mw + 2, max(4, int(mh * 0.72))), 0, 18, 162, line_color + [255], line_th, cv2.LINE_AA)

    c_laugh = head_crop_rgba.copy()
    cv2.ellipse(c_laugh, (mx, my + 1), (cover_rx + 1, cover_ry + 2), 0, 0, 360, skin_color + [255], -1, cv2.LINE_AA)
    cv2.ellipse(c_laugh, (mx, my - 2), (mw + 3, mh + 4), 0, 0, 180, [156, 54, 58, 255], -1, cv2.LINE_AA)
    cv2.ellipse(c_laugh, (mx, my + max(3, mh // 2)), (int(mw * 0.72), max(4, mh // 2)), 0, 0, 180, [238, 128, 125, 255], -1, cv2.LINE_AA)
    cv2.rectangle(c_laugh, (mx - mw, my - 2), (mx + mw, my + max(1, mh // 4)), [252, 250, 248, 255], -1)
    cv2.ellipse(c_laugh, (mx, my - 2), (mw + 3, mh + 4), 0, 0, 180, line_color + [255], line_th, cv2.LINE_AA)
    cv2.line(c_laugh, (mx - mw - 2, my - 2), (mx + mw + 2, my - 2), line_color + [255], line_th, cv2.LINE_AA)

    c_a = head_crop_rgba.copy()
    cv2.ellipse(c_a, (mx, my + 1), (cover_rx, cover_ry), 0, 0, 360, skin_color + [255], -1, cv2.LINE_AA)
    cv2.ellipse(c_a, (mx, my + 2), (max(5, int(mw * 0.82)), max(5, int(mh * 0.85))), 0, 0, 360, [160, 58, 60, 255], -1, cv2.LINE_AA)
    cv2.ellipse(c_a, (mx, my + max(4, mh // 2)), (max(4, int(mw * 0.60)), max(3, int(mh * 0.42))), 0, 0, 180, [235, 130, 126, 255], -1, cv2.LINE_AA)
    cv2.ellipse(c_a, (mx, my + 2), (max(5, int(mw * 0.82)), max(5, int(mh * 0.85))), 0, 0, 360, line_color + [255], line_th, cv2.LINE_AA)

    c_o = head_crop_rgba.copy()
    cv2.ellipse(c_o, (mx, my + 1), (cover_rx, cover_ry), 0, 0, 360, skin_color + [255], -1, cv2.LINE_AA)
    cv2.ellipse(c_o, (mx, my + 2), (max(4, int(mw * 0.48)), max(5, int(mh * 0.78))), 0, 0, 360, [150, 52, 55, 255], -1, cv2.LINE_AA)
    cv2.ellipse(c_o, (mx, my + 2), (max(4, int(mw * 0.48)), max(5, int(mh * 0.78))), 0, 0, 360, line_color + [255], line_th, cv2.LINE_AA)

    c_e = head_crop_rgba.copy()
    cv2.ellipse(c_e, (mx, my + 1), (cover_rx, cover_ry), 0, 0, 360, skin_color + [255], -1, cv2.LINE_AA)
    cv2.ellipse(c_e, (mx, my + 1), (max(5, int(mw * 0.88)), max(4, int(mh * 0.56))), 0, 0, 180, [215, 108, 105, 255], -1, cv2.LINE_AA)
    cv2.ellipse(c_e, (mx, my + 1), (max(5, int(mw * 0.88)), max(4, int(mh * 0.56))), 0, 0, 180, line_color + [255], line_th, cv2.LINE_AA)
    cv2.line(c_e, (mx - max(5, int(mw * 0.88)), my + 1), (mx + max(5, int(mw * 0.88)), my + 1), line_color + [255], line_th, cv2.LINE_AA)

    dummy_bbox = (0, 0, hw, hh)
    ey_mid = 0.5 * (ley + rey)
    eye_zone = (
        max(0.08, (lex - ew * 1.65) / hw),
        max(0.12, (ey_mid - eh * 1.85) / hh),
        min(0.92, (rex + ew * 1.65) / hw),
        min(0.88, (ey_mid + eh * 1.75) / hh),
    )
    mouth_zone = (
        max(0.16, (mx - mw * 1.85) / hw),
        max(0.45, (my - mh * 1.95) / hh),
        min(0.84, (mx + mw * 1.85) / hw),
        min(0.98, (my + mh * 2.15) / hh),
    )
    f_px = max(5.0, min(9.5, min(ew, mh) * 0.75))

    return {
        "eyes_closed": extract_feathered_patch(head_crop_rgba, c_blink, dummy_bbox, face_interior_mask, eye_zone, f_px),
        "eyes_happy": extract_feathered_patch(head_crop_rgba, c_happy, dummy_bbox, face_interior_mask, eye_zone, f_px),
        "mouth_smile": extract_feathered_patch(head_crop_rgba, c_smile, dummy_bbox, face_interior_mask, mouth_zone, f_px),
        "mouth_laugh": extract_feathered_patch(head_crop_rgba, c_laugh, dummy_bbox, face_interior_mask, mouth_zone, f_px),
        "mouth_a": extract_feathered_patch(head_crop_rgba, c_a, dummy_bbox, face_interior_mask, mouth_zone, f_px),
        "mouth_o": extract_feathered_patch(head_crop_rgba, c_o, dummy_bbox, face_interior_mask, mouth_zone, f_px),
        "mouth_e": extract_feathered_patch(head_crop_rgba, c_e, dummy_bbox, face_interior_mask, mouth_zone, f_px),
    }


# ==============================================================================
# 4. Universal 6-Layer Capsule-Joint Limb Segmentation (`pants` + `skirt` modes)
# ==============================================================================

def _trace_limb_skeleton_3pts(
    limb_rgba: np.ndarray,
    proximal_xy: Tuple[float, float],
    mid_frac: float = 0.46,
    distal_frac: float = 0.83,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    ys, xs = np.where(limb_rgba[..., 3] > 60)
    if len(xs) < 25:
        return (proximal_xy[0], proximal_xy[1] + 80.0), (proximal_xy[0], proximal_xy[1] + 165.0)

    sx, sy = proximal_xy
    y_max = float(np.percentile(ys, 97))
    span_y = max(40.0, y_max - sy)

    def centroid_at_y(target_y: float) -> Tuple[float, float]:
        band = (ys >= target_y - 8) & (ys <= target_y + 8)
        if np.sum(band) >= 6:
            return float(np.median(xs[band])), float(target_y)
        idx = int(np.argmin(np.abs(ys - target_y)))
        return float(xs[idx]), float(ys[idx])

    mid_pt = centroid_at_y(sy + span_y * mid_frac)
    dist_pt = centroid_at_y(sy + span_y * distal_frac)
    return mid_pt, dist_pt


def segment_universal_limbs(
    body_crop_rgba: np.ndarray,
    bx0: int,
    by0: int,
    neck_cx_canvas: float,
    outfit_mode: str = "auto",
    landmark_overrides: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Any]:
    body = body_crop_rgba.copy()
    H, W, _ = body.shape
    rgb = body[..., :3].astype(np.float32)
    alpha = body[..., 3].astype(np.float32)
    fg = alpha > 25
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    ys_fg, _ = np.where(fg)
    b_top = int(ys_fg.min()) if len(ys_fg) else 0
    b_bot = int(ys_fg.max()) if len(ys_fg) else H - 1
    b_h = max(80, b_bot - b_top)
    cx = float(np.clip(neck_cx_canvas - bx0, W * 0.25, W * 0.75))
    c_int = int(round(cx))

    # --- Step A: Find Armpit Valleys, Exact Hand Bottoms, & Upper Shoulder Capsule Pivots ---
    arm_scan_y0 = b_top + int(b_h * 0.08)
    arm_scan_y1 = b_top + int(b_h * 0.64)

    left_gap_x: Dict[int, float] = {}
    right_gap_x: Dict[int, float] = {}

    for y in range(arm_scan_y0, arm_scan_y1):
        row_fg = alpha[y] > 30
        num_r, r_lbl = cv2.connectedComponents(row_fg[None, :].astype(np.uint8), connectivity=8)
        segs_y: List[Tuple[int, int]] = []
        for rid in range(1, num_r):
            rxs = np.where(r_lbl[0] == rid)[0]
            if len(rxs) >= 4:
                segs_y.append((int(rxs[0]), int(rxs[-1])))
        if len(segs_y) < 2:
            continue
        segs_y.sort(key=lambda t: t[0])

        # Identify central torso run OR central pair of thigh runs straddling `cx`
        body_lx: Optional[int] = None
        body_rx: Optional[int] = None
        for lx, rx in segs_y:
            if lx <= cx - 4.0 and rx >= cx + 4.0:
                body_lx, body_rx = lx, rx
                break
        if body_lx is None:
            for i in range(len(segs_y) - 1):
                ls, rs = segs_y[i], segs_y[i + 1]
                mid_legs = 0.5 * (ls[1] + rs[0])
                if ls[0] < cx and rs[1] > cx and abs(mid_legs - cx) < W * 0.12:
                    body_lx, body_rx = ls[0], rs[1]
                    break
        if body_lx is None or body_rx is None:
            continue

        left_arms = [s for s in segs_y if s[1] < body_lx - 1 and 0.5 * (s[0] + s[1]) < cx - W * 0.18]
        if left_arms:
            inner_la = left_arms[-1]
            left_gap_x[y] = 0.5 * (inner_la[1] + body_lx)

        right_arms = [s for s in segs_y if s[0] > body_rx + 1 and 0.5 * (s[0] + s[1]) > cx + W * 0.18]
        if right_arms:
            inner_ra = right_arms[0]
            right_gap_x[y] = 0.5 * (body_rx + inner_ra[0])

    chest_y = b_top + int(b_h * 0.20)
    chest_xs = np.where(fg[chest_y])[0]
    default_half_torso = float(W * 0.24) if len(chest_xs) < 10 else min(W * 0.26, 0.34 * (chest_xs[-1] - chest_xs[0]))

    if len(left_gap_x) >= 6:
        ly_sorted = sorted(left_gap_x.keys())
        armpit_yl = int(ly_sorted[0])
        armpit_xl = float(left_gap_x[armpit_yl])
        hand_bot_yl = int(ly_sorted[-1])
    else:
        armpit_yl = b_top + int(b_h * 0.20)
        armpit_xl = cx - default_half_torso
        hand_bot_yl = b_top + int(b_h * 0.50)

    if len(right_gap_x) >= 6:
        ry_sorted = sorted(right_gap_x.keys())
        armpit_yr = int(ry_sorted[0])
        armpit_xr = float(right_gap_x[armpit_yr])
        hand_bot_yr = int(ry_sorted[-1])
    else:
        armpit_yr = b_top + int(b_h * 0.20)
        armpit_xr = cx + default_half_torso
        hand_bot_yr = b_top + int(b_h * 0.50)

    gx = np.abs(cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3))

    def compute_shoulder_pivot(is_left: bool, ap_x: float, ap_y: int) -> Tuple[float, float, float, int, float]:
        probe_y = max(b_top + 10, ap_y - max(10, int(b_h * 0.038)))
        row_xs = np.where(fg[probe_y])[0]
        outer_x = float(row_xs[0] if is_left else row_xs[-1]) if len(row_xs) >= 8 else ap_x + (-36.0 if is_left else 36.0)

        # Respect vest/blazer armhole seam if it curves slightly outward above `ap_y`
        if is_left:
            s0 = max(int(round(outer_x)) + 16, int(round(ap_x)) - 22)
            s1 = min(c_int - 8, int(round(ap_x)) + 6)
        else:
            s0 = max(c_int + 8, int(round(ap_x)) - 6)
            s1 = min(int(round(outer_x)) - 16, int(round(ap_x)) + 22)
        if s1 > s0 + 3 and float(np.min(luma[probe_y, s0:s1])) < 105.0:
            seam_probe_x = float(s0 + int(np.argmin(luma[probe_y, s0:s1] - 0.35 * gx[probe_y, s0:s1])))
        else:
            seam_probe_x = ap_x + (-3.0 if is_left else 3.0)

        sleeve_w = float(np.clip(abs(seam_probe_x - outer_x), 22.0, 74.0))
        sh_rad = 0.48 * sleeve_w
        sx = (seam_probe_x - sh_rad) if is_left else (seam_probe_x + sh_rad)

        # Sample shoulder top across the outer span of the capsule so sloped shoulders never get a bump
        c_min = int(np.clip(round(sx - sh_rad * (0.65 if is_left else 0.35)), 0, W - 1))
        c_max = int(np.clip(round(sx + sh_rad * (0.35 if is_left else 0.65)), c_min + 1, W))
        col_tops = []
        for col_x in range(c_min, c_max):
            c_ys = np.where(fg[b_top : ap_y + 4, col_x])[0]
            if len(c_ys) > 0:
                col_tops.append(b_top + int(c_ys[0]))
        if col_tops:
            sh_top_y = int(np.percentile(col_tops, 70))
        else:
            sh_top_y = max(b_top + int(b_h * 0.07), ap_y - int(sh_rad * 1.8))
        sy = float(sh_top_y) + sh_rad + 1.5
        return sx, sy, sh_rad, min(col_tops) if col_tops else sh_top_y, seam_probe_x

    sx_l, sy_l, sh_rad_l, sh_top_yl, seam_probe_xl = compute_shoulder_pivot(True, armpit_xl, armpit_yl)
    sx_r, sy_r, sh_rad_r, sh_top_yr, seam_probe_xr = compute_shoulder_pivot(False, armpit_xr, armpit_yr)

    if landmark_overrides:
        if "leftShoulder" in landmark_overrides and len(landmark_overrides["leftShoulder"]) == 2:
            sx_l = float(landmark_overrides["leftShoulder"][0] - bx0)
            sy_l = float(landmark_overrides["leftShoulder"][1] - by0)
            sh_top_yl = max(b_top + 2, int(round(sy_l - sh_rad_l)))
        if "rightShoulder" in landmark_overrides and len(landmark_overrides["rightShoulder"]) == 2:
            sx_r = float(landmark_overrides["rightShoulder"][0] - bx0)
            sy_r = float(landmark_overrides["rightShoulder"][1] - by0)
            sh_top_yr = max(b_top + 2, int(round(sy_r - sh_rad_r)))

    left_seam = np.full(H, -10.0, dtype=np.float32)
    right_seam = np.full(H, float(W + 10), dtype=np.float32)

    l_top_x = min(armpit_xl, seam_probe_xl)
    for y in range(max(0, sh_top_yl - 3), min(H, armpit_yl + 1)):
        u = (y - sh_top_yl) / max(1.0, float(armpit_yl - sh_top_yl))
        guide_x = l_top_x * (1.0 - u) + armpit_xl * u
        s0 = max(2, int(round(guide_x)) - 9)
        s1 = min(c_int - 4, int(round(guide_x)) + 10)
        if s1 > s0 + 2:
            cost = luma[y, s0:s1] - 0.45 * gx[y, s0:s1] + np.where(alpha[y, s0:s1] > 80, 0.0, 400.0)
            left_seam[y] = float(s0 + int(np.argmin(cost)))
        else:
            left_seam[y] = guide_x

    r_top_x = max(armpit_xr, seam_probe_xr)
    for y in range(max(0, sh_top_yr - 3), min(H, armpit_yr + 1)):
        u = (y - sh_top_yr) / max(1.0, float(armpit_yr - sh_top_yr))
        guide_x = r_top_x * (1.0 - u) + armpit_xr * u
        s0 = max(c_int + 4, int(round(guide_x)) - 9)
        s1 = min(W - 2, int(round(guide_x)) + 10)
        if s1 > s0 + 2:
            cost = luma[y, s0:s1] - 0.45 * gx[y, s0:s1] + np.where(alpha[y, s0:s1] > 80, 0.0, 400.0)
            right_seam[y] = float(s0 + int(np.argmin(cost)))
        else:
            right_seam[y] = guide_x

    prev_lx = left_seam[min(H - 1, armpit_yl)]
    actual_hand_bot_l = armpit_yl
    for y in range(armpit_yl + 1, min(H, hand_bot_yl + 35)):
        if y in left_gap_x:
            prev_lx = float(left_gap_x[y])
        else:
            s0 = max(2, int(round(prev_lx)) - 10)
            s1 = min(c_int - 8, int(round(prev_lx)) + 10)
            if s1 > s0 + 2 and np.min(alpha[y, s0:s1]) < 35:
                prev_lx = float(s0 + int(np.argmin(alpha[y, s0:s1])))
        if np.any(alpha[y, : max(1, int(round(prev_lx)) - 1)] > 40):
            left_seam[y] = prev_lx
            actual_hand_bot_l = y
        elif y > hand_bot_yl:
            break

    prev_rx = right_seam[min(H - 1, armpit_yr)]
    actual_hand_bot_r = armpit_yr
    for y in range(armpit_yr + 1, min(H, hand_bot_yr + 35)):
        if y in right_gap_x:
            prev_rx = float(right_gap_x[y])
        else:
            s0 = max(c_int + 8, int(round(prev_rx)) - 10)
            s1 = min(W - 2, int(round(prev_rx)) + 10)
            if s1 > s0 + 2 and np.min(alpha[y, s0:s1]) < 35:
                prev_rx = float(s0 + int(np.argmin(alpha[y, s0:s1])))
        if np.any(alpha[y, min(W - 1, int(round(prev_rx)) + 2) :] > 40):
            right_seam[y] = prev_rx
            actual_hand_bot_r = y
        elif y > hand_bot_yr:
            break

    if actual_hand_bot_l > sh_top_yl + 4:
        left_seam[sh_top_yl : actual_hand_bot_l + 1] = gaussian_filter(
            left_seam[sh_top_yl : actual_hand_bot_l + 1], sigma=1.5
        )
    if actual_hand_bot_r > sh_top_yr + 4:
        right_seam[sh_top_yr : actual_hand_bot_r + 1] = gaussian_filter(
            right_seam[sh_top_yr : actual_hand_bot_r + 1], sigma=1.5
        )

    # --- Step B: Detect Crotch / Skirt Hem via Exact 2-Segment Leg Run Finder ---
    crotch_gap_y = b_top + int(b_h * 0.50)
    leg_mid_x = np.full(H, cx, dtype=np.float32)

    found_leg_gaps: List[Tuple[int, float]] = []
    for y in range(b_top + int(b_h * 0.93), b_top + int(b_h * 0.35), -1):
        row_mask = alpha[y] > 45
        if y <= actual_hand_bot_l + 2:
            row_mask[: max(0, int(round(left_seam[y])) + 1)] = False
        if y <= actual_hand_bot_r + 2:
            row_mask[min(W, int(round(right_seam[y]))) :] = False

        # Find contiguous foreground runs on row `y`
        num_s, s_lbl = cv2.connectedComponents(row_mask[None, :].astype(np.uint8), connectivity=8)
        segs: List[Tuple[int, int]] = []
        for sid in range(1, num_s):
            sxs = np.where(s_lbl[0] == sid)[0]
            if len(sxs) >= 8:
                segs.append((int(sxs[0]), int(sxs[-1])))
        segs.sort(key=lambda t: t[0])

        # Find left-leg run and right-leg run straddling `cx`
        pair_gap = None
        for i in range(len(segs) - 1):
            l_seg, r_seg = segs[i], segs[i + 1]
            if l_seg[0] < cx + W * 0.06 and r_seg[1] > cx - W * 0.06:
                mid_gx = 0.5 * (l_seg[1] + r_seg[0])
                if abs(mid_gx - cx) < W * 0.14:
                    pair_gap = mid_gx
                    break

        if pair_gap is not None:
            found_leg_gaps.append((y, pair_gap))
            leg_mid_x[y] = pair_gap
        elif len(found_leg_gaps) > 8:
            inseam_s0 = max(0, int(round(found_leg_gaps[-1][1])) - 6)
            inseam_s1 = min(W, int(round(found_leg_gaps[-1][1])) + 7)
            if inseam_s1 > inseam_s0 and np.min(luma[y, inseam_s0:inseam_s1]) < 95.0:
                gx_in = float(inseam_s0 + int(np.argmin(luma[y, inseam_s0:inseam_s1])))
                horiz_dark = np.mean(luma[y, max(0, int(gx_in) - 18) : min(W, int(gx_in) + 19)] < 110.0)
                if horiz_dark > 0.55:
                    crotch_gap_y = y + 1
                    break
                found_leg_gaps.append((y, gx_in))
                leg_mid_x[y] = gx_in
            else:
                crotch_gap_y = y + 1
                break

    if found_leg_gaps:
        crotch_gap_y = int(min(y for y, _ in found_leg_gaps))
        ys_g = np.array([y for y, _ in reversed(found_leg_gaps)], dtype=np.float32)
        xs_g = np.array([x for _, x in reversed(found_leg_gaps)], dtype=np.float32)
        leg_mid_x = np.interp(np.arange(H, dtype=np.float32), ys_g, xs_g).astype(np.float32)
        leg_mid_x = gaussian_filter(leg_mid_x, sigma=2.0)

    crotch_frac = (crotch_gap_y - b_top) / float(b_h)
    if outfit_mode in ("pants", "skirt"):
        resolved_outfit = outfit_mode
    else:
        resolved_outfit = "skirt" if crotch_frac >= 0.58 else "pants"

    override_hip_y: Optional[float] = None
    if landmark_overrides:
        hy_vals = [
            float(landmark_overrides[hk][1] - by0)
            for hk in ("leftHip", "rightHip")
            if hk in landmark_overrides and len(landmark_overrides[hk]) == 2
        ]
        if hy_vals:
            override_hip_y = float(np.mean(hy_vals))

    torso_hem_y = np.full(W, float(crotch_gap_y - 2), dtype=np.float32)
    gy_abs = np.abs(cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3))

    # Measure torso horizontal span near `crotch_gap_y - 10`
    hem_probe_y = max(b_top + 20, crotch_gap_y - 10)
    # Exclude arm columns if arms still exist at `hem_probe_y`
    hem_row_mask = fg[hem_probe_y].copy()
    if hem_probe_y <= actual_hand_bot_l:
        hem_row_mask[: max(0, int(round(left_seam[hem_probe_y])) + 1)] = False
    if hem_probe_y <= actual_hand_bot_r:
        hem_row_mask[min(W, int(round(right_seam[hem_probe_y]))) :] = False
    hem_xs = np.where(hem_row_mask)[0]
    hem_x0 = int(hem_xs[0]) if len(hem_xs) >= 10 else max(4, c_int - int(W * 0.24))
    hem_x1 = int(hem_xs[-1]) if len(hem_xs) >= 10 else min(W - 4, c_int + int(W * 0.24))

    if resolved_outfit == "skirt":
        s_top = max(b_top + int(b_h * 0.42), crotch_gap_y - int(b_h * 0.06))
        s_bot = min(b_bot - 25, crotch_gap_y + int(b_h * 0.035))
        hx_pts, hy_pts = [], []
        for x in range(hem_x0, hem_x1 + 1):
            if s_bot > s_top + 4:
                col_score = gy_abs[s_top:s_bot, x] - 0.65 * luma[s_top:s_bot, x]
                valid_a = alpha[s_top:s_bot, x] > 140
                if np.any(valid_a):
                    col_score = np.where(valid_a, col_score, -1e6)
                    hx_pts.append(x)
                    hy_pts.append(float(s_top + int(np.argmax(col_score))))
        if len(hx_pts) >= 12:
            torso_hem_y = np.interp(np.arange(W), hx_pts, hy_pts).astype(np.float32)
            torso_hem_y = median_filter(torso_hem_y, size=15)
            torso_hem_y = gaussian_filter(torso_hem_y, sigma=2.2)
    else:
        s_top = max(b_top + int(b_h * 0.38), crotch_gap_y - int(b_h * 0.045))
        s_bot = min(b_bot - 30, crotch_gap_y + 8)
        edge_strength = (
            float(np.mean(gy_abs[s_top:s_bot, hem_x0:hem_x1]))
            if (s_bot > s_top + 2 and hem_x1 > hem_x0 + 4)
            else 0.0
        )
        if edge_strength > 34.0 and s_bot > s_top + 4:
            hx_pts, hy_pts = [], []
            for x in range(hem_x0, hem_x1 + 1):
                col_score = gy_abs[s_top:s_bot, x] - 0.55 * luma[s_top:s_bot, x]
                valid_a = alpha[s_top:s_bot, x] > 140
                if np.any(valid_a):
                    col_score = np.where(valid_a, col_score, -1e6)
                    hx_pts.append(x)
                    hy_pts.append(float(s_top + int(np.argmax(col_score))))
            if len(hx_pts) >= 12:
                torso_hem_y = np.interp(np.arange(W), hx_pts, hy_pts).astype(np.float32)
                torso_hem_y = median_filter(torso_hem_y, size=15)
                torso_hem_y = gaussian_filter(torso_hem_y, sigma=2.0)
        else:
            # Long trousers: convex V-arch peaking at `crotch_gap_y - 2` at center and curving UP 24px
            # toward the outer hip sockets so rotating thighs tuck smoothly behind the pelvis!
            base_py = float(crotch_gap_y - 2)
            xs_arr = np.arange(W, dtype=np.float32)
            norm_dx = np.clip((xs_arr - cx) / max(25.0, 0.5 * (hem_x1 - hem_x0)), -1.2, 1.2)
            torso_hem_y = base_py - 24.0 * (norm_dx ** 2)

    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")

    # --- Step C: Build `left_arm` & `right_arm` FIRST with Single-Component Capsule Caps ---
    body_outer_dist = distance_transform_edt(alpha > 180)

    def build_universal_arm(
        is_left: bool,
        sx: float,
        sy: float,
        sh_rad: float,
        ap_y: int,
        hand_bot_y: int,
        seam_arr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        arm = np.zeros_like(body)
        dist_sh = np.hypot(xx - sx, yy - sy)
        raw_mask = np.zeros((H, W), dtype=np.float32)

        y_start = max(0, int(round(sy - sh_rad - 2)))
        y_end = min(H - 1, hand_bot_y + 3)
        for y in range(y_start, y_end + 1):
            if y < ap_y + 4:
                dy = y - sy
                cap_dx = np.sqrt(max(0.0, sh_rad * sh_rad - min(sh_rad, abs(dy)) ** 2))
                blend = np.clip((y - (sy + 2.0)) / max(10.0, float(ap_y - sy)), 0.0, 1.0)
                if is_left:
                    cap_in_x = sx + cap_dx
                    bound_x = cap_in_x * (1.0 - blend) + (seam_arr[y] - 0.2) * blend
                    raw_mask[y] = np.clip(((bound_x + 0.6) - xx[y]) / 1.4, 0.0, 1.0)
                else:
                    cap_in_x = sx - cap_dx
                    bound_x = cap_in_x * (1.0 - blend) + (seam_arr[y] + 0.2) * blend
                    raw_mask[y] = np.clip((xx[y] - (bound_x - 0.6)) / 1.4, 0.0, 1.0)
            else:
                if is_left:
                    raw_mask[y] = (xx[y] < seam_arr[y]).astype(np.float32)
                else:
                    raw_mask[y] = (xx[y] > seam_arr[y]).astype(np.float32)

        above_cap = yy <= (sy + 4.0)
        cap_feather = np.clip((sh_rad + 1.0 - dist_sh) / 2.2, 0.0, 1.0)
        raw_mask = np.where(above_cap, np.minimum(raw_mask, cap_feather), raw_mask)

        arm[..., :3] = body[..., :3]
        arm[..., 3] = np.round(alpha * raw_mask).astype(np.uint8)
        arm = keep_component_near_anchor(arm, sx, sy, alpha_thresh=10)

        # Effective rest-pose claim mask (strictly restricted to the single connected arm component!)
        claim_mask = np.clip(arm[..., 3].astype(np.float32) / np.maximum(alpha, 1.0), 0.0, 1.0)
        claim_mask[alpha < 15.0] = 0.0

        sleeve_seed = (dist_sh <= sh_rad * 1.1) & (arm[..., 3] > 210) & (body_outer_dist >= 3.5)
        if np.sum(sleeve_seed) >= 12:
            seed_luma = luma[sleeve_seed]
            bright_enough = seed_luma >= np.percentile(seed_luma, 30)
            sleeve_rgb = np.median(rgb[sleeve_seed][bright_enough], axis=0)
        else:
            sleeve_rgb = np.array([220.0, 220.0, 225.0], dtype=np.float32)
        sleeve_luma = float(0.299 * sleeve_rgb[0] + 0.587 * sleeve_rgb[1] + 0.114 * sleeve_rgb[2])

        in_ball = (dist_sh <= sh_rad) & (yy <= sy + sh_rad * 0.85) & (alpha > 140.0)
        is_outer_rim = (body_outer_dist > 0) & (body_outer_dist < 3.4) & (
            (xx < sx - 2.0) if is_left else (xx > sx + 2.0)
        )
        fill_cap = in_ball & (~is_outer_rim) & ((arm[..., 3] < 245) | (luma < sleeve_luma * 0.68))
        arm[fill_cap, :3] = np.round(sleeve_rgb).astype(np.uint8)
        ball_alpha = np.round(np.clip((sh_rad + 0.8 - dist_sh) / 2.2, 0.0, 1.0) * 255.0).astype(np.uint8)
        arm[fill_cap, 3] = np.maximum(arm[fill_cap, 3], ball_alpha[fill_cap])

        return keep_component_near_anchor(arm, sx, sy, alpha_thresh=10), claim_mask

    left_arm, arm_claim_l = build_universal_arm(True, sx_l, sy_l, sh_rad_l, armpit_yl, actual_hand_bot_l, left_seam)
    right_arm, arm_claim_r = build_universal_arm(False, sx_r, sy_r, sh_rad_r, armpit_yr, actual_hand_bot_r, right_seam)

    # --- Step D: Build `torso` via Partition of Unity (`1 - arm_claim`) + Hem Curve ---
    # Guarantees ZERO missing shoulder/collar wedges at rest while carving clean armholes!
    torso = np.zeros_like(body)
    above_hem_2d = np.clip((torso_hem_y[None, :] + 1.4) - yy, 0.0, 1.0)
    # 1.5px subpixel overlap behind arm boundaries so bilinear rendering has zero hairline seam
    not_left_arm = 1.0 - np.clip((arm_claim_l - 0.18) / 0.82, 0.0, 1.0)
    not_right_arm = 1.0 - np.clip((arm_claim_r - 0.18) / 0.82, 0.0, 1.0)
    torso_mask = not_left_arm * not_right_arm * above_hem_2d

    torso[..., :3] = body[..., :3]
    torso[..., 3] = np.round(alpha * torso_mask).astype(np.uint8)
    torso = keep_largest_components(torso, alpha_thresh=10, min_ratio=0.05)

    torso_fg = torso[..., 3] > 200
    dist_torso_in = distance_transform_edt(torso_fg)
    for is_left_side, sh_top, ap_y in ((True, sh_top_yl, armpit_yl), (False, sh_top_yr, armpit_yr)):
        seam_arr = left_seam if is_left_side else right_seam
        sample_mask = (
            (yy >= sh_top + 6)
            & (yy <= ap_y - 2)
            & (np.abs(xx - (seam_arr[np.clip(yy.astype(int), 0, H - 1)] + (14.0 if is_left_side else -14.0))) <= 6.0)
            & torso_fg
            & (luma > 65.0)
        )
        if np.sum(sample_mask) >= 15:
            local_torso_rgb = np.median(rgb[sample_mask], axis=0)
            for y in range(sh_top + 2, min(H, ap_y - 4)):
                s_x = seam_arr[y]
                band = (np.abs(xx[y] - s_x) <= 5.0) & (dist_torso_in[y] >= 3.5) & (luma[y] < np.min(local_torso_rgb) * 0.72)
                torso[y, band, :3] = np.round(local_torso_rgb).astype(np.uint8)

    # --- Step E: Build `left_leg` & `right_leg` with Circular Hip Socket Dome Behind Pelvis/Skirt ---
    if override_hip_y is not None:
        anat_hip_y = float(np.clip(override_hip_y, b_top + b_h * 0.36, b_bot - 60))
    elif resolved_outfit == "skirt":
        anat_hip_y = float(min(crotch_gap_y - b_h * 0.12, b_top + b_h * 0.49))
    else:
        anat_hip_y = float(crotch_gap_y - max(24.0, b_h * 0.042))

    def build_universal_leg(is_left: bool) -> Tuple[np.ndarray, float]:
        leg = np.zeros_like(body)
        leg_start_y = max(b_top + int(b_h * 0.34), int(np.min(torso_hem_y)) - 2)
        hem_margin = 2.2 if resolved_outfit == "skirt" else 0.5
        for y in range(leg_start_y, H):
            mid_x = leg_mid_x[y]
            side_ok = (xx[y] < mid_x) if is_left else (xx[y] >= mid_x)
            if is_left and y <= actual_hand_bot_l + 2:
                side_ok = side_ok & (arm_claim_l[y] < 0.25)
            elif (not is_left) and y <= actual_hand_bot_r + 2:
                side_ok = side_ok & (arm_claim_r[y] < 0.25)
            below_hem = float(y) >= (torso_hem_y + hem_margin)
            valid = side_ok & below_hem
            leg[y, valid] = body[y, valid]

        anchor_x = (cx - W * 0.12) if is_left else (cx + W * 0.12)
        anchor_y = b_top + b_h * 0.82
        leg = keep_component_near_anchor(leg, anchor_x, anchor_y, alpha_thresh=12)

        ys_l, _ = np.where(leg[..., 3] > 180)
        hip_cx = anchor_x
        if len(ys_l) >= 30:
            # Sample strictly below `crotch_gap_y` so convex V-arches always capture the FULL thigh width!
            ref_y = min(H - 10, max(int(ys_l.min()) + 14, crotch_gap_y + 10))
            xs_ref = np.where(leg[ref_y, :, 3] > 170)[0]
            if len(xs_ref) < 8:
                ref_y = min(H - 6, int(np.percentile(ys_l, 15)) + 10)
                xs_ref = np.where(leg[ref_y, :, 3] > 150)[0]
            if len(xs_ref) >= 8:
                tx0, tx1 = int(xs_ref[0]), int(xs_ref[-1])
                thigh_w = float(tx1 - tx0 + 1)
                hip_cx = 0.5 * (tx0 + tx1)
                ref_row = leg[ref_y, tx0 : tx1 + 1].copy()

                cap_r = 0.50 * thigh_w
                top_cap_y = max(b_top + int(b_h * 0.30), int(round(anat_hip_y - cap_r)))
                for y in range(ref_y - 1, top_cap_y - 1, -1):
                    if y >= anat_hip_y:
                        cur_w = max(8, int(round(thigh_w)))
                        fade = 1.0
                    else:
                        dy_cap = anat_hip_y - float(y)
                        dome = np.sqrt(max(0.02, 1.0 - min(1.0, dy_cap / max(4.0, cap_r)) ** 2))
                        cur_w = max(6, int(round(thigh_w * dome)))
                        fade = float(np.clip((cap_r + 1.0 - dy_cap) / 2.5, 0.0, 1.0))

                    c0 = max(0, int(round(hip_cx - cur_w * 0.5)))
                    c1 = min(W, c0 + cur_w)
                    if c1 <= c0 + 2:
                        continue
                    resized = cv2.resize(ref_row[np.newaxis, :, :], (c1 - c0, 1), interpolation=cv2.INTER_LINEAR)[0]
                    new_a = np.round(resized[:, 3].astype(np.float32) * fade).astype(np.uint8)
                    shade = 0.96 if (resolved_outfit == "skirt" and y < ref_y - 12) else 1.0
                    new_rgb = np.clip(resized[:, :3].astype(np.float32) * shade, 0, 255).astype(np.uint8)
                    # Cleanly overwrite above `ref_y` so any residual hem fringe at `ys_l.min()` is erased!
                    leg[y, max(0, c0 - 2) : min(W, c1 + 2)] = 0
                    valid_px = new_a > 0
                    leg[y, c0:c1, :3][valid_px] = new_rgb[valid_px]
                    leg[y, c0:c1, 3][valid_px] = new_a[valid_px]

        return keep_component_near_anchor(leg, hip_cx, b_top + b_h * 0.80, alpha_thresh=10), hip_cx

    left_leg, l_hip_cx = build_universal_leg(True)
    right_leg, r_hip_cx = build_universal_leg(False)

    # --- Step F: Compute 12 Anatomical 4-Limb Landmarks in Canvas Coordinates ---
    l_el, l_wr = _trace_limb_skeleton_3pts(left_arm, (sx_l, sy_l), mid_frac=0.46, distal_frac=0.81)
    r_el, r_wr = _trace_limb_skeleton_3pts(right_arm, (sx_r, sy_r), mid_frac=0.46, distal_frac=0.81)
    l_kn, l_an = _trace_limb_skeleton_3pts(left_leg, (l_hip_cx, anat_hip_y), mid_frac=0.48, distal_frac=0.85)
    r_kn, r_an = _trace_limb_skeleton_3pts(right_leg, (r_hip_cx, anat_hip_y), mid_frac=0.48, distal_frac=0.85)

    limb_landmarks: Dict[str, List[float]] = {
        "leftShoulder": [round(bx0 + sx_l, 1), round(by0 + sy_l, 1)],
        "leftElbow": [round(bx0 + l_el[0], 1), round(by0 + l_el[1], 1)],
        "leftWrist": [round(bx0 + l_wr[0], 1), round(by0 + l_wr[1], 1)],
        "rightShoulder": [round(bx0 + sx_r, 1), round(by0 + sy_r, 1)],
        "rightElbow": [round(bx0 + r_el[0], 1), round(by0 + r_el[1], 1)],
        "rightWrist": [round(bx0 + r_wr[0], 1), round(by0 + r_wr[1], 1)],
        "leftHip": [round(bx0 + l_hip_cx, 1), round(by0 + anat_hip_y, 1)],
        "leftKnee": [round(bx0 + l_kn[0], 1), round(by0 + l_kn[1], 1)],
        "leftAnkle": [round(bx0 + l_an[0], 1), round(by0 + l_an[1], 1)],
        "rightHip": [round(bx0 + r_hip_cx, 1), round(by0 + anat_hip_y, 1)],
        "rightKnee": [round(bx0 + r_kn[0], 1), round(by0 + r_kn[1], 1)],
        "rightAnkle": [round(bx0 + r_an[0], 1), round(by0 + r_an[1], 1)],
    }

    if landmark_overrides:
        for k in limb_landmarks:
            if k in landmark_overrides and len(landmark_overrides[k]) == 2:
                limb_landmarks[k] = [round(float(landmark_overrides[k][0]), 1), round(float(landmark_overrides[k][1]), 1)]

    return {
        "torso": torso,
        "left_arm": left_arm,
        "right_arm": right_arm,
        "left_leg": left_leg,
        "right_leg": right_leg,
        "limb_landmarks": limb_landmarks,
        "resolved_outfit": resolved_outfit,
        "hips_y_canvas": round(by0 + anat_hip_y, 1),
        "knees_y_canvas": round(by0 + 0.5 * (l_kn[1] + r_kn[1]), 1),
    }


# ==============================================================================
# 5. End-to-End Universal Bundle Compiler & Re-Rigging Entrypoint
# ==============================================================================

def build_universal_bundle(
    base_image_path: Path,
    output_dir: Path,
    character_id: str,
    character_name: str,
    outfit_mode: str = "auto",
    variants: Optional[Dict[str, Path]] = None,
    landmark_overrides: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    patches_dir = output_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    pil_raw = Image.open(base_image_path)
    max_dim = max(pil_raw.size)
    if max_dim > 1500:
        scale = 1280.0 / float(max_dim)
        new_sz = (max(64, int(round(pil_raw.width * scale))), max(64, int(round(pil_raw.height * scale))))
        pil_raw = pil_raw.resize(new_sz, Image.Resampling.LANCZOS)

    W, H = pil_raw.size
    src_out = output_dir / "source.png"
    if base_image_path.resolve() != src_out.resolve():
        pil_raw.convert("RGBA").save(src_out, format="PNG")

    base_rgba = remove_universal_background(pil_raw)
    save_rgba_png_clean(base_rgba, output_dir / "full_rgba.png")

    fg = base_rgba[..., 3] > 25
    ys_all, xs_all = np.where(fg)
    if len(xs_all) < 50:
        raise ValueError("未检测到有效卡通人物主体，请上传主体清晰的人物图片。")
    char_x0, char_y0 = int(xs_all.min()), int(ys_all.min())
    char_x1, char_y1 = int(xs_all.max()), int(ys_all.max())

    override_neck = None
    if landmark_overrides and "neckPivot" in landmark_overrides and len(landmark_overrides["neckPivot"]) == 2:
        override_neck = (float(landmark_overrides["neckPivot"][0]), float(landmark_overrides["neckPivot"][1]))

    neck_y, neck_xl, neck_xr, head_mask = find_adaptive_head_neck_split(base_rgba, override_neck)
    head_full_rgba, body_full_rgba = split_head_and_body_universal(
        base_rgba, neck_y, neck_xl, neck_xr, head_mask
    )

    pad = 14
    head_crop, head_spec = crop_layer_spec(head_full_rgba, 0, 0, pad=pad, cols=16, rows=16, tex_name="head.png")
    body_crop, body_spec = crop_layer_spec(body_full_rgba, 0, 0, pad=pad, cols=12, rows=20, tex_name="body.png")
    save_rgba_png_clean(head_crop, output_dir / "head.png")
    save_rgba_png_clean(body_crop, output_dir / "body.png")

    hx0, hy0 = int(head_spec["x"]), int(head_spec["y"])
    head_bbox = (hx0, hy0, hx0 + int(head_spec["width"]), hy0 + int(head_spec["height"]))
    bx0, by0 = int(body_spec["x"]), int(body_spec["y"])

    feat = detect_face_and_features(head_crop, head_bbox, landmark_overrides=landmark_overrides)
    face_interior = feat["face_interior_mask"]
    patch_files: Dict[str, str] = {}

    if variants:
        base_bgr = cv2.cvtColor(np.array(pil_raw.convert("RGB")), cv2.COLOR_RGB2BGR)
        aligned_rgba_cache: Dict[str, np.ndarray] = {}
        for key, vpath in variants.items():
            if not Path(vpath).exists():
                continue
            v_pil = Image.open(vpath).convert("RGB")
            if v_pil.size != (W, H):
                v_pil = v_pil.resize((W, H), Image.Resampling.LANCZOS)
            vbgr = cv2.cvtColor(np.array(v_pil), cv2.COLOR_RGB2BGR)
            aligned_bgr = align_variant_to_base(base_bgr, vbgr, head_bbox)
            aligned_pil = Image.fromarray(cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB))
            aligned_rgba_cache[key] = remove_universal_background(aligned_pil)

        hh_c, hw_c = head_crop.shape[:2]
        lex_c, ley_c = feat["left_eye"]
        rex_c, rey_c = feat["right_eye"]
        ew_c, eh_c = feat["eye_radius"]
        mx_c, my_c = feat["mouth"]
        mw_c = feat["mouth_half_w"]
        ey_mid = 0.5 * (ley_c + rey_c)
        eye_zone_norm = (
            max(0.10, (lex_c - ew_c * 1.8) / hw_c),
            max(0.20, (ey_mid - eh_c * 2.1) / hh_c),
            min(0.90, (rex_c + ew_c * 1.8) / hw_c),
            min(0.82, (ey_mid + eh_c * 1.9) / hh_c),
        )
        mouth_blush_zone_norm = (
            max(0.15, (mx_c - mw_c * 2.3) / hw_c),
            max(0.55, (my_c - mw_c * 1.4) / hh_c),
            min(0.85, (mx_c + mw_c * 2.3) / hw_c),
            min(0.97, (my_c + mw_c * 1.6) / hh_c),
        )
        mouth_tight_zone_norm = (
            max(0.22, (mx_c - mw_c * 1.8) / hw_c),
            max(0.60, (my_c - mw_c * 1.2) / hh_c),
            min(0.78, (mx_c + mw_c * 1.8) / hw_c),
            min(0.97, (my_c + mw_c * 1.5) / hh_c),
        )

        if "eyes_closed" in aligned_rgba_cache:
            p = extract_feathered_patch(head_crop, aligned_rgba_cache["eyes_closed"], head_bbox, face_interior, eye_zone_norm, 10.0)
            save_rgba_png_clean(p, patches_dir / "eyes_closed.png")
            patch_files["eyes_closed"] = "patches/eyes_closed.png"
        if "laugh_joy" in aligned_rgba_cache:
            p_eh = extract_feathered_patch(head_crop, aligned_rgba_cache["laugh_joy"], head_bbox, face_interior, eye_zone_norm, 10.0)
            save_rgba_png_clean(p_eh, patches_dir / "eyes_happy.png")
            patch_files["eyes_happy"] = "patches/eyes_happy.png"
            p_ml = extract_feathered_patch(head_crop, aligned_rgba_cache["laugh_joy"], head_bbox, face_interior, mouth_blush_zone_norm, 11.0)
            save_rgba_png_clean(p_ml, patches_dir / "mouth_laugh.png")
            patch_files["mouth_laugh"] = "patches/mouth_laugh.png"
        if "smile_warm" in aligned_rgba_cache:
            p_ms = extract_feathered_patch(head_crop, aligned_rgba_cache["smile_warm"], head_bbox, face_interior, mouth_blush_zone_norm, 11.0)
            save_rgba_png_clean(p_ms, patches_dir / "mouth_smile.png")
            patch_files["mouth_smile"] = "patches/mouth_smile.png"
        for viseme_key, out_name in [("talk_a", "mouth_a"), ("talk_e", "mouth_e"), ("talk_o", "mouth_o")]:
            if viseme_key in aligned_rgba_cache:
                p_v = extract_feathered_patch(head_crop, aligned_rgba_cache[viseme_key], head_bbox, face_interior, mouth_tight_zone_norm, 9.5)
                save_rgba_png_clean(p_v, patches_dir / f"{out_name}.png")
                patch_files[out_name] = f"patches/{out_name}.png"

    proc_patches = synthesize_universal_patches(head_crop, feat)
    for k, arr in proc_patches.items():
        if k not in patch_files:
            save_rgba_png_clean(arr, patches_dir / f"{k}.png")
            patch_files[k] = f"patches/{k}.png"

    neck_cx = 0.5 * (neck_xl + neck_xr)
    limb_res = segment_universal_limbs(
        body_crop_rgba=body_crop,
        bx0=bx0,
        by0=by0,
        neck_cx_canvas=neck_cx,
        outfit_mode=outfit_mode,
        landmark_overrides=landmark_overrides,
    )

    torso_crop, torso_spec = crop_layer_spec(limb_res["torso"], bx0, by0, pad=8, cols=12, rows=16, tex_name="torso.png")
    l_arm_crop, l_arm_spec = crop_layer_spec(limb_res["left_arm"], bx0, by0, pad=8, cols=10, rows=22, tex_name="left_arm.png")
    r_arm_crop, r_arm_spec = crop_layer_spec(limb_res["right_arm"], bx0, by0, pad=8, cols=10, rows=22, tex_name="right_arm.png")
    l_leg_crop, l_leg_spec = crop_layer_spec(limb_res["left_leg"], bx0, by0, pad=8, cols=10, rows=22, tex_name="left_leg.png")
    r_leg_crop, r_leg_spec = crop_layer_spec(limb_res["right_leg"], bx0, by0, pad=8, cols=10, rows=22, tex_name="right_leg.png")

    save_rgba_png_clean(torso_crop, output_dir / "torso.png")
    save_rgba_png_clean(l_arm_crop, output_dir / "left_arm.png")
    save_rgba_png_clean(r_arm_crop, output_dir / "right_arm.png")
    save_rgba_png_clean(l_leg_crop, output_dir / "left_leg.png")
    save_rgba_png_clean(r_leg_crop, output_dir / "right_leg.png")

    lex_w = hx0 + feat["left_eye"][0]
    ley_w = hy0 + feat["left_eye"][1]
    rex_w = hx0 + feat["right_eye"][0]
    rey_w = hy0 + feat["right_eye"][1]
    mx_w = hx0 + feat["mouth"][0]
    my_w = hy0 + feat["mouth"][1]
    fx0_l, fy0_l, fx1_l, fy1_l = feat["face_box"]
    face_cx_w = hx0 + 0.5 * (fx0_l + fx1_l)
    face_cy_w = hy0 + 0.5 * (fy0_l + fy1_l)
    nose_y_w = 0.45 * (0.5 * (ley_w + rey_w)) + 0.55 * my_w

    char_h = float(char_y1 - char_y0)
    landmarks: Dict[str, List[float]] = {
        "headCenter": [round(face_cx_w, 1), round(face_cy_w, 1)],
        "leftEye": [round(lex_w, 1), round(ley_w, 1)],
        "rightEye": [round(rex_w, 1), round(rey_w, 1)],
        "nose": [round(face_cx_w, 1), round(nose_y_w, 1)],
        "mouth": [round(mx_w, 1), round(my_w, 1)],
        "neckPivot": (
            [round(override_neck[0], 1), round(override_neck[1], 1)]
            if override_neck
            else [round(neck_cx, 1), round(float(neck_y - 4), 1)]
        ),
        "chest": [round(neck_cx, 1), round(neck_y + char_h * 0.14, 1)],
        "hips": [round(neck_cx, 1), limb_res["hips_y_canvas"]],
        "knees": [round(neck_cx, 1), limb_res["knees_y_canvas"]],
        "feetGround": [round(neck_cx, 1), round(float(char_y1 - 6), 1)],
        **limb_res["limb_landmarks"],
    }

    if landmark_overrides and "feetGround" in landmark_overrides and len(landmark_overrides["feetGround"]) == 2:
        landmarks["feetGround"] = [
            round(float(landmark_overrides["feetGround"][0]), 1),
            round(float(landmark_overrides["feetGround"][1]), 1),
        ]

    manifest: Dict[str, Any] = {
        "version": "1.2.0",
        "id": character_id,
        "name": character_name,
        "outfitMode": limb_res["resolved_outfit"],
        "hasGlasses": bool(feat.get("has_glasses", False)),
        "canvasSize": {"width": W, "height": H},
        "characterBounds": {
            "x": char_x0,
            "y": char_y0,
            "width": char_x1 - char_x0,
            "height": char_y1 - char_y0,
        },
        "layers": {
            "head": head_spec,
            "body": body_spec,
            "torso": torso_spec,
            "leftArm": l_arm_spec,
            "rightArm": r_arm_spec,
            "leftLeg": l_leg_spec,
            "rightLeg": r_leg_spec,
        },
        "patches": patch_files,
        "landmarks": landmarks,
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
