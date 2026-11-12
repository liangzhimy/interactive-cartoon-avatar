#!/usr/bin/env python3
"""
AI Avatar Asset Preprocessor & Bundle Compiler (`build_avatar_bundle.py`)

Given a base full-body cartoon character image (and optional AI-generated facial
expression / viseme variants), this pipeline produces a standardized `AvatarBundle`:
  - `source.png`        : Clean copy of the original input full-body illustration
  - `full_rgba.png`     : Alpha-matted full-body character with de-fringed outlines
  - `body.png`          : Decoupled body/torso/limbs layer with upward-extended neck
  - `head.png`          : Decoupled head & hair layer (clean chin/hair silhouette)
  - `patches/*.png`     : Sub-pixel ECC-aligned, color-matched, feathered facial patches
                          in exact `head.png` local coordinates so they share the 2.5D
                          head deformation mesh with zero drift
  - `manifest.json`     : Normalized rig landmarks, pivot coordinates, bounds & textures
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt


def remove_white_background(bgr: np.ndarray) -> np.ndarray:
    """
    Extract clean RGBA from a cartoon character drawn with crisp outlines on a
    solid white (#FFFFFF) or near-white studio background, eliminating white edge halos.
    """
    h, w, _ = bgr.shape
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 1. Flood-fill exterior near-white background from all four image borders
    luma = np.min(rgb, axis=2)
    chroma = np.max(rgb, axis=2).astype(np.int16) - np.min(rgb, axis=2).astype(np.int16)
    bg_candidate = ((luma > 234) & (chroma < 22)).astype(np.uint8) * 255

    # Connected components from border pixels
    num_labels, labels = cv2.connectedComponents(bg_candidate, connectivity=8)
    border_labels = set(labels[0, :]) | set(labels[h - 1, :]) | set(labels[:, 0]) | set(labels[:, w - 1])
    border_labels.discard(0)
    exterior_bg = np.isin(labels, list(border_labels))

    # 2. Soft anti-aliased edge matting along the 2px silhouette boundary
    dilated_bg = binary_dilation(exterior_bg, iterations=2)
    eroded_fg = binary_erosion(~exterior_bg, iterations=1)
    edge_band = dilated_bg & (~exterior_bg) & (~eroded_fg)

    alpha = (~exterior_bg).astype(np.float32)
    min_c = np.min(rgb, axis=2).astype(np.float32)

    # Estimate coverage along cartoon dark outline against white (255)
    # Dark brown outline base ~ 45..70; coverage = (255 - luma) / (255 - 55)
    edge_alpha = np.clip((252.0 - min_c) / (252.0 - 65.0), 0.0, 1.0)
    alpha[edge_band] = np.minimum(alpha[edge_band], edge_alpha[edge_band])

    # Un-premultiply white background from semi-transparent edge pixels
    clean_rgb = rgb.astype(np.float32)
    semi = (alpha > 0.03) & (alpha < 0.99)
    for c in range(3):
        clean_rgb[semi, c] = np.clip(
            (clean_rgb[semi, c] - (1.0 - alpha[semi]) * 255.0) / np.maximum(alpha[semi], 0.08),
            0.0,
            255.0,
        )

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = np.round(clean_rgb).astype(np.uint8)
    rgba[..., 3] = np.round(alpha * 255.0).astype(np.uint8)
    return rgba


def align_variant_to_base(base_bgr: np.ndarray, var_bgr: np.ndarray, head_roi: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Sub-pixel ECC affine alignment anchored on the upper hair/head silhouette
    (excluding movable eyes and mouth) so expression swaps have <0.1px drift.
    """
    x0, y0, x1, y1 = head_roi
    h, w, _ = base_bgr.shape

    base_gray = cv2.cvtColor(base_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    var_gray = cv2.cvtColor(var_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)

    # Mask out interior movable face box so eyes/mouth differences don't pull the alignment
    rh, rw = base_gray.shape
    mask = np.ones((rh, rw), dtype=np.uint8) * 255
    fx0, fx1 = int(rw * 0.25), int(rw * 0.75)
    fy0, fy1 = int(rh * 0.42), int(rh * 0.96)
    mask[fy0:fy1, fx0:fx1] = 0

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 120, 1e-6)
    try:
        cv2.findTransformECC(
            base_gray,
            var_gray,
            warp_matrix,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            inputMask=mask,
            gaussFiltSize=5,
        )
        # Convert crop-space translation to full-image translation
        # Wait: x_var = R * (x_base - offset) + t + offset
        r_mat = warp_matrix[:, :2]
        t_vec = warp_matrix[:, 2]
        offset = np.array([x0, y0], dtype=np.float32)
        full_t = t_vec + offset - r_mat @ offset
        full_warp = np.zeros((2, 3), dtype=np.float32)
        full_warp[:, :2] = r_mat
        full_warp[:, 2] = full_t
        aligned = cv2.warpAffine(
            var_bgr,
            full_warp,
            (w, h),
            flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        return aligned
    except cv2.error:
        return var_bgr


def find_neck_and_chin_split(rgba: np.ndarray) -> Tuple[int, int, int, np.ndarray]:
    """
    Automatically locate the narrowest horizontal neck waist and trace the bottom
    edge of the dark chin outline above the neck.
    Returns (neck_y, neck_x_left, neck_x_right, head_mask_bool).
    """
    h, w, _ = rgba.shape
    fg = rgba[..., 3] > 25

    # Search for narrowest horizontal foreground span between 20% and 42% of character height
    ys_fg = np.where(np.any(fg, axis=1))[0]
    char_top, char_bot = int(ys_fg.min()), int(ys_fg.max())
    char_h = char_bot - char_top

    search_y0 = char_top + int(char_h * 0.18)
    search_y1 = char_top + int(char_h * 0.36)

    best_y = search_y0
    best_w = w
    neck_xl, neck_xr = w // 2 - 25, w // 2 + 25
    for y in range(search_y0, search_y1):
        xs = np.where(fg[y])[0]
        if len(xs) >= 12:
            span = int(xs[-1] - xs[0])
            if span < best_w:
                best_w = span
                best_y = y
                neck_xl, neck_xr = int(xs[0]), int(xs[-1])

    # Trace dark chin curve inside [neck_xl - 4 .. neck_xr + 4] just above best_y
    rgb = rgba[..., :3].astype(np.float32)
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    chin_y_per_x = np.full(w, best_y - 2, dtype=np.int32)
    scan_top = max(char_top, best_y - int(char_h * 0.06))
    scan_bot = best_y + 2

    margin_x0 = max(0, neck_xl - 4)
    margin_x1 = min(w - 1, neck_xr + 4)

    found_xs = []
    found_ys = []
    for x in range(margin_x0, margin_x1 + 1):
        col = luma[scan_top:scan_bot, x]
        dark_idx = np.where(col < 115)[0]
        if len(dark_idx) > 0:
            # Bottom-most dark pixel of the chin line at column x
            cy = scan_top + int(dark_idx[-1])
            found_xs.append(x)
            found_ys.append(cy)

    if len(found_xs) >= 5:
        # Smooth quadratic curve fit for clean U-shaped chin boundary
        poly = np.poly1d(np.polyfit(found_xs, found_ys, deg=2))
        for x in range(margin_x0, margin_x1 + 1):
            chin_y_per_x[x] = int(np.clip(np.round(poly(x)), scan_top, best_y + 1))

    # Build boolean head region: outside neck columns, anything <= `best_y` belongs to head zone;
    # inside neck columns, anything <= chin_y_per_x[x] belongs to head zone!
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cutoff_y = np.where((xx >= margin_x0) & (xx <= margin_x1), chin_y_per_x[xx], best_y)
    head_zone = yy <= cutoff_y
    return best_y, neck_xl, neck_xr, head_zone


def split_head_and_body_with_neck_extension(
    rgba: np.ndarray, neck_y: int, neck_xl: int, neck_xr: int, head_zone: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce decoupled full-canvas `head_rgba` and `body_rgba` where `body_rgba`
    smoothly extends the shaded neck cylinder upward behind the chin so head pitch/yaw/roll
    never exposes a transparent gap or residual chin line.
    """
    h, w, _ = rgba.shape

    # Soft 1px anti-alias transition along the chin cut for head_rgba
    head_rgba = rgba.copy()
    head_float = head_zone.astype(np.float32)
    head_soft = cv2.GaussianBlur(head_float, (3, 3), 0.6)
    head_rgba[..., 3] = np.round(rgba[..., 3].astype(np.float32) * np.clip(head_soft * 1.15, 0.0, 1.0)).astype(np.uint8)
    head_rgba[head_soft < 0.05] = 0

    # Body strictly excludes the entire upper head_zone (including semi-transparent hair edges!)
    body_rgba = rgba.copy()
    body_rgba[head_zone] = 0

    # Sample reference neck cross-section row safely below the chin line (at neck_y + 5)
    sample_y = min(h - 1, neck_y + 5)
    ref_row = rgba[sample_y, neck_xl : neck_xr + 1].copy()
    ext_height = max(32, int((neck_xr - neck_xl) * 0.78))

    # Overwrite from sample_y - 1 upward so any 1px anti-aliased chin trace on the upper neck is cleanly painted over
    start_fill_y = sample_y - 1
    for step in range(0, ext_height + (start_fill_y - neck_y) + 1):
        y = start_fill_y - step
        if y < 0:
            break
        dy_above_neck = max(0, neck_y - y)
        t = dy_above_neck / float(ext_height)
        # Taper gently inward above chin so extended neck never peeks past cheeks on wide yaw
        taper_px = int(round(t * 4.0))
        cur_xl = neck_xl + taper_px
        cur_xr = neck_xr - taper_px
        cur_w = cur_xr - cur_xl + 1
        if cur_w <= 6:
            break

        resized_row = cv2.resize(
            ref_row[np.newaxis, :, :], (cur_w, 1), interpolation=cv2.INTER_LINEAR
        )[0]
        fade = 1.0 if t < 0.75 else max(0.0, (1.0 - t) / 0.25)
        patch_alpha = np.round(resized_row[:, 3].astype(np.float32) * fade).astype(np.uint8)

        # Subtle natural cast shadow under the chin
        shade = 1.0 - 0.06 * min(1.0, max(0.0, (sample_y - y) / 18.0))
        patch_rgb = np.clip(resized_row[:, :3].astype(np.float32) * shade, 0, 255).astype(np.uint8)

        # Clear any dark chin remnant across neck_xl-3 .. neck_xr+3 and write clean neck cylinder
        body_rgba[y, max(0, neck_xl - 3) : min(w, neck_xr + 4)] = 0
        valid = patch_alpha > 0
        body_rgba[y, cur_xl : cur_xr + 1, :3][valid] = patch_rgb[valid]
        body_rgba[y, cur_xl : cur_xr + 1, 3][valid] = patch_alpha[valid]

    return head_rgba, body_rgba


def compute_interior_face_skin_mask(head_rgba: np.ndarray) -> np.ndarray:
    """
    Build a smooth mask strictly inside the lower/mid face skin (eroded away from
    outer jaw outline and hair silhouette) so facial patches blend seamlessly on skin.
    """
    rgb = head_rgba[..., :3].astype(np.float32)
    alpha = head_rgba[..., 3] > 200

    # Warm skin color detector vs dark brown hair/outline:
    # Base face skin is around RGB=(248, 198, 163); hair is darker brown R<160, G<105
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    skin_like = alpha & (r > 185) & (g > 130) & (r - b > 35)

    # Keep the largest connected component (the face!) to exclude ears
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skin_like.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return alpha.astype(np.float32)

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    face_comp = labels == largest_label

    # Fill internal holes (eyes, eyebrows, mouth inside the face skin!)
    inv = (~face_comp).astype(np.uint8)
    h, w = inv.shape
    num_inv, inv_labels = cv2.connectedComponents(inv, connectivity=8)
    border_inv = set(inv_labels[0, :]) | set(inv_labels[h - 1, :]) | set(inv_labels[:, 0]) | set(inv_labels[:, w - 1])
    filled_face = ~np.isin(inv_labels, list(border_inv))

    # Erode 1px inward from jawline/hair bangs and smooth over 4.5px so outer eyelash tips are 100% covered
    eroded = binary_erosion(filled_face, iterations=1)
    dist = distance_transform_edt(eroded)
    smooth_skin = np.clip(dist / 4.5, 0.0, 1.0).astype(np.float32)
    return smooth_skin


def extract_feathered_patch(
    base_head_rgba: np.ndarray,
    var_full_rgba: np.ndarray,
    head_bbox: Tuple[int, int, int, int],
    face_interior_mask: np.ndarray,
    zone_box_norm: Tuple[float, float, float, float],
    feather_px: float = 11.0,
) -> np.ndarray:
    """
    Extract an aligned, skin-tone matched, elliptically/box-feathered RGBA patch
    in the exact `head_bbox` dimensions (`W_head x H_head`).
    `zone_box_norm` is (nx0, ny0, nx1, ny1) inside `head_bbox`.
    """
    hx0, hy0, hx1, hy1 = head_bbox
    var_crop = var_full_rgba[hy0:hy1, hx0:hx1].copy()
    hh, hw, _ = base_head_rgba.shape

    zx0 = int(round(zone_box_norm[0] * hw))
    zy0 = int(round(zone_box_norm[1] * hh))
    zx1 = int(round(zone_box_norm[2] * hw))
    zy1 = int(round(zone_box_norm[3] * hh))

    # Build rounded pill/superellipse mask inside [zx0..zx1, zy0..zy1]
    yy, xx = np.meshgrid(np.arange(hh, dtype=np.float32), np.arange(hw, dtype=np.float32), indexing="ij")
    cx = 0.5 * (zx0 + zx1)
    cy = 0.5 * (zy0 + zy1)
    rx = max(4.0, 0.5 * (zx1 - zx0))
    ry = max(4.0, 0.5 * (zy1 - zy0))

    # Superellipse (n=3.2) fits rectangular-oval eye pairs and wide mouths cleanly
    dx = np.abs((xx - cx) / rx)
    dy = np.abs((yy - cy) / ry)
    super_dist = (dx ** 3.2 + dy ** 3.2) ** (1.0 / 3.2)

    feather_norm = feather_px / min(rx, ry)
    inner = max(0.35, 1.0 - feather_norm)
    zone_mask = np.clip((1.0 - super_dist) / max(1e-4, 1.0 - inner), 0.0, 1.0)
    # Smooth cosine curve
    zone_mask = 0.5 - 0.5 * np.cos(zone_mask * np.pi)

    # Intersect with interior face mask so jawline & hair bangs are 100% untouched
    final_mask = zone_mask * face_interior_mask

    # Subtle skin-tone offset calibration on the feather ring so there is zero seam
    ring = (final_mask > 0.12) & (final_mask < 0.55)
    base_rgb = base_head_rgba[..., :3].astype(np.float32)
    var_rgb = var_crop[..., :3].astype(np.float32)

    if np.sum(ring) > 30:
        # Exclude dark linework pixels from color calibration
        skin_ring = ring & (np.min(base_rgb, axis=2) > 135) & (np.min(var_rgb, axis=2) > 135)
        if np.sum(skin_ring) > 20:
            delta = np.median(base_rgb[skin_ring] - var_rgb[skin_ring], axis=0)
            delta = np.clip(delta, -12.0, 12.0)
            # Apply gentle correction only to bright skin pixels (preserve dark lashes/mouth interior)
            var_luma = np.min(var_rgb, axis=2)
            skin_weight = np.clip((var_luma - 110.0) / 70.0, 0.0, 1.0)[..., np.newaxis]
            var_rgb = np.clip(var_rgb + delta * skin_weight, 0.0, 255.0)

    patch = np.zeros((hh, hw, 4), dtype=np.uint8)
    patch[..., :3] = np.round(var_rgb).astype(np.uint8)
    patch[..., 3] = np.round(final_mask * 255.0).astype(np.uint8)
    # Zero out RGB where alpha == 0 for crisp compression and clean WebGL bilinear filtering
    # Actually keep neighbor RGB in a 2px dilation border to prevent dark fringe under premultiplied/straight alpha!
    return patch


def synthesize_procedural_patches(
    head_crop_rgba: np.ndarray,
    face_interior_mask: np.ndarray,
    eye_y_norm: float,
    mouth_y_norm: float,
) -> Dict[str, np.ndarray]:
    """
    Fallback procedural patch generator when importing a brand-new single flat cartoon
    image WITHOUT pre-generated AI expression variants. Synthesizes clean cartoon blink,
    happy eyes, smile, laugh, and A/O/E viseme overlays matched to the character's skin tone!
    """
    hh, hw, _ = head_crop_rgba.shape
    rgb = head_crop_rgba[..., :3].astype(np.float32)
    skin_pixels = rgb[(face_interior_mask > 0.7) & (np.min(rgb, axis=2) > 135)]
    if len(skin_pixels) > 20:
        skin_color = np.median(skin_pixels, axis=0).astype(np.uint8).tolist()
    else:
        skin_color = [246, 198, 164]
    line_color = [58, 28, 18]

    def blank_head() -> np.ndarray:
        canvas = head_crop_rgba.copy()
        return canvas

    # Left & right eye centers in head crop coordinates
    lex, rex = int(hw * 0.385), int(hw * 0.615)
    ey = int(hh * eye_y_norm)
    ew = max(12, int(hw * 0.085))
    eh = max(10, int(hh * 0.085))

    mx, my = int(hw * 0.50), int(hh * mouth_y_norm)
    mw = max(12, int(hw * 0.085))

    # 1. Procedural Blink
    c_blink = blank_head()
    cv2.ellipse(c_blink, (lex, ey), (ew + 4, eh + 4), 0, 0, 360, skin_color + [255], -1, cv2.LINE_AA)
    cv2.ellipse(c_blink, (rex, ey), (ew + 4, eh + 4), 0, 0, 360, skin_color + [255], -1, cv2.LINE_AA)
    cv2.ellipse(c_blink, (lex, ey - 1), (ew, int(eh * 0.45)), 0, 15, 165, line_color + [255], 3, cv2.LINE_AA)
    cv2.ellipse(c_blink, (rex, ey - 1), (ew, int(eh * 0.45)), 0, 15, 165, line_color + [255], 3, cv2.LINE_AA)

    # 2. Procedural Happy Eyes (^_^)
    c_happy = blank_head()
    cv2.ellipse(c_happy, (lex, ey), (ew + 4, eh + 4), 0, 0, 360, skin_color + [255], -1, cv2.LINE_AA)
    cv2.ellipse(c_happy, (rex, ey), (ew + 4, eh + 4), 0, 0, 360, skin_color + [255], -1, cv2.LINE_AA)
    cv2.ellipse(c_happy, (lex, ey + 4), (ew, int(eh * 0.55)), 0, 195, 345, line_color + [255], 3, cv2.LINE_AA)
    cv2.ellipse(c_happy, (rex, ey + 4), (ew, int(eh * 0.55)), 0, 195, 345, line_color + [255], 3, cv2.LINE_AA)

    # 3. Procedural Smile, Laugh, A, O, E
    c_smile = blank_head()
    cv2.ellipse(c_smile, (mx, my), (mw + 6, 10), 0, 0, 360, skin_color + [255], -1, cv2.LINE_AA)
    cv2.ellipse(c_smile, (mx, my - 4), (mw + 2, 8), 0, 20, 160, line_color + [255], 2, cv2.LINE_AA)

    c_laugh = blank_head()
    cv2.ellipse(c_laugh, (mx, my + 2), (mw + 6, 13), 0, 0, 360, skin_color + [255], -1, cv2.LINE_AA)
    cv2.ellipse(c_laugh, (mx, my - 2), (mw + 3, 12), 0, 0, 180, [155, 55, 58, 255], -1, cv2.LINE_AA)
    cv2.ellipse(c_laugh, (mx, my + 4), (int(mw * 0.75), 6), 0, 0, 180, [238, 128, 125, 255], -1, cv2.LINE_AA)
    cv2.rectangle(c_laugh, (mx - mw, my - 2), (mx + mw, my + 2), [252, 250, 248, 255], -1)
    cv2.ellipse(c_laugh, (mx, my - 2), (mw + 3, 12), 0, 0, 180, line_color + [255], 2, cv2.LINE_AA)
    cv2.line(c_laugh, (mx - mw - 3, my - 2), (mx + mw + 3, my - 2), line_color + [255], 2, cv2.LINE_AA)

    c_a = blank_head()
    cv2.ellipse(c_a, (mx, my + 1), (mw + 4, 12), 0, 0, 360, skin_color + [255], -1, cv2.LINE_AA)
    cv2.ellipse(c_a, (mx, my + 2), (int(mw * 0.85), 9), 0, 0, 360, [160, 58, 60, 255], -1, cv2.LINE_AA)
    cv2.ellipse(c_a, (mx, my + 5), (int(mw * 0.65), 5), 0, 0, 180, [235, 130, 126, 255], -1, cv2.LINE_AA)
    cv2.ellipse(c_a, (mx, my + 2), (int(mw * 0.85), 9), 0, 0, 360, line_color + [255], 2, cv2.LINE_AA)

    c_o = blank_head()
    cv2.ellipse(c_o, (mx, my + 1), (mw + 4, 12), 0, 0, 360, skin_color + [255], -1, cv2.LINE_AA)
    cv2.ellipse(c_o, (mx, my + 2), (int(mw * 0.5), 8), 0, 0, 360, [150, 52, 55, 255], -1, cv2.LINE_AA)
    cv2.ellipse(c_o, (mx, my + 2), (int(mw * 0.5), 8), 0, 0, 360, line_color + [255], 2, cv2.LINE_AA)

    c_e = blank_head()
    cv2.ellipse(c_e, (mx, my + 1), (mw + 4, 10), 0, 0, 360, skin_color + [255], -1, cv2.LINE_AA)
    cv2.ellipse(c_e, (mx, my + 1), (int(mw * 0.9), 6), 0, 0, 180, [215, 108, 105, 255], -1, cv2.LINE_AA)
    cv2.ellipse(c_e, (mx, my + 1), (int(mw * 0.9), 6), 0, 0, 180, line_color + [255], 2, cv2.LINE_AA)
    cv2.line(c_e, (mx - int(mw * 0.9), my + 1), (mx + int(mw * 0.9), my + 1), line_color + [255], 2, cv2.LINE_AA)

    hx0, hy0 = 0, 0
    dummy_bbox = (0, 0, hw, hh)
    eye_zone = (0.20, max(0.35, eye_y_norm - 0.16), 0.80, min(0.82, eye_y_norm + 0.11))
    mouth_zone = (0.24, max(0.62, mouth_y_norm - 0.10), 0.76, min(0.98, mouth_y_norm + 0.10))

    return {
        "eyes_closed": extract_feathered_patch(head_crop_rgba, c_blink, dummy_bbox, face_interior_mask, eye_zone, 9.0),
        "eyes_happy": extract_feathered_patch(head_crop_rgba, c_happy, dummy_bbox, face_interior_mask, eye_zone, 9.0),
        "mouth_smile": extract_feathered_patch(head_crop_rgba, c_smile, dummy_bbox, face_interior_mask, mouth_zone, 9.0),
        "mouth_laugh": extract_feathered_patch(head_crop_rgba, c_laugh, dummy_bbox, face_interior_mask, mouth_zone, 9.0),
        "mouth_a": extract_feathered_patch(head_crop_rgba, c_a, dummy_bbox, face_interior_mask, mouth_zone, 9.0),
        "mouth_o": extract_feathered_patch(head_crop_rgba, c_o, dummy_bbox, face_interior_mask, mouth_zone, 9.0),
        "mouth_e": extract_feathered_patch(head_crop_rgba, c_e, dummy_bbox, face_interior_mask, mouth_zone, 9.0),
    }


def save_rgba_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGBA").save(path, format="PNG", optimize=True)


def build_bundle(
    base_image_path: Path,
    output_dir: Path,
    character_id: str = "sunny",
    character_name: str = "小阳 (Sunny)",
    variants: Optional[Dict[str, Path]] = None,
) -> Dict:
    """
    Compile a complete `AvatarBundle` directory (`manifest.json`, `head.png`, `body.png`, `patches/*.png`).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    patches_dir = output_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    base_bgr = cv2.imread(str(base_image_path), cv2.IMREAD_COLOR)
    if base_bgr is None:
        raise FileNotFoundError(f"Cannot read base image: {base_image_path}")

    H, W, _ = base_bgr.shape
    Image.fromarray(cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)).save(output_dir / "source.png", format="PNG")

    # 1. Remove white background cleanly
    base_rgba = remove_white_background(base_bgr)
    save_rgba_png(base_rgba, output_dir / "full_rgba.png")

    # Character overall bounding box on the canvas
    fg = base_rgba[..., 3] > 25
    ys_all, xs_all = np.where(fg)
    char_x0, char_y0 = int(xs_all.min()), int(ys_all.min())
    char_x1, char_y1 = int(xs_all.max()), int(ys_all.max())

    # 2. Split head and body at the neck + extend neck behind chin
    neck_y, neck_xl, neck_xr, head_mask = find_neck_and_chin_split(base_rgba)
    head_full_rgba, body_full_rgba = split_head_and_body_with_neck_extension(
        base_rgba, neck_y, neck_xl, neck_xr, head_mask
    )

    # Tight padded crops for head.png and body.png so GPU textures are compact & sharp
    hys, hxs = np.where(head_full_rgba[..., 3] > 5)
    pad = 16
    hx0 = max(0, int(hxs.min()) - pad)
    hy0 = max(0, int(hys.min()) - pad)
    hx1 = min(W, int(hxs.max()) + pad + 1)
    hy1 = min(H, int(hys.max()) + pad + 1)
    head_bbox = (hx0, hy0, hx1, hy1)

    bys, bxs = np.where(body_full_rgba[..., 3] > 5)
    bx0 = max(0, int(bxs.min()) - pad)
    by0 = max(0, int(bys.min()) - pad)
    bx1 = min(W, int(bxs.max()) + pad + 1)
    by1 = min(H, int(bys.max()) + pad + 1)
    body_bbox = (bx0, by0, bx1, by1)

    head_crop = head_full_rgba[hy0:hy1, hx0:hx1]
    body_crop = body_full_rgba[by0:by1, bx0:bx1]
    save_rgba_png(head_crop, output_dir / "head.png")
    save_rgba_png(body_crop, output_dir / "body.png")

    # 3. Compute interior face skin mask on `head_crop`
    face_interior = compute_interior_face_skin_mask(head_crop)
    hh, hw, _ = head_crop.shape

    # Normalized eye & mouth zones within `head_crop`
    # Based on face proportion: eyes/brows sit around y=[0.42..0.71] of head crop, mouth/blush around y=[0.69..0.95]
    eye_zone_norm = (0.14, 0.36, 0.86, 0.735)
    mouth_blush_zone_norm = (0.17, 0.665, 0.83, 0.965)
    mouth_tight_zone_norm = (0.26, 0.71, 0.74, 0.96)

    patch_files: Dict[str, str] = {}

    if variants:
        aligned_rgba_cache: Dict[str, np.ndarray] = {}
        for key, vpath in variants.items():
            vbgr = cv2.imread(str(vpath), cv2.IMREAD_COLOR)
            if vbgr is None:
                continue
            if vbgr.shape[:2] != (H, W):
                vbgr = cv2.resize(vbgr, (W, H), interpolation=cv2.INTER_LANCZOS4)
            aligned_bgr = align_variant_to_base(base_bgr, vbgr, head_bbox)
            aligned_rgba_cache[key] = remove_white_background(aligned_bgr)

        # Extract eye patches
        if "eyes_closed" in aligned_rgba_cache:
            p = extract_feathered_patch(
                head_crop, aligned_rgba_cache["eyes_closed"], head_bbox, face_interior, eye_zone_norm, 10.0
            )
            save_rgba_png(p, patches_dir / "eyes_closed.png")
            patch_files["eyes_closed"] = "patches/eyes_closed.png"

        if "laugh_joy" in aligned_rgba_cache:
            # Extract happy crescent eyes (^_^) from upper face of laugh_joy
            p_eh = extract_feathered_patch(
                head_crop, aligned_rgba_cache["laugh_joy"], head_bbox, face_interior, eye_zone_norm, 10.0
            )
            save_rgba_png(p_eh, patches_dir / "eyes_happy.png")
            patch_files["eyes_happy"] = "patches/eyes_happy.png"

            # Extract joyful wide laughing mouth + rosy cheeks from lower face of laugh_joy
            p_ml = extract_feathered_patch(
                head_crop, aligned_rgba_cache["laugh_joy"], head_bbox, face_interior, mouth_blush_zone_norm, 11.0
            )
            save_rgba_png(p_ml, patches_dir / "mouth_laugh.png")
            patch_files["mouth_laugh"] = "patches/mouth_laugh.png"

        if "smile_warm" in aligned_rgba_cache:
            p_ms = extract_feathered_patch(
                head_crop, aligned_rgba_cache["smile_warm"], head_bbox, face_interior, mouth_blush_zone_norm, 11.0
            )
            save_rgba_png(p_ms, patches_dir / "mouth_smile.png")
            patch_files["mouth_smile"] = "patches/mouth_smile.png"

        for viseme_key, out_name in [("talk_a", "mouth_a"), ("talk_e", "mouth_e"), ("talk_o", "mouth_o")]:
            if viseme_key in aligned_rgba_cache:
                p_v = extract_feathered_patch(
                    head_crop, aligned_rgba_cache[viseme_key], head_bbox, face_interior, mouth_tight_zone_norm, 9.5
                )
                save_rgba_png(p_v, patches_dir / f"{out_name}.png")
                patch_files[out_name] = f"patches/{out_name}.png"

    # Fill any missing patches procedurally (allows single-image custom uploads to work seamlessly!)
    proc_patches = synthesize_procedural_patches(head_crop, face_interior, eye_y_norm=0.585, mouth_y_norm=0.815)
    for k, arr in proc_patches.items():
        if k not in patch_files:
            save_rgba_png(arr, patches_dir / f"{k}.png")
            patch_files[k] = f"patches/{k}.png"

    # 4. Build standardized normalized rig manifest (`manifest.json`)
    neck_cx = 0.5 * (neck_xl + neck_xr)
    neck_pivot_canvas = [round(neck_cx, 1), round(float(neck_y - 4), 1)]

    # Anatomical landmarks in canvas pixel coordinates (also editable in Web Rig Calibrator!)
    char_h = float(char_y1 - char_y0)
    face_cx = neck_cx
    eye_y_canvas = hy0 + hh * 0.585
    nose_y_canvas = hy0 + hh * 0.688
    mouth_y_canvas = hy0 + hh * 0.805

    manifest = {
        "version": "1.0.0",
        "id": character_id,
        "name": character_name,
        "canvasSize": {"width": W, "height": H},
        "characterBounds": {
            "x": char_x0,
            "y": char_y0,
            "width": char_x1 - char_x0,
            "height": char_y1 - char_y0,
        },
        "layers": {
            "head": {
                "texture": "head.png",
                "x": hx0,
                "y": hy0,
                "width": hx1 - hx0,
                "height": hy1 - hy0,
                "meshCols": 16,
                "meshRows": 16,
            },
            "body": {
                "texture": "body.png",
                "x": bx0,
                "y": by0,
                "width": bx1 - bx0,
                "height": by1 - by0,
                "meshCols": 12,
                "meshRows": 20,
            },
        },
        "patches": patch_files,
        "landmarks": {
            "headCenter": [round(face_cx, 1), round(hy0 + hh * 0.54, 1)],
            "leftEye": [round(face_cx - (hx1 - hx0) * 0.132, 1), round(eye_y_canvas, 1)],
            "rightEye": [round(face_cx + (hx1 - hx0) * 0.132, 1), round(eye_y_canvas, 1)],
            "nose": [round(face_cx, 1), round(nose_y_canvas, 1)],
            "mouth": [round(face_cx, 1), round(mouth_y_canvas, 1)],
            "neckPivot": neck_pivot_canvas,
            "chest": [round(face_cx, 1), round(char_y0 + char_h * 0.36, 1)],
            "hips": [round(face_cx, 1), round(char_y0 + char_h * 0.57, 1)],
            "knees": [round(face_cx, 1), round(char_y0 + char_h * 0.77, 1)],
            "feetGround": [round(face_cx, 1), round(float(char_y1 - 6), 1)],
        },
    }

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile AI Cartoon Image + Variants into an Interactive AvatarBundle")
    parser.add_argument("--base", required=True, help="Path to base full-body cartoon image")
    parser.add_argument("--out", required=True, help="Output AvatarBundle directory")
    parser.add_argument("--id", default="sunny", help="Avatar ID")
    parser.add_argument("--name", default="小阳 (Sunny)", help="Display name")
    parser.add_argument("--eyes-closed", default=None)
    parser.add_argument("--smile", default=None)
    parser.add_argument("--laugh", default=None)
    parser.add_argument("--talk-a", default=None)
    parser.add_argument("--talk-e", default=None)
    parser.add_argument("--talk-o", default=None)
    args = parser.parse_args()

    variants = {}
    for k, val in [
        ("eyes_closed", args.eyes_closed),
        ("smile_warm", args.smile),
        ("laugh_joy", args.laugh),
        ("talk_a", args.talk_a),
        ("talk_e", args.talk_e),
        ("talk_o", args.talk_o),
    ]:
        if val and os.path.exists(val):
            variants[k] = Path(val)

    manifest = build_bundle(
        base_image_path=Path(args.base),
        output_dir=Path(args.out),
        character_id=args.id,
        character_name=args.name,
        variants=variants,
    )
    print(f"[OK] Built AvatarBundle '{manifest['id']}' at {args.out}")
    print(f"     Head layer: {manifest['layers']['head']}")
    print(f"     Body layer: {manifest['layers']['body']}")
    print(f"     Patches   : {list(manifest['patches'].keys())}")


if __name__ == "__main__":
    main()
