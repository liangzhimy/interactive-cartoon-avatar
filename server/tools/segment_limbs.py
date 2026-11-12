#!/usr/bin/env python3
"""
Capsule-Joint Limb Segmentation for Sunny (`public/avatars/sunny/body.png`).
Splits `body.png` into 5 clean RGBA skeletal layers (`torso`, `left_arm`, `right_arm`,
`left_leg`, `right_leg`) with rounded shoulder ball caps and upper-thigh hip caps so
large-angle 2D Dual-Quaternion limb animations (`wave`, `cheer`, `walk`, `kick`, `jump`)
have zero gaps, zero armpit bulges, and zero shorts bleed onto hands.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter


def save_rgba_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Dilate RGB by 2px into alpha==0 boundary to prevent dark bilinear fringes in WebGL
    out = arr.copy()
    mask = out[..., 3] > 0
    if np.any(mask) and np.any(~mask):
        _, indices = distance_transform_edt(~mask, return_indices=True)
        ring = (~mask) & (distance_transform_edt(~mask) <= 2.5)
        out[..., :3][ring] = out[..., :3][indices[0][ring], indices[1][ring]]
    Image.fromarray(out, mode="RGBA").save(path, format="PNG", optimize=True)


def crop_layer(
    full_body_layer: np.ndarray, bx0: int, by0: int, pad: int = 8, cols: int = 10, rows: int = 22, tex_name: str = ""
) -> Tuple[np.ndarray, Dict]:
    H, W, _ = full_body_layer.shape
    ys, xs = np.where(full_body_layer[..., 3] > 3)
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(W, int(xs.max()) + pad + 1)
    y1 = min(H, int(ys.max()) + pad + 1)
    cropped = full_body_layer[y0:y1, x0:x1].copy()
    spec = {
        "texture": tex_name,
        "x": bx0 + x0,
        "y": by0 + y0,
        "width": x1 - x0,
        "height": y1 - y0,
        "meshCols": cols,
        "meshRows": rows,
    }
    return cropped, spec


def keep_largest_component(rgba: np.ndarray, alpha_thresh: int = 12) -> np.ndarray:
    out = rgba.copy()
    mask = (out[..., 3] > alpha_thresh).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return out
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    drop = (labels != largest) & (labels != 0)
    out[drop] = 0
    return out


def segment_sunny_limbs(bundle_dir: Path) -> Dict:
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    body_spec = manifest["layers"]["body"]
    bx0, by0 = int(body_spec["x"]), int(body_spec["y"])
    body = np.array(Image.open(bundle_dir / body_spec["texture"]).convert("RGBA"))
    H, W, _ = body.shape  # 911 x 400

    # Clean any tiny enclosed white/grey background pockets above sneakers (y < 750),
    # e.g., trapped between puffed sleeve cuff, waistband, and shorts top!
    ys_grid = np.arange(H)[:, None]
    c_min = np.min(body[..., :3], axis=2)
    c_max = np.max(body[..., :3], axis=2)
    trapped_bg = (ys_grid < 750) & (c_min > 175) & ((c_max.astype(np.int32) - c_min.astype(np.int32)) < 32)
    if np.any(trapped_bg):
        body[trapped_bg] = 0
        # Also clean 1px halo around trapped pocket
        dil = cv2.dilate(trapped_bg.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        halo = dil & (ys_grid < 750) & (c_min > 145) & ((c_max.astype(np.int32) - c_min.astype(np.int32)) < 35)
        body[halo] = 0

    rgb = body[..., :3].astype(np.float32)
    alpha = body[..., 3].astype(np.float32)
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    coral = body[145, 155, :3].astype(np.float32)  # [255, 113, 77]

    # 1. Trace left & right sleeve-to-torso dark seam (`y = 118..349`)
    guide_pts = np.array(
        [
            [84.0, 124.0],
            [99.0, 150.0],
            [112.0, 175.0],
            [111.0, 225.0],
            [102.0, 275.0],
            [96.5, 305.0],
            [99.0, 330.0],
            [101.5, 349.0],
        ],
        dtype=np.float32,
    )
    left_seam = np.zeros(H, dtype=np.float32)
    right_seam = np.zeros(H, dtype=np.float32)

    for y in range(118, 350):
        gx_l = float(np.interp(y, guide_pts[:, 1], guide_pts[:, 0]))
        s0 = max(60, int(round(gx_l)) - 8)
        s1 = min(130, int(round(gx_l)) + 9)
        seg_l = luma[y, s0:s1] + np.where(alpha[y, s0:s1] > 80, 0.0, 500.0)
        left_seam[y] = s0 + float(np.argmin(seg_l))

        gx_r = (W - 1.0) - gx_l
        r0 = max(268, int(round(gx_r)) - 8)
        r1 = min(340, int(round(gx_r)) + 9)
        seg_r = luma[y, r0:r1] + np.where(alpha[y, r0:r1] > 80, 0.0, 500.0)
        right_seam[y] = r0 + float(np.argmin(seg_r))

    left_seam[118:350] = gaussian_filter(left_seam[118:350], sigma=1.4)
    right_seam[118:350] = gaussian_filter(right_seam[118:350], sigma=1.4)

    # 2. Trace empty alpha valley between hands and flared shorts (`y = 350..505`)
    for y in range(350, 506):
        # Left valley between hand (x ~ 20..76) and flared shorts (x ~ 79..199)
        valley_l = alpha[y, 66:106]
        zeros_l = np.where(valley_l < 15)[0]
        if len(zeros_l) > 0:
            left_seam[y] = 66.0 + float(np.median(zeros_l))
        else:
            left_seam[y] = 66.0 + float(np.argmin(valley_l))

        # Right valley between flared shorts (x ~ 200..320) and right hand (x ~ 323..380)
        valley_r = alpha[y, 294:334]
        zeros_r = np.where(valley_r < 15)[0]
        if len(zeros_r) > 0:
            right_seam[y] = 294.0 + float(np.median(zeros_r))
        else:
            right_seam[y] = 294.0 + float(np.argmin(valley_r))

    # 3. Trace bottom hem of grey-blue shorts (`y = 470..499`, `x = 78..322`)
    from scipy.ndimage import median_filter

    shorts_bot = np.full(W, 488.0, dtype=np.float32)
    found_sx, found_sy = [], []
    for x in range(78, 323):
        col_r = rgb[470:499, x, 0]
        col_b = rgb[470:499, x, 2]
        col_a = alpha[470:499, x]
        is_shorts = (col_a > 120) & ((col_r - col_b) < 30)
        idx = np.where(is_shorts)[0]
        if len(idx) > 0:
            found_sx.append(x)
            found_sy.append(470.0 + float(idx[-1]))
    if len(found_sx) > 10:
        shorts_bot = np.interp(np.arange(W), found_sx, found_sy).astype(np.float32)
        shorts_bot = median_filter(shorts_bot, size=11)
        shorts_bot = gaussian_filter(shorts_bot, sigma=1.4)

    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")

    # 4. Build Torso Layer (Neck + Sweater + Intact Flared Shorts)
    torso = np.zeros_like(body)
    torso_mask = np.zeros((H, W), dtype=np.float32)
    # Above y=126: full neck & upper shoulder slope belongs to torso
    torso_mask[yy < 126] = 1.0
    # Sweater torso y=126..349: inside [left_seam - 1.2, right_seam + 1.2] with soft 1.5px edge
    for y in range(126, 350):
        xl = left_seam[y] - 1.2
        xr = right_seam[y] + 1.2
        w_l = np.clip((xx[y] - (xl - 1.5)) / 1.8, 0.0, 1.0)
        w_r = np.clip(((xr + 1.5) - xx[y]) / 1.8, 0.0, 1.0)
        torso_mask[y] = w_l * w_r
    # Shorts region y=350..505: inside alpha valleys [left_seam, right_seam] and above shorts_bot + 1.0
    for y in range(350, 506):
        in_x = (xx[y] > left_seam[y]) & (xx[y] < right_seam[y])
        above_hem = np.clip((shorts_bot + 1.2) - y, 0.0, 1.0)
        torso_mask[y] = np.where(in_x, above_hem, 0.0)

    torso[..., :3] = body[..., :3]
    torso[..., 3] = np.round(alpha * torso_mask).astype(np.uint8)
    torso = keep_largest_component(torso, alpha_thresh=10)

    # Paint clean coral fill over the upper diagonal sleeve-seam remnant inside torso (`y = 128..164`)
    # while preserving the dark outer shoulder silhouette (`dist_to_empty >= 3.5`)!
    torso_fg = torso[..., 3] > 200
    dist_torso_in = distance_transform_edt(torso_fg)
    for y in range(128, 166):
        for side_seam in (left_seam[y], right_seam[y]):
            band = (
                (np.abs(xx[y] - side_seam) <= 5.5)
                & (dist_torso_in[y] >= 3.5)
                & (luma[y] < 135)
            )
            torso[y, band, :3] = np.round(coral).astype(np.uint8)

    # 5. Build Left & Right Arm Layers with Strict Circular Capsule Ball Shoulder Cap!
    # Shoulder pivots in `body.png` local coords:
    # Left shoulder: (93.0, 152.0) -> world (341.0, 417.0)
    # Right shoulder: (306.0, 152.0) -> world (554.0, 417.0)
    sh_rad = 21.0

    def build_arm(is_left: bool) -> np.ndarray:
        arm = np.zeros_like(body)
        sx, sy = (93.0, 152.0) if is_left else (306.0, 152.0)
        dist_sh = np.hypot(xx - sx, yy - sy)

        # Base arm extraction from body (`y = 128..505`)
        raw_mask = np.zeros((H, W), dtype=np.float32)
        for y in range(128, 505):
            if y < 350:
                # Smoothly blend inner boundary from circular cap near shoulder (`y <= sy + 20`)
                # to the natural sleeve seam below `sy + 20` so there is zero notch at underarm!
                dy = y - sy
                cap_dx = np.sqrt(max(0.0, sh_rad * sh_rad - min(sh_rad, abs(dy)) ** 2))
                if is_left:
                    cap_in_x = sx + cap_dx
                    seam_x = left_seam[y]
                    blend = np.clip((y - (sy + 4.0)) / 18.0, 0.0, 1.0)
                    bound_x = cap_in_x * (1.0 - blend) + (seam_x - 0.2) * blend
                    in_arm = np.clip(((bound_x + 0.6) - xx[y]) / 1.4, 0.0, 1.0)
                else:
                    cap_in_x = sx - cap_dx
                    seam_x = right_seam[y]
                    blend = np.clip((y - (sy + 4.0)) / 18.0, 0.0, 1.0)
                    bound_x = cap_in_x * (1.0 - blend) + (seam_x + 0.2) * blend
                    in_arm = np.clip((xx[y] - (bound_x - 0.6)) / 1.4, 0.0, 1.0)
                raw_mask[y] = in_arm
            else:
                if is_left:
                    raw_mask[y] = (xx[y] < left_seam[y]).astype(np.float32)
                else:
                    raw_mask[y] = (xx[y] > right_seam[y]).astype(np.float32)

        # Strict circular hemisphere above `sy + 5` so rotating +-130 deg NEVER exposes an armpit corner!
        above_cap = yy <= (sy + 5.0)
        cap_feather = np.clip((sh_rad + 1.0 - dist_sh) / 2.2, 0.0, 1.0)
        raw_mask = np.where(above_cap, np.minimum(raw_mask, cap_feather), raw_mask)

        arm[..., :3] = body[..., :3]
        arm[..., 3] = np.round(alpha * raw_mask).astype(np.uint8)

        # Fill rounded coral ball cap inside `dist_sh <= sh_rad` (`y <= sy + 18`),
        # preserving original outer dark outline on the outer sleeve edge!
        body_outer_dist = distance_transform_edt(alpha > 180)
        in_ball = (dist_sh <= sh_rad) & (yy <= sy + 18.0)
        # Pixels inside ball that were empty or part of the internal seam (not outer silhouette rim)
        is_outer_rim = (body_outer_dist > 0) & (body_outer_dist < 3.4) & (
            (xx < sx - 4.0) if is_left else (xx > sx + 4.0)
        )
        fill_coral = in_ball & (~is_outer_rim) & ((arm[..., 3] < 245) | (luma < 145))
        arm[fill_coral, :3] = np.round(coral).astype(np.uint8)
        ball_alpha = np.round(np.clip((sh_rad + 0.8 - dist_sh) / 2.2, 0.0, 1.0) * 255.0).astype(np.uint8)
        arm[fill_coral, 3] = np.maximum(arm[fill_coral, 3], ball_alpha[fill_coral])

        return keep_largest_component(arm, alpha_thresh=10)

    left_arm = build_arm(is_left=True)
    right_arm = build_arm(is_left=False)

    # 6. Build Left & Right Leg Layers (Behind Shorts, with Rounded Upper Thigh Hip Cap!)
    def build_leg(is_left: bool) -> np.ndarray:
        leg = np.zeros_like(body)
        for y in range(478, H):
            if is_left:
                side_ok = xx[y] < 200.0
            else:
                side_ok = xx[y] >= 200.0
            below_shorts = y >= (shorts_bot + 2.0)
            valid = side_ok & below_shorts
            leg[y, valid] = body[y, valid]

        leg = keep_largest_component(leg, alpha_thresh=12)

        # Sample clean bare thigh cross-section at `y = 508` and extend upward to `y = 424` (`hip_y = 450 - 26`)
        # so kicking / marching legs NEVER reveal a gap beneath the shorts hem!
        ref_y = 508
        xs_ref = np.where(leg[ref_y, :, 3] > 180)[0]
        if len(xs_ref) >= 12:
            tx0, tx1 = int(xs_ref[0]), int(xs_ref[-1])
            ref_row = leg[ref_y, tx0 : tx1 + 1].copy()
            top_cap_y = 424
            span = float(ref_y - top_cap_y)
            for y in range(ref_y - 1, top_cap_y - 1, -1):
                t = (ref_y - y) / span  # 0 at ref_y, 1 at top_cap_y
                # Gentle rounded dome taper near top of hip socket
                dome = np.sqrt(max(0.05, 1.0 - (max(0.0, (t - 0.55) / 0.45)) ** 2))
                cur_w = max(8, int(round((tx1 - tx0 + 1) * (0.94 + 0.06 * (1.0 - t)) * dome)))
                cx = 0.5 * (tx0 + tx1)
                c0 = max(0, int(round(cx - cur_w * 0.5)))
                c1 = min(W, c0 + cur_w)
                resized = cv2.resize(ref_row[np.newaxis, :, :], (c1 - c0, 1), interpolation=cv2.INTER_LINEAR)[0]
                fade = 1.0 if t < 0.82 else max(0.0, (1.0 - t) / 0.18)
                new_a = np.round(resized[:, 3].astype(np.float32) * fade).astype(np.uint8)
                # Subtle cast shadow right under shorts hem (y < 496) that looks super natural when kicking
                shade = 0.96 if y < 492 else 1.0
                new_rgb = np.clip(resized[:, :3].astype(np.float32) * shade, 0, 255).astype(np.uint8)
                upd = new_a > leg[y, c0:c1, 3]
                leg[y, c0:c1, :3][upd] = new_rgb[upd]
                leg[y, c0:c1, 3][upd] = new_a[upd]

        return keep_largest_component(leg, alpha_thresh=10)

    left_leg = build_leg(is_left=True)
    right_leg = build_leg(is_left=False)

    # Save cropped PNGs and update manifest
    torso_crop, torso_spec = crop_layer(torso, bx0, by0, pad=8, cols=12, rows=16, tex_name="torso.png")
    l_arm_crop, l_arm_spec = crop_layer(left_arm, bx0, by0, pad=8, cols=10, rows=22, tex_name="left_arm.png")
    r_arm_crop, r_arm_spec = crop_layer(right_arm, bx0, by0, pad=8, cols=10, rows=22, tex_name="right_arm.png")
    l_leg_crop, l_leg_spec = crop_layer(left_leg, bx0, by0, pad=8, cols=10, rows=22, tex_name="left_leg.png")
    r_leg_crop, r_leg_spec = crop_layer(right_leg, bx0, by0, pad=8, cols=10, rows=22, tex_name="right_leg.png")

    save_rgba_png(torso_crop, bundle_dir / "torso.png")
    save_rgba_png(l_arm_crop, bundle_dir / "left_arm.png")
    save_rgba_png(r_arm_crop, bundle_dir / "right_arm.png")
    save_rgba_png(l_leg_crop, bundle_dir / "left_leg.png")
    save_rgba_png(r_leg_crop, bundle_dir / "right_leg.png")

    manifest["version"] = "1.1.0"
    manifest["layers"]["torso"] = torso_spec
    manifest["layers"]["leftArm"] = l_arm_spec
    manifest["layers"]["rightArm"] = r_arm_spec
    manifest["layers"]["leftLeg"] = l_leg_spec
    manifest["layers"]["rightLeg"] = r_leg_spec

    # Add skeletal 4-limb joints in canvas world coordinates
    lm = manifest["landmarks"]
    lm["leftShoulder"] = [round(bx0 + 93.0, 1), round(by0 + 152.0, 1)]
    lm["leftElbow"] = [round(bx0 + 78.0, 1), round(by0 + 258.0, 1)]
    lm["leftWrist"] = [round(bx0 + 58.0, 1), round(by0 + 392.0, 1)]
    lm["rightShoulder"] = [round(bx0 + 306.0, 1), round(by0 + 152.0, 1)]
    lm["rightElbow"] = [round(bx0 + 321.0, 1), round(by0 + 258.0, 1)]
    lm["rightWrist"] = [round(bx0 + 341.0, 1), round(by0 + 392.0, 1)]
    lm["leftHip"] = [round(bx0 + 146.0, 1), round(by0 + 450.0, 1)]
    lm["leftKnee"] = [round(bx0 + 145.0, 1), round(by0 + 635.0, 1)]
    lm["leftAnkle"] = [round(bx0 + 135.0, 1), round(by0 + 785.0, 1)]
    lm["rightHip"] = [round(bx0 + 253.0, 1), round(by0 + 450.0, 1)]
    lm["rightKnee"] = [round(bx0 + 254.0, 1), round(by0 + 635.0, 1)]
    lm["rightAnkle"] = [round(bx0 + 264.0, 1), round(by0 + 785.0, 1)]

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    sunny_dir = Path("/home/antigravity/interactive-cartoon-avatar/public/avatars/sunny")
    out_manifest = segment_sunny_limbs(sunny_dir)
    print("[OK] Segmented Sunny 5 skeletal layers:")
    for k in ("torso", "leftArm", "rightArm", "leftLeg", "rightLeg"):
        print(f"  - {k:9s}: {out_manifest['layers'][k]}")
