#!/usr/bin/env python3
"""
Stereo disparity pipeline following specified steps with configurable parameters.

Steps:
 1) Load data via utils/image_loader.py
 2) Optional rectification verification (and optional warp)
 3) Optional denoising (Gaussian/Median)
 4) Optional CLAHE
 5) Stereo matching via SGBM
 6) Post-processing: Left-Right consistency check
 7) Post-processing: Hole filling (dilate + neighborhood fill)
 8) Post-processing: Disparity smoothing (guided/bilateral)
 9) Post-processing: Final median denoise
10) Compute PSNR vs ground-truth with valid mask

Outputs:
 - results/disp_pred_raw.png           : raw disparity visualization
 - results/disp_pred_refined.png       : refined disparity visualization
 - results/psnr_pipeline.txt           : PSNR report
"""
import os
import math
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

from utils.image_loader import ImageLoader
from config import (
    DATA_DIR,
    ENABLE_RECTIFICATION_CHECK,
    ENABLE_RECTIFICATION_WARP,
    ENABLE_DENOISE,
    DENOISE_METHOD,
    GAUSSIAN_KERNEL,
    GAUSSIAN_SIGMA,
    MEDIAN_KERNEL,
    ENABLE_CLAHE,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    SGBM_MIN_DISPARITY,
    SGBM_NUM_DISPARITIES,
    SGBM_BLOCK_SIZE,
    SGBM_P1,
    SGBM_P2,
    SGBM_DISP12_MAX_DIFF,
    SGBM_UNIQUENESS_RATIO,
    SGBM_SPECKLE_WINDOW_SIZE,
    SGBM_SPECKLE_RANGE,
    SGBM_PREFILTER_CAP,
    SGBM_MODE,
    ENABLE_LR_CHECK,
    LR_MAX_DIFF,
    ENABLE_HOLE_FILL,
    DILATE_KERNEL,
    DILATE_ITERATIONS,
    FILL_STRATEGY,
    FILL_WINDOW,
    ENABLE_SMOOTHING,
    USE_GUIDED_FILTER,
    GUIDED_RADIUS,
    GUIDED_EPS,
    USE_BILATERAL_FILTER,
    BILATERAL_D,
    BILATERAL_SIGMA_COLOR,
    BILATERAL_SIGMA_SPACE,
    ENABLE_FINAL_MEDIAN,
    FINAL_MEDIAN_KERNEL,
    INVALID_DISPARITY_VALUE,
    PSNR_ALIGN_TO_GT,
    PSNR_ALIGN_PERC,
)


def ensure_results_dir():
    os.makedirs("results", exist_ok=True)


def to_gray_u8(img):
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def verify_horizontal_epipolar(img_left, img_right):
    """Verify rectification by checking matched points' y-coordinates are similar."""
    sift = cv2.SIFT_create()
    kp_l, des_l = sift.detectAndCompute(img_left, None)
    kp_r, des_r = sift.detectAndCompute(img_right, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw = bf.knnMatch(des_l, des_r, k=2)
    good = []
    for m, n in raw:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    pts_l = np.float32([kp_l[m.queryIdx].pt for m in good])
    pts_r = np.float32([kp_r[m.trainIdx].pt for m in good])
    if len(pts_l) == 0:
        return False, 0.0
    dy = np.abs(pts_l[:, 1] - pts_r[:, 1])
    mean_dy = float(np.mean(dy))
    return mean_dy < 1.0, mean_dy  # threshold 1px


def maybe_rectify(img_left, img_right):
    if not ENABLE_RECTIFICATION_WARP:
        return img_left, img_right
    # Estimate F and compute uncalibrated rectification
    sift = cv2.SIFT_create()
    kp_l, des_l = sift.detectAndCompute(img_left, None)
    kp_r, des_r = sift.detectAndCompute(img_right, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw = bf.knnMatch(des_l, des_r, k=2)
    good = []
    for m, n in raw:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    pts_l = np.float32([kp_l[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_r = np.float32([kp_r[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    h, w = img_left.shape
    F, mask = cv2.findFundamentalMat(pts_l, pts_r, cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.99)
    retval, H1, H2 = cv2.stereoRectifyUncalibrated(pts_l[mask.ravel() == 1], pts_r[mask.ravel() == 1], F, (w, h))
    left_rect = cv2.warpPerspective(img_left, H1, (w, h))
    right_rect = cv2.warpPerspective(img_right, H2, (w, h))
    return left_rect, right_rect


def denoise(img):
    if not ENABLE_DENOISE:
        return img
    if DENOISE_METHOD == "gaussian":
        return cv2.GaussianBlur(img, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)
    else:
        return cv2.medianBlur(img, MEDIAN_KERNEL)


def apply_clahe(img):
    if not ENABLE_CLAHE:
        return img
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    return clahe.apply(img)


def compute_sgbm(left, right):
    num_disp = SGBM_NUM_DISPARITIES if SGBM_NUM_DISPARITIES % 16 == 0 else (SGBM_NUM_DISPARITIES // 16 + 1) * 16
    mode = cv2.StereoSGBM_MODE_SGBM_3WAY if SGBM_MODE == "SGBM_3WAY" else cv2.StereoSGBM_MODE_SGBM
    sgbm = cv2.StereoSGBM_create(
        minDisparity=SGBM_MIN_DISPARITY,
        numDisparities=num_disp,
        blockSize=SGBM_BLOCK_SIZE,
        P1=SGBM_P1,
        P2=SGBM_P2,
        disp12MaxDiff=SGBM_DISP12_MAX_DIFF,
        uniquenessRatio=SGBM_UNIQUENESS_RATIO,
        speckleWindowSize=SGBM_SPECKLE_WINDOW_SIZE,
        speckleRange=SGBM_SPECKLE_RANGE,
        preFilterCap=SGBM_PREFILTER_CAP,
        mode=mode,
    )
    disp_l = sgbm.compute(left, right).astype(np.float32) / 16.0
    # Right matcher for LR-check guidance
    try:
        right_matcher = cv2.ximgproc.createRightMatcher(sgbm)
        disp_r = right_matcher.compute(right, left).astype(np.float32) / 16.0
    except Exception:
        disp_r = sgbm.compute(right, left).astype(np.float32) / 16.0
    disp_l[disp_l < 0] = 0
    disp_r[disp_r < 0] = 0
    return disp_l, disp_r


def left_right_consistency(disp_l, disp_r):
    if not ENABLE_LR_CHECK:
        return np.ones_like(disp_l, dtype=np.uint8)
    h, w = disp_l.shape
    mask = np.ones((h, w), dtype=np.uint8)
    xs = np.arange(w)
    for y in range(h):
        d = disp_l[y]
        x_prime = (xs - d).astype(np.int32)
        oob = (x_prime < 0) | (x_prime >= w)
        mask[y, oob] = 0
        inb = ~oob
        if np.any(inb):
            diff = np.abs(d[inb] - disp_r[y, x_prime[inb]])
            mask[y, inb] = (diff <= LR_MAX_DIFF).astype(np.uint8)
    return mask


def hole_fill(disp, valid_mask):
    if not ENABLE_HOLE_FILL:
        return disp
    # Dilate valid regions to shrink holes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, DILATE_KERNEL)
    valid_dilated = cv2.dilate(valid_mask, kernel, iterations=DILATE_ITERATIONS)
    # Fill invalid pixels using neighborhood stats
    filled = disp.copy()
    half = FILL_WINDOW // 2
    h, w = disp.shape
    for y in range(h):
        for x in range(w):
            # Fill pixels that were invalid in the original mask (source of holes)
            if valid_mask[y, x] == 0:
                y0, y1 = max(0, y - half), min(h, y + half + 1)
                x0, x1 = max(0, x - half), min(w, x + half + 1)
                neigh = disp[y0:y1, x0:x1]
                # Prefer original valid pixels; if none exist, allow dilated as donors
                neigh_mask = valid_mask[y0:y1, x0:x1]
                if not np.any(neigh_mask):
                    neigh_mask = valid_dilated[y0:y1, x0:x1]
                vals = neigh[neigh_mask == 1]
                if vals.size > 0:
                    if FILL_STRATEGY == "mean":
                        filled[y, x] = float(np.mean(vals))
                    else:
                        filled[y, x] = float(np.median(vals))
                else:
                    filled[y, x] = disp[y, x]
    return filled


def smooth_disparity(disp, guide):
    if not ENABLE_SMOOTHING:
        return disp
    out = disp.copy()
    # Guided filter (requires ximgproc)
    if USE_GUIDED_FILTER:
        try:
            # guidedFilter expects guide and src in CV_8U or CV_32F
            guide_u8 = guide if guide.dtype == np.uint8 else cv2.normalize(guide, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            disp_32 = out.astype(np.float32)
            out = cv2.ximgproc.guidedFilter(guide_u8, disp_32, GUIDED_RADIUS, GUIDED_EPS)
        except Exception:
            pass
    # Bilateral filter (edge-preserving)
    if USE_BILATERAL_FILTER:
        # Apply bilateral directly on float map to avoid rescaling artifacts
        out = cv2.bilateralFilter(out.astype(np.float32), BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)
    return out


def final_median(disp):
    if not ENABLE_FINAL_MEDIAN:
        return disp
    return cv2.medianBlur(disp.astype(np.float32), FINAL_MEDIAN_KERNEL)


def normalize_to_uint8(img):
    minv = float(np.min(img))
    maxv = float(np.max(img))
    if maxv - minv < 1e-6:
        return np.zeros(img.shape, dtype=np.uint8)
    return ((img - minv) / (maxv - minv) * 255.0).astype(np.uint8)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def compute_psnr_with_mask(gt, pred):
    # Align sizes
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    # Build valid mask: exclude INVALID_DISPARITY_VALUE in either map
    valid = (gt != INVALID_DISPARITY_VALUE) & (pred != INVALID_DISPARITY_VALUE)
    if not np.any(valid):
        return 0.0
    gt_valid = gt[valid]
    pred_valid = pred[valid]
    mse = float(np.mean((gt_valid.astype(np.float32) - pred_valid.astype(np.float32)) ** 2))
    if mse <= 1e-12:
        return 100.0
    psnr = 20 * math.log10(255.0) - 10 * math.log10(mse)
    return psnr


def scale_pred_to_gt_u8(pred_float, gt_u8):
    """Scale predicted float disparity to GT 0..255 space for fair PSNR.

    Uses robust percentile-based scaling to limit outliers' influence.
    """
    if not PSNR_ALIGN_TO_GT:
        # fallback to simple visualization normalization
        return normalize_to_uint8(pred_float)
    eps = 1e-6
    # robust max from percentiles
    gt_valid = gt_u8[gt_u8 != INVALID_DISPARITY_VALUE].astype(np.float32)
    if gt_valid.size == 0:
        return normalize_to_uint8(pred_float)
    gt_p = np.percentile(gt_valid, PSNR_ALIGN_PERC)
    pred_valid = pred_float[pred_float > eps].astype(np.float32)
    if pred_valid.size == 0:
        return normalize_to_uint8(pred_float)
    pred_p = float(np.percentile(pred_valid, PSNR_ALIGN_PERC))
    scale = gt_p / max(pred_p, eps)
    pred_scaled = pred_float * scale
    pred_scaled_u8 = np.clip(pred_scaled, 0, 255).astype(np.uint8)
    return pred_scaled_u8


def main():
    ensure_results_dir()
    loader = ImageLoader()
    left, right, gt_disp = loader.load_from(DATA_DIR)

    # Prepare grayscale uint8
    left_g = to_gray_u8(left)
    right_g = to_gray_u8(right)
    gt_g = to_gray_u8(gt_disp)

    # Step 2: rectification verification
    if ENABLE_RECTIFICATION_CHECK:
        ok, mean_dy = verify_horizontal_epipolar(left_g, right_g)
        print(f"Rectification check: {'OK' if ok else 'NOT OK'}, mean |dy| = {mean_dy:.3f} px")
    # Optional rectify
    left_g, right_g = maybe_rectify(left_g, right_g)

    # Step 3: denoise
    left_p = denoise(left_g)
    right_p = denoise(right_g)

    # Step 4: CLAHE
    left_p = apply_clahe(left_p)
    right_p = apply_clahe(right_p)

    # Step 5: SGBM
    disp_l, disp_r = compute_sgbm(left_p, right_p)

    # Raw visualization aligned to GT scale for fair PSNR
    disp_raw_vis = scale_pred_to_gt_u8(disp_l, gt_g)
    cv2.imwrite("results/disp_pred_raw.png", disp_raw_vis)

    # Step 6: LR consistency
    valid_mask = left_right_consistency(disp_l, disp_r)

    # Step 7: hole filling
    disp_filled = hole_fill(disp_l, valid_mask)

    # Step 8: smoothing (guided/bilateral)
    disp_smooth = smooth_disparity(disp_filled, left_g)

    # Step 9: final median denoise
    disp_final = final_median(disp_smooth)

    # Save refined (aligned to GT scale)
    disp_ref_vis = scale_pred_to_gt_u8(disp_final, gt_g)
    cv2.imwrite("results/disp_pred_refined.png", disp_ref_vis)

    # Step 10: PSNR
    psnr_raw = compute_psnr_with_mask(gt_g, disp_raw_vis)
    psnr_ref = compute_psnr_with_mask(gt_g, disp_ref_vis)
    print(f"PSNR raw: {psnr_raw:.2f} dB, refined: {psnr_ref:.2f} dB")
    with open("results/psnr_pipeline.txt", "w") as f:
        f.write(f"PSNR raw: {psnr_raw:.2f} dB\n")
        f.write(f"PSNR refined: {psnr_ref:.2f} dB\n")


if __name__ == "__main__":
    main()
