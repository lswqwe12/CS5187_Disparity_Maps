#!/usr/bin/env python3
"""
PSMNet-based stereo disparity pipeline with configurable pre/post-processing and PSNR evaluation.

Steps:
 1) Load data via openCV/code/utils/image_loader.py
 2) Optional rectification verification (and optional uncalibrated rectify)
 3) Optional denoising (Gaussian/Median)
 4) Optional CLAHE
 5) PSMNet inference (supports dynamic import from external repo)
 6) Post-processing: Left-Right consistency check
 7) Post-processing: Hole filling (dilate + neighborhood fill)
 8) Post-processing: Disparity smoothing (guided/bilateral)
 9) Post-processing: Final median denoise
10) Compute PSNR vs ground-truth with valid mask and robust scale alignment

Outputs:
 - results/disp_pred_raw.png           : raw disparity (scaled for visualization/PSNR)
 - results/disp_pred_refined.png       : refined disparity (scaled for visualization/PSNR)
 - results/psnr_pipeline.txt           : PSNR report
"""
import os
import sys
import math
import importlib
from typing import Tuple

import cv2
import numpy as np
import torch

# Add project root to sys.path for importing the ImageLoader utility
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from utils.image_loader import ImageLoader
from config import (
    DATA_DIR,
    RESULTS_DIR,
    DEVICE,
    USE_HALF,
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
    PSMNET_REPO_DIR,
    PSMNET_CHECKPOINT,
    MAX_DISP,
    NORM_MEAN,
    NORM_STD,
    USE_IMAGENET_NORM,
    RESIZE_TO,
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
    os.makedirs(RESULTS_DIR, exist_ok=True)


def to_gray_u8(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def verify_horizontal_epipolar(img_left: np.ndarray, img_right: np.ndarray) -> Tuple[bool, float]:
    """Verify rectification by checking matched points' y-coordinates are similar."""
    sift = cv2.SIFT_create()
    kp_l, des_l = sift.detectAndCompute(img_left, None)
    kp_r, des_r = sift.detectAndCompute(img_right, None)
    if des_l is None or des_r is None:
        return False, float('inf')
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw = bf.knnMatch(des_l, des_r, k=2)
    good = []
    for m_n in raw:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) == 0:
        return False, float('inf')
    pts_l = np.float32([kp_l[m.queryIdx].pt for m in good])
    pts_r = np.float32([kp_r[m.trainIdx].pt for m in good])
    dy = np.abs(pts_l[:, 1] - pts_r[:, 1])
    mean_dy = float(np.mean(dy))
    return mean_dy < 1.0, mean_dy  # threshold 1px


def maybe_rectify(img_left: np.ndarray, img_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not ENABLE_RECTIFICATION_WARP:
        return img_left, img_right
    sift = cv2.SIFT_create()
    kp_l, des_l = sift.detectAndCompute(img_left, None)
    kp_r, des_r = sift.detectAndCompute(img_right, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw = bf.knnMatch(des_l, des_r, k=2)
    good = []
    for m_n in raw:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 8:
        return img_left, img_right
    pts_l = np.float32([kp_l[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_r = np.float32([kp_r[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    h, w = img_left.shape
    F, mask = cv2.findFundamentalMat(pts_l, pts_r, cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.99)
    if F is None:
        return img_left, img_right
    retval, H1, H2 = cv2.stereoRectifyUncalibrated(pts_l[mask.ravel() == 1], pts_r[mask.ravel() == 1], F, (w, h))
    if not retval:
        return img_left, img_right
    left_rect = cv2.warpPerspective(img_left, H1, (w, h))
    right_rect = cv2.warpPerspective(img_right, H2, (w, h))
    return left_rect, right_rect


def denoise(img: np.ndarray) -> np.ndarray:
    if not ENABLE_DENOISE:
        return img
    if DENOISE_METHOD == "gaussian":
        return cv2.GaussianBlur(img, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)
    else:
        return cv2.medianBlur(img, MEDIAN_KERNEL)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    if not ENABLE_CLAHE:
        return img
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    return clahe.apply(img)


# ------------------------ PSMNet loading & inference ------------------------

def _dynamic_import_psmnet(psmnet_repo_dir: str):
    """Dynamically import PSMNet from an external repo directory.

    Expects a module path models/stackhourglass.py that defines class `PSMNet`.
    Returns the PSMNet class.
    """
    if psmnet_repo_dir:
        if psmnet_repo_dir not in sys.path:
            sys.path.insert(0, psmnet_repo_dir)
    try:
        mod = importlib.import_module("models.stackhourglass")
    except Exception as e:
        raise ImportError(
            "Could not import PSMNet from 'models.stackhourglass'. "
            "Set PSMNET_REPO_DIR in config to a valid PSMNet repo directory."
        ) from e
    if not hasattr(mod, "PSMNet"):
        raise ImportError("Module models.stackhourglass does not define PSMNet class.")
    return getattr(mod, "PSMNet")


def _load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"PSMNet checkpoint not found at {ckpt_path}. Set PSMNET_CHECKPOINT in config."
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    state = None
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        else:
            # assume it is already a state dict
            state = ckpt
    else:
        state = ckpt
    # Remove possible "module." prefixes from DDP
    new_state = {}
    for k, v in state.items():
        nk = k
        if k.startswith("module."):
            nk = k[len("module."):]
        new_state[nk] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"[Warn] Missing keys when loading: {len(missing)} e.g., {missing[:5]}")
    if unexpected:
        print(f"[Warn] Unexpected keys when loading: {len(unexpected)} e.g., {unexpected[:5]}")


def _prepare_input(left_bgr: np.ndarray, right_bgr: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int, int]]:
    """Convert BGR images to normalized CHW tensors and pad to multiples of 16.

    Returns tensors on CPU (will be moved to device later) and pad sizes (top, left, bottom, right).
    """
    # Ensure 3 channels
    if left_bgr.ndim == 2:
        left_bgr = cv2.cvtColor(left_bgr, cv2.COLOR_GRAY2BGR)
    if right_bgr.ndim == 2:
        right_bgr = cv2.cvtColor(right_bgr, cv2.COLOR_GRAY2BGR)

    # Optional resize
    if RESIZE_TO is not None:
        Ht, Wt = RESIZE_TO
        left_bgr = cv2.resize(left_bgr, (Wt, Ht), interpolation=cv2.INTER_AREA)
        right_bgr = cv2.resize(right_bgr, (Wt, Ht), interpolation=cv2.INTER_AREA)

    # BGR -> RGB
    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    if USE_IMAGENET_NORM:
        mean = np.array(NORM_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(NORM_STD, dtype=np.float32).reshape(1, 1, 3)
        left_rgb = (left_rgb - mean) / std
        right_rgb = (right_rgb - mean) / std

    # HWC -> CHW
    left_chw = np.transpose(left_rgb, (2, 0, 1))
    right_chw = np.transpose(right_rgb, (2, 0, 1))

    # Pad to multiple of 16 (PSMNet requirement)
    _, h, w = left_chw.shape
    pad_h = (16 - (h % 16)) % 16
    pad_w = (16 - (w % 16)) % 16
    # Pad at bottom and right
    left_pad = np.pad(left_chw, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
    right_pad = np.pad(right_chw, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')

    left_t = torch.from_numpy(left_pad).unsqueeze(0)  # 1x3xHxW
    right_t = torch.from_numpy(right_pad).unsqueeze(0)
    return left_t, right_t, (0, 0, pad_h, pad_w)


def _unpad_disp(disp: np.ndarray, pads: Tuple[int, int, int, int]) -> np.ndarray:
    top, left, bottom, right = pads
    h, w = disp.shape
    return disp[top:h - bottom if bottom > 0 else h, left:w - right if right > 0 else w]


def run_psmnet(left_bgr: np.ndarray, right_bgr: np.ndarray, device_str: str) -> np.ndarray:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    PSMNet = _dynamic_import_psmnet(PSMNET_REPO_DIR)
    model = PSMNet(MAX_DISP)
    model.eval()
    model.to(device)

    if USE_HALF and device.type == 'cuda':
        model.half()

    if PSMNET_CHECKPOINT:
        _load_checkpoint(model, PSMNET_CHECKPOINT, device)

    left_t, right_t, pads = _prepare_input(left_bgr, right_bgr)
    left_t = left_t.to(device)
    right_t = right_t.to(device)
    if USE_HALF and device.type == 'cuda':
        left_t = left_t.half()
        right_t = right_t.half()

    with torch.no_grad():
        # Common PSMNet forward returns disparity in original scale
        pred = model(left_t, right_t)
        if isinstance(pred, (list, tuple)):
            disp = pred[-1]
        else:
            disp = pred
        disp = disp.squeeze(0).squeeze(0).float().cpu().numpy()

    disp = _unpad_disp(disp, pads)
    # Negative disparities are invalid
    disp[disp < 0] = 0.0
    return disp.astype(np.float32)


# ------------------------ Post-processing helpers ------------------------

def left_right_consistency(disp_l: np.ndarray, disp_r: np.ndarray) -> np.ndarray:
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


def hole_fill(disp: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    if not ENABLE_HOLE_FILL:
        return disp
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, DILATE_KERNEL)
    valid_dilated = cv2.dilate(valid_mask, kernel, iterations=DILATE_ITERATIONS)
    filled = disp.copy()
    half = FILL_WINDOW // 2
    h, w = disp.shape
    for y in range(h):
        for x in range(w):
            if valid_mask[y, x] == 0:
                y0, y1 = max(0, y - half), min(h, y + half + 1)
                x0, x1 = max(0, x - half), min(w, x + half + 1)
                neigh = disp[y0:y1, x0:x1]
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


def smooth_disparity(disp: np.ndarray, guide_gray_u8: np.ndarray) -> np.ndarray:
    if not ENABLE_SMOOTHING:
        return disp
    out = disp.copy().astype(np.float32)
    if USE_GUIDED_FILTER:
        try:
            guide_u8 = guide_gray_u8 if guide_gray_u8.dtype == np.uint8 else cv2.normalize(guide_gray_u8, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            out = cv2.ximgproc.guidedFilter(guide_u8, out.astype(np.float32), GUIDED_RADIUS, GUIDED_EPS)
        except Exception:
            pass
    if USE_BILATERAL_FILTER:
        out = cv2.bilateralFilter(out.astype(np.float32), BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)
    return out


def final_median(disp: np.ndarray) -> np.ndarray:
    if not ENABLE_FINAL_MEDIAN:
        return disp
    return cv2.medianBlur(disp.astype(np.float32), FINAL_MEDIAN_KERNEL)


# ------------------------ PSNR helpers ------------------------

def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    minv = float(np.min(img))
    maxv = float(np.max(img))
    if maxv - minv < 1e-6:
        return np.zeros(img.shape, dtype=np.uint8)
    return ((img - minv) / (maxv - minv) * 255.0).astype(np.uint8)


def scale_pred_to_gt_u8(pred_float: np.ndarray, gt_u8: np.ndarray) -> np.ndarray:
    if not PSNR_ALIGN_TO_GT:
        return normalize_to_uint8(pred_float)
    eps = 1e-6
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


def compute_psnr_with_mask(gt: np.ndarray, pred: np.ndarray) -> float:
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    valid = (gt != INVALID_DISPARITY_VALUE) & (pred != INVALID_DISPARITY_VALUE)
    if not np.any(valid):
        return 0.0
    gt_valid = gt[valid].astype(np.float32)
    pred_valid = pred[valid].astype(np.float32)
    mse = float(np.mean((gt_valid - pred_valid) ** 2))
    if mse <= 1e-12:
        return 100.0
    psnr = 20 * math.log10(255.0) - 10 * math.log10(mse)
    return psnr


# ------------------------ Main pipeline ------------------------

def main():
    ensure_results_dir()

    loader = ImageLoader()
    left_img, right_img, gt_disp = loader.load_from(DATA_DIR)

    # Prepare grayscale uint8 for pre-processing and guidance
    left_g = to_gray_u8(left_img)
    right_g = to_gray_u8(right_img)
    gt_g = to_gray_u8(gt_disp)

    # Step 2: rectification verification (and optional rectify)
    if ENABLE_RECTIFICATION_CHECK:
        ok, mean_dy = verify_horizontal_epipolar(left_g, right_g)
        print(f"Rectification check: {'OK' if ok else 'NOT OK'}, mean |dy| = {mean_dy:.3f} px")
    left_p, right_p = maybe_rectify(left_g, right_g)

    # Step 3: denoise
    left_p = denoise(left_p)
    right_p = denoise(right_p)

    # Step 4: CLAHE
    left_p = apply_clahe(left_p)
    right_p = apply_clahe(right_p)

    # For PSMNet, provide 3-channel BGR images; use original color if available
    if left_img.ndim == 3:
        left_for_net = left_img.copy()
        right_for_net = right_img.copy()
    else:
        left_for_net = cv2.cvtColor(left_p, cv2.COLOR_GRAY2BGR)
        right_for_net = cv2.cvtColor(right_p, cv2.COLOR_GRAY2BGR)

    # Step 5: PSMNet inference (left disparity)
    try:
        disp_l = run_psmnet(left_for_net, right_for_net, DEVICE)
    except Exception as e:
        print("[Error] PSMNet inference failed:", e)
        print("Please ensure PSMNET_REPO_DIR and PSMNET_CHECKPOINT are correctly set in DL/code/config.py.")
        sys.exit(1)

    # Also compute right disparity for LR consistency
    try:
        disp_r = run_psmnet(right_for_net, left_for_net, DEVICE)
    except Exception:
        # If right pass fails, use left disparity as a placeholder to keep pipeline running
        disp_r = disp_l.copy()

    # Raw visualization scaled to GT range
    disp_raw_vis = scale_pred_to_gt_u8(disp_l, gt_g)
    cv2.imwrite(os.path.join(RESULTS_DIR, "disp_pred_raw.png"), disp_raw_vis)

    # Step 6: Left-Right consistency
    valid_mask = left_right_consistency(disp_l, disp_r)

    # Step 7: Hole filling
    disp_filled = hole_fill(disp_l, valid_mask)

    # Step 8: Smoothing
    disp_smooth = smooth_disparity(disp_filled, left_g)

    # Step 9: Final median denoise
    disp_final = final_median(disp_smooth)

    # Save refined (aligned to GT scale)
    disp_ref_vis = scale_pred_to_gt_u8(disp_final, gt_g)
    cv2.imwrite(os.path.join(RESULTS_DIR, "disp_pred_refined.png"), disp_ref_vis)

    # Step 10: PSNR
    psnr_raw = compute_psnr_with_mask(gt_g, disp_raw_vis)
    psnr_ref = compute_psnr_with_mask(gt_g, disp_ref_vis)
    print(f"PSNR raw: {psnr_raw:.2f} dB, refined: {psnr_ref:.2f} dB")
    with open(os.path.join(RESULTS_DIR, "psnr_pipeline.txt"), "w") as f:
        f.write(f"PSNR raw: {psnr_raw:.2f} dB\n")
        f.write(f"PSNR refined: {psnr_ref:.2f} dB\n")


if __name__ == "__main__":
    main()
