#!/usr/bin/env python3
"""Compute disparity for a rectified stereo pair and compare with ground-truth.

Loads `view1.png` (left), `view5.png` (right) and `disp1.png` (ground-truth left disparity)
from a provided folder (default: `StereoMatchingTestings/Art`). Uses OpenCV's stereo
matcher (SGBM if available, otherwise BM) to compute a disparity map for `view1` and
computes PSNR vs the ground-truth disparity.

Usage:
    python code/compute_disparity_and_psnr.py --dir StereoMatchingTestings/Art

The script writes:
  - `results/disp_estimated.png` : estimated disparity (uint8, 0-255)
  - `results/psnr_disparity.txt` : PSNR value
"""
import os
import argparse
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

from image_loader import ImageLoader


def ensure_results_dir():
    os.makedirs("results", exist_ok=True)


def to_gray_u8(img):
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert to uint8 if necessary
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def compute_disparity_sgbm(left_gray, right_gray, num_disparities=128, block_size=7):
    # num_disparities must be divisible by 16
    if num_disparities % 16 != 0:
        num_disparities = (num_disparities // 16 + 1) * 16

    try:
        matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 1 * block_size * block_size,
            P2=32 * 1 * block_size * block_size,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.StereoSGBM_MODE_SGBM_3WAY,
        )
    except Exception:
        matcher = None

    if matcher is None:
        # fallback to StereoBM
        matcher = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)

    disp = matcher.compute(left_gray, right_gray)
    # SGBM returns fixed-point with 4 fractional bits (i.e., scaled by 16)
    disp = disp.astype(np.float32)
    # If disparity was scaled by 16 (typical for SGBM), divide; for BM it might be already in pixels
    disp = disp / 16.0
    # Replace invalid values (usually negative) with 0
    disp[disp < 0] = 0
    return disp


def normalize_to_uint8(img):
    if img is None:
        return None
    minv = float(np.min(img))
    maxv = float(np.max(img))
    if maxv - minv < 1e-6:
        out = np.zeros(img.shape, dtype=np.uint8)
    else:
        out = ((img - minv) / (maxv - minv) * 255.0).astype(np.uint8)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", default="StereoMatchingTestings/Art", help="Dataset directory containing view1.png, view5.png, disp1.png")
    parser.add_argument("--num_disp", type=int, default=128, help="Maximum disparity search range (will be rounded to multiple of 16)")
    parser.add_argument("--block", type=int, default=7, help="Block size for matching (odd number)")
    args = parser.parse_args()

    ensure_results_dir()

    loader = ImageLoader()
    try:
        left, right, true_disp = loader.load_from(args.dir)
    except FileNotFoundError as e:
        print("File not found:", e)
        raise

    left_gray = to_gray_u8(left)
    right_gray = to_gray_u8(right)
    if left_gray is None or right_gray is None:
        raise RuntimeError("Failed to load images as grayscale uint8")

    # compute disparity (float, pixels)
    disp_float = compute_disparity_sgbm(left_gray, right_gray, num_disparities=args.num_disp, block_size=args.block)

    # normalize estimated disparity to uint8 for saving/visualization
    disp_vis = normalize_to_uint8(disp_float)
    out_path = os.path.join("results", "disp_estimated.png")
    cv2.imwrite(out_path, disp_vis)
    print(f"Estimated disparity saved to: {out_path}")

    # prepare ground-truth
    if true_disp is None:
        raise RuntimeError("Ground-truth disparity not loaded")
    true_gray = to_gray_u8(true_disp)

    # resize estimated to match true_disp if needed
    if disp_vis.shape != true_gray.shape:
        disp_for_compare = cv2.resize(disp_vis, (true_gray.shape[1], true_gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        disp_for_compare = disp_vis

    # compute PSNR (both uint8 0-255)
    psnr = peak_signal_noise_ratio(true_gray, disp_for_compare, data_range=255)
    print(f"PSNR between estimated disparity and ground-truth: {psnr:.2f} dB")

    with open(os.path.join("results", "psnr_disparity.txt"), "w") as f:
        f.write(f"PSNR: {psnr:.2f} dB\n")


if __name__ == "__main__":
    main()
