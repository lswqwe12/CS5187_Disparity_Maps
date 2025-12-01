"""
Configuration for stereo disparity pipeline.

All parameters are documented for clarity and easy tuning.
"""

# ------------------------ Data paths ------------------------
# Directory containing `view1.png`, `view5.png`, `disp1.png`.
DATA_DIR = "StereoMatchingTestings/Art"

# ------------------------ Step toggles ------------------------
# Whether to run rectification verification (epipolar check) and optionally rectify.
ENABLE_RECTIFICATION_CHECK = True  # Validate horizontal epipolar alignment using matches
ENABLE_RECTIFICATION_WARP = False  # If True, perform stereoRectifyUncalibrated + warpPerspective

# Denoising toggle and method.
ENABLE_DENOISE = True             # Apply denoising before matching
DENOISE_METHOD = "median"       # "gaussian" or "median"
GAUSSIAN_KERNEL = (3, 3)          # Gaussian kernel size
GAUSSIAN_SIGMA = 0.0              # Gaussian sigma
MEDIAN_KERNEL = 3                 # Median filter kernel size (odd)

# Contrast enhancement (CLAHE).
ENABLE_CLAHE = True               # Enhance local contrast to improve matching
CLAHE_CLIP_LIMIT = 2.0            # Contrast clip limit
CLAHE_TILE_GRID = (8, 8)          # Tile grid size

# ------------------------ SGBM parameters ------------------------
SGBM_MIN_DISPARITY = 0            # Minimum possible disparity value
SGBM_NUM_DISPARITIES = 128        # Must be multiple of 16; search range
SGBM_BLOCK_SIZE = 7               # Matching block size (odd)
SGBM_P1 = 8 * SGBM_BLOCK_SIZE * SGBM_BLOCK_SIZE   # Smoothness penalty for small disparity changes
SGBM_P2 = 40 * SGBM_BLOCK_SIZE * SGBM_BLOCK_SIZE  # Smoothness penalty for larger disparity changes
SGBM_DISP12_MAX_DIFF = 1         # Max allowed difference in left-right disparity check inside SGBM
SGBM_UNIQUENESS_RATIO = 10         # Uniqueness threshold; lower can increase matches but also noise
SGBM_SPECKLE_WINDOW_SIZE = 100    # Remove speckles (small regions of noise)
SGBM_SPECKLE_RANGE = 32           # Speckle disparity range
SGBM_PREFILTER_CAP = 63           # Pre-filter cap
SGBM_MODE = "SGBM_3WAY"           # "SGBM" or "SGBM_3WAY"

# ------------------------ Post-processing ------------------------
# Left-right consistency check (LR-check)
ENABLE_LR_CHECK = False
LR_MAX_DIFF = 1.0                 # Max allowed disparity difference for LR consistency (slightly relaxed)

# Hole filling parameters
ENABLE_HOLE_FILL = True
DILATE_KERNEL = (3, 3)            # Morphological dilation kernel to expand valid regions
DILATE_ITERATIONS = 2             # Slightly stronger dilation to reduce tiny holes
FILL_STRATEGY = "mean"          # "mean" or "median" neighborhood filling
FILL_WINDOW = 3                  # Neighborhood window size for filling (odd, larger window)

# Disparity smoothing
ENABLE_SMOOTHING = True
USE_GUIDED_FILTER = True          # Requires cv2.ximgproc; guidedFilter preserves edges
GUIDED_RADIUS = 9                 # Guided filter radius
GUIDED_EPS = 1e-3                 # Regularization epsilon
USE_BILATERAL_FILTER = False       # Bilateral filter for edge-preserving smoothing
BILATERAL_D = 7                   # Diameter of pixel neighborhood (tuned)
BILATERAL_SIGMA_COLOR = 75        # Filter sigma in the color space
BILATERAL_SIGMA_SPACE = 75        # Filter sigma in the coordinate space

# Final noise removal
ENABLE_FINAL_MEDIAN = False
FINAL_MEDIAN_KERNEL = 3           # Final median blur kernel size

# ------------------------ PSNR evaluation ------------------------
# When computing PSNR, align sizes and mask invalid pixels (e.g., zeros in GT or prediction)
INVALID_DISPARITY_VALUE = 0       # Pixels with this value are considered invalid in masks

# Align predicted disparity scale to GT for PSNR instead of per-image min-max normalization.
PSNR_ALIGN_TO_GT = True           # Scale prediction to GT dynamic range before PSNR
PSNR_ALIGN_PERC = 99.0            # Percentile used for robust scale estimation
