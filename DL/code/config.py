"""
Configuration for PSMNet-based stereo disparity pipeline (DL).

All tunable parameters are centralized here with detailed comments.
"""

# ------------------------ Data paths ------------------------
# Directory containing stereo pair and ground-truth:
#   - view1.png (left), view5.png (right), disp1.png (ground-truth disparity)
DATA_DIR = "StereoMatchingTestings/Art"

# Output directory for predictions, visualizations and logs.
RESULTS_DIR = "results"

# ------------------------ Device & precision ------------------------
# Compute device: "cuda" to use GPU if available, else "cpu".
DEVICE = "cpu"

# Use mixed precision (fp16) during inference to reduce memory and speed up.
# This requires a CUDA device; will be ignored on CPU.
USE_HALF = True

# ------------------------ Preprocessing toggles ------------------------
# Whether to run rectification verification using feature matching to check
# that corresponding points lie on the same scanline (horizontal epipolar geometry).
ENABLE_RECTIFICATION_CHECK = True

# Whether to attempt uncalibrated rectification via fundamental matrix and
# homographies. For the provided data, images are already rectified, so keep False.
ENABLE_RECTIFICATION_WARP = False

# Denoising before matching to suppress sensor noise and textureless speckles.
ENABLE_DENOISE = True            # Enable/disable denoising
DENOISE_METHOD = "median"       # "gaussian" or "median"
GAUSSIAN_KERNEL = (3, 3)         # Kernel size for Gaussian blur (odd sizes recommended)
GAUSSIAN_SIGMA = 0.0             # Standard deviation for Gaussian blur; 0 lets OpenCV compute from kernel
MEDIAN_KERNEL = 3                # Kernel size for median blur (must be odd)

# Local contrast enhancement using CLAHE to improve local texture and edges.
ENABLE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0           # Contrast limit for local histogram equalization
CLAHE_TILE_GRID = (8, 8)         # Tile grid size; smaller increases local adaptation

# ------------------------ PSMNet model settings ------------------------
# Path to a local clone of a PSMNet repository that defines the network.
# The common structure has a module at "models/stackhourglass.py" exporting class `PSMNet`.
# Example: 
#   PSMNET_REPO_DIR = "/path/to/PSMNet"  (folder containing a subfolder `models/`)
PSMNET_REPO_DIR = ""  # Leave empty if `models` are importable already

# Checkpoint path to pretrained PSMNet weights (.pth/.tar).
# Many checkpoints store weights under the key "state_dict"; this loader handles common formats.
PSMNET_CHECKPOINT = ""

# Maximum disparity for PSMNet; typical values are 192 or 256.
MAX_DISP = 192

# Input normalization for the network: scale to [0,1] then optionally standardize.
# If your pretrained weights expect ImageNet normalization, set these to ImageNet stats.
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)
USE_IMAGENET_NORM = False  # If False, only scale to [0,1] without mean/std

# Resize policy before network: typically keep original size. PSMNet requires
# dimensions divisible by 16; the pipeline pads internally and unpads outputs.
RESIZE_TO = None  # e.g., (H, W) or None to keep original

# ------------------------ Post-processing ------------------------
# Left-right consistency check to invalidate mismatched pixels.
ENABLE_LR_CHECK = True
LR_MAX_DIFF = 1.0                # Max disparity difference tolerated between L->R and R->L maps

# Hole filling to fill invalid pixels (from LR check) assuming local continuity.
ENABLE_HOLE_FILL = True
DILATE_KERNEL = (3, 3)           # Morphological dilation kernel to grow valid regions
DILATE_ITERATIONS = 1            # Iterations for dilation
FILL_STRATEGY = "median"        # "mean" or "median" for neighborhood-based filling
FILL_WINDOW = 3                  # Neighborhood window size (odd)

# Edge-preserving smoothing for disparity using the left image as guidance.
ENABLE_SMOOTHING = True
USE_GUIDED_FILTER = True         # Requires ximgproc in OpenCV contrib; ignored if unavailable
GUIDED_RADIUS = 9                # Guided filter radius (pixels)
GUIDED_EPS = 1e-3                # Regularization epsilon for guided filter
USE_BILATERAL_FILTER = False     # Bilateral filter for additional edge-preserving smoothing
BILATERAL_D = 7                  # Bilateral neighborhood diameter (pixels)
BILATERAL_SIGMA_COLOR = 75       # Color sigma for bilateral filter
BILATERAL_SIGMA_SPACE = 75       # Coordinate sigma for bilateral filter

# Final median filter to suppress isolated noisy disparity spikes.
ENABLE_FINAL_MEDIAN = True
FINAL_MEDIAN_KERNEL = 3          # Kernel size (odd)

# ------------------------ PSNR evaluation ------------------------
# Invalid disparity value used to mask out pixels for PSNR computation.
INVALID_DISPARITY_VALUE = 0      # 0 is common in GT disparity for invalid/occluded regions

# Align predicted disparity to the ground-truth dynamic range before computing PSNR.
# This reduces the impact of absolute-scale differences between methods.
PSNR_ALIGN_TO_GT = True
PSNR_ALIGN_PERC = 99.0           # Percentile for robust scale matching (limits outliers)
