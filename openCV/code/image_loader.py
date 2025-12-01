import os
from typing import Tuple

import cv2
import numpy as np


class ImageLoader:
    """Load stereo images and disparity from a directory.

    Expects the following filenames inside the provided directory:
      - `view1.png`  (left image)
      - `view5.png`  (right image)
      - `disp1.png`  (disparity map)

    Usage:
        loader = ImageLoader()
        left, right, disp = loader.load_from('StereoMatchingTestings/Art')
    """

    LEFT_FILENAME = "view1.png"
    RIGHT_FILENAME = "view5.png"
    DISP_FILENAME = "disp1.png"

    def __init__(self, base_dir: str = ""):
        """Optional `base_dir` is prepended to provided directory paths.

        If `base_dir` is empty, `dir_path` should be a path relative to
        the current working directory.
        """
        self.base_dir = base_dir

    def _full_path(self, dir_path: str, filename: str) -> str:
        if self.base_dir:
            return os.path.join(self.base_dir, dir_path, filename)
        return os.path.join(dir_path, filename)

    def load_from(self, dir_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and return (left, right, disp) as numpy arrays.

        All files are read using `cv2.imread(..., cv2.IMREAD_UNCHANGED)` so
        the original image type/bit-depth is preserved.

        Raises:
            FileNotFoundError: if any of the expected files is missing.
        """
        left_path = self._full_path(dir_path, self.LEFT_FILENAME)
        right_path = self._full_path(dir_path, self.RIGHT_FILENAME)
        disp_path = self._full_path(dir_path, self.DISP_FILENAME)

        missing = [p for p in (left_path, right_path, disp_path) if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(f"Missing expected files: {missing}")

        left_img = cv2.imread(left_path, cv2.IMREAD_UNCHANGED)
        right_img = cv2.imread(right_path, cv2.IMREAD_UNCHANGED)
        disp_img = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
        
        # left_img = cv2.imread(left_path, 0)
        # right_img = cv2.imread(right_path, 0)
        # disp_img = cv2.imread(disp_path, 0)
        

        return left_img, right_img, disp_img
    

loader = ImageLoader()
left_image, right_image, disparity_map = loader.load_from('StereoMatchingTestings/Art')

print(f"Left image shape: {left_image.shape}, dtype: {left_image.dtype}")
print(f"Right image shape: {right_image.shape}, dtype: {right_image.dtype}")
print(f"Disparity map shape: {disparity_map.shape}, dtype: {disparity_map.dtype}")


