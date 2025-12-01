import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from image_loader import ImageLoader

IMAGE_PAIRS_PATH = 'StereoMatchingTestings/Art'

# 1. 使用 ImageLoader 读取并预处理图像对
loader = ImageLoader()
try:
    left_img, right_img, disp_img = loader.load_from(IMAGE_PAIRS_PATH)
except FileNotFoundError as e:
    raise


def preprocess_images(img_left, img_right):
    """Ensure images are grayscale and apply a small Gaussian blur."""
    if img_left is None or img_right is None:
        raise ValueError("Loaded images are None")

    # 转为灰度（如果是彩色）
    if img_left.ndim == 3:
        img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    if img_right.ndim == 3:
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # 高斯滤波去噪（核大小3×3，标准差0）
    img_left = cv2.GaussianBlur(img_left, (3, 3), 0)
    img_right = cv2.GaussianBlur(img_right, (3, 3), 0)
    return img_left, img_right


# 预处理并设置真实视差图
img_left, img_right = preprocess_images(left_img, right_img)
true_disp = disp_img
if true_disp is not None and getattr(true_disp, 'ndim', 2) == 3:
    true_disp = cv2.cvtColor(true_disp, cv2.COLOR_BGR2GRAY)


# 2. 代价计算（SAD）
def compute_sad_cost(img_left, img_right, window_size=5, max_disparity=64):
    height, width = img_left.shape
    half_win = window_size // 2  # 窗口半宽
    # 初始化代价矩阵：[高度, 宽度, 最大视差]
    cost = np.zeros((height, width, max_disparity), dtype=np.float32)
    
    # 遍历每个像素（避开边界，防止窗口越界）
    for y in range(half_win, height - half_win):
        for x in range(half_win, width - half_win):
            # 遍历视差范围
            for d in range(max_disparity):
                # 右图对应像素位置：x-d（水平匹配），需保证不越界
                if x - d < half_win:
                    cost[y, x, d] = float("inf")  # 越界则代价设为无穷大
                    continue
                # 提取左右窗口
                left_win = img_left[y-half_win:y+half_win+1, x-half_win:x+half_win+1]
                right_win = img_right[y-half_win:y+half_win+1, (x-d)-half_win:(x-d)+half_win+1]
                # 计算SAD代价
                cost[y, x, d] = np.sum(np.abs(left_win - right_win))
    return cost

# 调用函数
cost_matrix = compute_sad_cost(img_left, img_right, window_size=5, max_disparity=64)


# 3. 代价聚合（窗口聚合）
def aggregate_cost(cost_matrix, window_size=5):
    height, width, max_disp = cost_matrix.shape
    agg_cost = np.zeros_like(cost_matrix)
    # 对每个视差维度做盒滤波聚合
    for d in range(max_disp):
        agg_cost[:, :, d] = cv2.boxFilter(cost_matrix[:, :, d], -1, (window_size, window_size))
    return agg_cost

agg_cost_matrix = aggregate_cost(cost_matrix)


# 4. 视差计算（WTA）
disp_map = np.argmin(agg_cost_matrix, axis=2)  # 沿视差维度找最小值索引
# 归一化到0~255（方便保存和可视化）
disp_map = (disp_map / disp_map.max() * 255).astype(np.uint8)


# 5. 视差优化（中值滤波）
disp_map_opt = cv2.medianBlur(disp_map, 3)  # 3×3核
# 保存优化后的视差图
cv2.imwrite("results/group1_disp_opt.png", disp_map_opt)


# 6. 计算PSNR（需保证生成图与真实图尺寸一致）
def calculate_psnr(true_disp, pred_disp):
    # 统一尺寸（若不一致，resize）
    pred_disp = cv2.resize(pred_disp, (true_disp.shape[1], true_disp.shape[0]))
    # 计算PSNR（data_range=255表示像素范围0~255）
    psnr = peak_signal_noise_ratio(true_disp, pred_disp, data_range=255)
    return psnr

# 调用函数
psnr = calculate_psnr(true_disp, disp_map_opt)
# 保存PSNR结果
with open("results/psnr_all_groups.txt", "a") as f:
    f.write(f"group1 PSNR: {psnr:.2f} dB\n")
print(f"group1 PSNR: {psnr:.2f} dB")