import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt
from image_loader import ImageLoader

# ===================== 第一步：SIFT特征提取+BFMatcher匹配 =====================
def sift_bf_matching(img_left, img_right, ratio_thresh=0.75):
    """
    提取SIFT特征并通过BFMatcher+比值法过滤错误匹配
    :param img_left: 左视图（灰度图）
    :param img_right: 右视图（灰度图）
    :param ratio_thresh: 比值法阈值（过滤错误匹配）
    :return: 筛选后的匹配点对（pts_left, pts_right）
    """
    # 1. 创建SIFT检测器（OpenCV 4.x需用SIFT_create()）
    sift = cv2.SIFT_create()
    kp_left, des_left = sift.detectAndCompute(img_left, None)
    kp_right, des_right = sift.detectAndCompute(img_right, None)

    # 2. BFMatcher匹配（L2距离，适合SIFT描述子）
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_matches = bf.knnMatch(des_left, des_right, k=2)  # k=2用于比值法

    # 3. 比值法过滤错误匹配（David Lowe提出，提升匹配精度）
    good_matches = []
    for m, n in raw_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # 4. 提取匹配点对
    pts_left = np.float32([kp_left[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_right = np.float32([kp_right[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    return pts_left, pts_right

# ===================== 第二步：基础矩阵估计+极线验证 =====================
def fundamental_matrix_epiline(pts_left, pts_right, img_left, img_right):
    """
    估计基础矩阵+极线验证，过滤不符合极线约束的匹配点
    :param pts_left/pts_right: 初始匹配点对
    :param img_left/img_right: 原始图像（用于绘制极线，可选）
    :return: 基础矩阵F、筛选后的有效匹配点对
    """
    # 1. 用RANSAC估计基础矩阵（鲁棒过滤外点）
    F, mask = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.99)
    mask = mask.ravel().astype(bool)  # 转换为布尔掩码

    # 2. 过滤不符合基础矩阵的匹配点
    pts_left_valid = pts_left[mask]
    pts_right_valid = pts_right[mask]

    # 3. 极线验证（可选，可视化验证效果）
    def draw_epilines(img1, img2, lines, pts1, pts2):
        h, w = img1.shape
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2]/r[1]])
            x1, y1 = map(int, [w, -(r[2]+r[0]*w)/r[1]])
            # 裁剪坐标到图像范围内
            x0 = np.clip(x0, 0, w-1)
            y0 = np.clip(y0, 0, h-1)
            x1 = np.clip(x1, 0, w-1)
            y1 = np.clip(y1, 0, h-1)
            img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
            # ✅ 修复2：匹配点坐标转整数（核心修复）
            pt1_int = tuple(map(int, pt1.ravel()))  # 浮点→整数
            pt2_int = tuple(map(int, pt2.ravel()))
            img1 = cv2.circle(img1, pt1_int, 5, color, -1)
            img2 = cv2.circle(img2, pt2_int, 5, color, -1)
        return img1, img2

    # 绘制极线（可选，验证校正前的极线分布）
    lines1 = cv2.computeCorrespondEpilines(pts_right_valid.reshape(-1,1,2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    # ✅ 修复3：限制绘制的极线数量（避免过多点导致卡顿/报错）
    draw_num = min(50, len(lines1))  # 最多绘制50条极线
    img_left_epi, img_right_epi = draw_epilines(
        img_left, img_right, 
        lines1[:draw_num], 
        pts_left_valid[:draw_num], 
        pts_right_valid[:draw_num]
    )
    plt.subplot(121), plt.imshow(img_left_epi), plt.title('Left Image + Epilines')
    plt.subplot(122), plt.imshow(img_right_epi), plt.title('Right Image + Matches')
    plt.savefig('results/epiline_verification.png')  # 保存极线验证图（报告用）
    plt.close()

    return F, pts_left_valid, pts_right_valid

# ===================== 第三步：无标定立体校正+透视变换 =====================
def stereo_rectify_uncalibrated_transform(F, pts_left, pts_right, img_left, img_right):
    """
    无标定立体校正（stereoRectifyUncalibrated）+ 透视变换（warpPerspective）
    :param F: 基础矩阵
    :param pts_left/pts_right: 有效匹配点对
    :param img_left/img_right: 原始灰度图
    :return: 校正后的左右图像（极线水平对齐）
    """
    h, w = img_left.shape
    # 1. 计算校正变换矩阵H1（左图）、H2（右图）
    retval, H1, H2 = cv2.stereoRectifyUncalibrated(
        pts_left, pts_right, F, (w, h),
        threshold=5.0  # 校正阈值，越小越严格
    )

    # 2. 透视变换（warpPerspective），得到极线水平对齐的图像
    img_left_rect = cv2.warpPerspective(img_left, H1, (w, h))
    img_right_rect = cv2.warpPerspective(img_right, H2, (w, h))

    # 保存校正后的图像（报告用）
    cv2.imwrite('results/img_left_rect.png', img_left_rect)
    cv2.imwrite('results/img_right_rect.png', img_right_rect)

    return img_left_rect, img_right_rect

# ===================== 第四步：校正后立体匹配（优化版SAD） =====================
def optimized_sad_matching(img_left_rect, img_right_rect, window_size=5, max_disparity=64):
    """
    基于校正后图像的优化版SAD立体匹配（加入代价聚合+中值滤波）
    :param img_left_rect/img_right_rect: 校正后的左右图像
    :param window_size: 匹配窗口大小
    :param max_disparity: 最大视差
    :return: 优化后的视差图
    """
    half_win = window_size // 2
    h, w = img_left_rect.shape
    # 1. 初始化代价矩阵
    cost_matrix = np.zeros((h, w, max_disparity), dtype=np.float32)

    # 2. 代价计算（SAD）+ 极线约束（仅水平搜索，校正后无需垂直搜索）
    for y in range(half_win, h - half_win):
        for x in range(half_win, w - half_win):
            for d in range(max_disparity):
                if x - d < half_win:
                    cost_matrix[y, x, d] = np.inf
                    continue
                # 提取窗口并计算SAD
                left_win = img_left_rect[y-half_win:y+half_win+1, x-half_win:x+half_win+1]
                right_win = img_right_rect[y-half_win:y+half_win+1, (x-d)-half_win:(x-d)+half_win+1]
                cost_matrix[y, x, d] = np.sum(np.abs(left_win - right_win))

    # 3. 代价聚合（盒滤波，减少噪声）
    agg_cost = np.zeros_like(cost_matrix)
    for d in range(max_disparity):
        agg_cost[:, :, d] = cv2.boxFilter(cost_matrix[:, :, d], -1, (window_size, window_size))

    # 4. 视差计算（WTA）+ 优化
    disp_map = np.argmin(agg_cost, axis=2)
    disp_map = (disp_map / disp_map.max() * 255).astype(np.uint8)
    disp_map_opt = cv2.medianBlur(disp_map, 3)  # 中值滤波去噪

    return disp_map_opt

# ===================== 主流程：整合所有步骤 =====================
def main():
    # 1. 配置路径（替换为你的数据路径）
    dir_path = 'StereoMatchingTestings/Art'
    loader = ImageLoader()

    # 2. 读取原始图像（灰度模式）
    img_left, img_right, true_disp = loader.load_from(dir_path)
    
    # 检查图像是否读取成功
    if img_left is None or img_right is None or true_disp is None:
        raise ValueError("图像读取失败！请检查文件路径是否正确")

    # 3. SIFT+BFMatcher匹配
    pts_left, pts_right = sift_bf_matching(img_left, img_right)
    print(f"初始匹配点数量：{len(pts_left)}")
    
    if len(pts_left) < 8:  # 基础矩阵估计至少需要8个点
        raise ValueError("有效匹配点数量不足！请降低ratio_thresh阈值")

    # 4. 基础矩阵+极线验证
    F, pts_left_valid, pts_right_valid = fundamental_matrix_epiline(pts_left, pts_right, img_left, img_right)
    print(f"过滤后有效匹配点数量：{len(pts_left_valid)}")

    # 5. 无标定校正+透视变换
    img_left_rect, img_right_rect = stereo_rectify_uncalibrated_transform(F, pts_left_valid, pts_right_valid, img_left, img_right)

    # 6. 优化版SAD匹配生成视差图
    disp_map_opt = optimized_sad_matching(img_left_rect, img_right_rect)

    # 7. 计算PSNR（评估优化效果）
    disp_map_opt = cv2.resize(disp_map_opt, (true_disp.shape[1], true_disp.shape[0]))
    psnr = peak_signal_noise_ratio(true_disp, disp_map_opt, data_range=255)
    
    print(f"优化后视差图PSNR：{psnr:.2f} dB")

    # 8. 保存结果
    cv2.imwrite('results/disp_map_optimized.png', disp_map_opt)
    with open('results/psnr_optimized.txt', 'w') as f:
        f.write(f"优化后PSNR：{psnr:.2f} dB")

if __name__ == "__main__":
    main()