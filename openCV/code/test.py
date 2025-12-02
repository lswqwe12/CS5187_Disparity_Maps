from utils.image_loader import ImageLoader
import cv2
import numpy as np
import math
from PIL import Image

loader = ImageLoader()
left_image, right_image, disparity_map = loader.load_from('StereoMatchingTestings/Art')
opt_img = cv2.imread('/Test_disparity.png', 0)

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def test():
    gt_name = "./StereoMatchingTestings/Art/disp1.png"
    gt_img = np.array(Image.open(gt_name),dtype=float)
    
    
    pred_name = "./Test_disparity.png"
    pred_img = np.array(Image.open(pred_name),dtype=float)
        
    # When calculate the PSNR:
    # 1.) The pixels in ground-truth disparity map with '0' value will be neglected.
    # 2.) The left part region (1-250 columns) of view1 is not included as there is no
    #   corresponding pixels in the view5.
    [h,l] = gt_img.shape
    gt_img = gt_img[:, 250:l]
    pred_img = pred_img[:, 250:l]
    pred_img[gt_img==0]= 0

    peaksnr = calculate_psnr(pred_img, gt_img)
    print('The Peak-SNR value isss %0.4f \n', peaksnr)

if __name__== '__main__':
    test()   
