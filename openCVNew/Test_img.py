from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import math
import cv2
from PIL import Image

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--loadmodel', default='./trained/pretrained_model_KITTI2015.tar',
                    help='loading model')
parser.add_argument('--leftimg', default= './StereoMatchingTestings/Art/view1.png',
                    help='load model')
parser.add_argument('--rightimg', default= './StereoMatchingTestings/Art/view5.png',
                    help='load model')                                      
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto',
                    help='device to run on: auto, cpu, cuda, or mps (Apple Silicon)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def resolve_device(pref: str) -> torch.device:
    if pref == 'cpu':
        return torch.device('cpu')
    if pref == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            print('CUDA requested but not available, falling back to CPU.')
            return torch.device('cpu')
    if pref == 'mps':
        mps_ok = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built()
        if mps_ok:
            return torch.device('mps')
        else:
            print('MPS requested but not available, falling back to CPU.')
            return torch.device('cpu')
    # auto
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device('mps')
    return torch.device('cpu')


def compute_SGBM(imgL, imgR):
    """Use OpenCV SGBM to generate disparity for the left image.

    Note: imgL/imgR are already preprocessed (ToTensor + Normalize + pad) in main.
    We only convert back to uint8 images for OpenCV and run SGBM, without extra
    preprocessing (no filtering/CLAHE/etc.).
    Returns HxW float32 disparity in original padded size.
    """
    # Remove batch dimension [1,3,H,W] -> [3,H,W]
    if imgL.dim() == 4:
        imgL = imgL[0]
    if imgR.dim() == 4:
        imgR = imgR[0]

    # De-normalize back to [0,255] RGB to feed OpenCV
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=imgL.dtype, device=imgL.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=imgL.dtype, device=imgL.device).view(3, 1, 1)
    imgL_uint8 = ((imgL * std + mean).clamp(0, 1) * 255.0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    imgR_uint8 = ((imgR * std + mean).clamp(0, 1) * 255.0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)

    # Convert to grayscale for SGBM
    left_gray = cv2.cvtColor(imgL_uint8, cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(imgR_uint8, cv2.COLOR_RGB2GRAY)

    # Configure SGBM (numDisparities must be multiple of 16)
    block_size = 7
    min_disp = 0
    num_disp = max(16, (int(math.ceil(args.maxdisp / 16.0)) * 16))
    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.StereoSGBM_MODE_SGBM_3WAY,
    )

    disp = sgbm.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disp[disp < 0] = 0.0
    return disp


def pre_process_disparity(leftimg_path, rightimg_path):
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(**normal_mean_var)])    
    imgL_o = Image.open(leftimg_path).convert('RGB')
    imgR_o = Image.open(rightimg_path).convert('RGB')
    imgL = infer_transform(imgL_o)
    imgR = infer_transform(imgR_o) 
    # pad to width and hight to 16 times
    if imgL.shape[1] % 16 != 0:
        times = imgL.shape[1]//16       
        top_pad = (times+1)*16 -imgL.shape[1]
    else:
        top_pad = 0

    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2]//16                       
        right_pad = (times+1)*16-imgL.shape[2]
    else:
        right_pad = 0    
    imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
    imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)
    return imgL, imgR, top_pad, right_pad


def post_process_disparity(pred_disp, top_pad, right_pad, parent_dir):
    if top_pad !=0 and right_pad != 0:
        img = pred_disp[top_pad:,:-right_pad]
    elif top_pad ==0 and right_pad != 0:
        img = pred_disp[:,:-right_pad]
    elif top_pad !=0 and right_pad == 0:
        img = pred_disp[top_pad:,:]
    else:
        img = pred_disp
    img = (img*256).astype('uint16')
    img = Image.fromarray(img)
    img.save(f'pred_disparity_{parent_dir}.png')


def main():
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(**normal_mean_var)])    

    imgL_o = Image.open(args.leftimg).convert('RGB')
    imgR_o = Image.open(args.rightimg).convert('RGB')

    imgL = infer_transform(imgL_o)
    imgR = infer_transform(imgR_o) 
    

    # pad to width and hight to 16 times
    if imgL.shape[1] % 16 != 0:
        times = imgL.shape[1]//16       
        top_pad = (times+1)*16 -imgL.shape[1]
    else:
        top_pad = 0

    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2]//16                       
        right_pad = (times+1)*16-imgL.shape[2]
    else:
        right_pad = 0    

    imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
    imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

    start_time = time.time()
    pred_disp = compute_SGBM(imgL,imgR)
    print('time = %.2f' %(time.time() - start_time))

    
    if top_pad !=0 and right_pad != 0:
        img = pred_disp[top_pad:,:-right_pad]
    elif top_pad ==0 and right_pad != 0:
        img = pred_disp[:,:-right_pad]
    elif top_pad !=0 and right_pad == 0:
        img = pred_disp[top_pad:,:]
    else:
        img = pred_disp
    
    img = (img*256).astype('uint16')
    img = Image.fromarray(img)
    img.save(f'Test_disparity_2.png')

if __name__ == '__main__':
   main()






