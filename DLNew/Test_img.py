from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import math
from models import *
import cv2
from PIL import Image

RESULT_PATH = './DLNew/results'
ORIGIN_PATH = './StereoMatchingTestings/'
RESULT_PREFIX = 'PSMNet_pred_disparity_'
PSNR_RESULT_FILENAME = 'PSNR_results.txt'

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--loadmodel', default='./DLNew/trained/pretrained_model_KITTI2012.tar',
                    help='loading model')                                   
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

device = resolve_device(args.device)
print(f"Using device: {device}")

torch.manual_seed(args.seed)
if device.type == 'cuda':
    torch.cuda.manual_seed(args.seed)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

if device.type == 'cuda' and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to(device)

def _load_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device):
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict) and 'state_dict' in obj:
        state = obj['state_dict']
    elif isinstance(obj, dict):
        # Might be a raw state_dict
        state = obj
    else:
        raise RuntimeError('Unsupported checkpoint format')

    # Strip potential 'module.' prefix from DataParallel
    def strip_module(k):
        return k[7:] if k.startswith('module.') else k
    state = {strip_module(k): v for k, v in state.items()}

    try:
        model.load_state_dict(state, strict=True)
        print('Checkpoint loaded with strict=True')
    except RuntimeError as e:
        print('Strict load failed, retrying with strict=False...')
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f'Missing keys: {list(missing)[:10]}... (+{max(0, len(missing)-10)} more)')
        if unexpected:
            print(f'Unexpected keys: {list(unexpected)[:10]}... (+{max(0, len(unexpected)-10)} more)')

if args.loadmodel is not None and os.path.isfile(args.loadmodel):
    print(f'Load PSMNet from {args.loadmodel}')
    _load_checkpoint(model, args.loadmodel, device)
else:
    if args.loadmodel is not None:
        print(f'Checkpoint not found at {args.loadmodel}, running with random weights.')

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


def apply_model(imgL,imgR):
    model.eval()

    imgL = imgL.to(device)
    imgR = imgR.to(device)

    with torch.no_grad():
        disp = model(imgL,imgR)

    disp = torch.squeeze(disp)
    pred_disp = disp.data.cpu().numpy()

    return pred_disp


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
    img.save(f'{RESULT_PATH}/{RESULT_PREFIX}{parent_dir}.png')
    

def psnr(img1, img2):
    mse = np.mean( ((img1 - img2)) ** 2 )
    if mse == 0:
        return 'INF'
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def calculate_psnr(parent_dir):
        gt_names = f"{ORIGIN_PATH}/{parent_dir}/disp1.png"
        gt_img = np.array(Image.open(gt_names),dtype=float)
        
        pred_names =  f"{RESULT_PATH}/{RESULT_PREFIX}{parent_dir}.png"
        pred_img = np.array(Image.open(pred_names),dtype=float)
        
        # When calculate the PSNR:
        # 1.) The pixels in ground-truth disparity map with '0' value will be neglected.
        # 2.) The left part region (1-250 columns) of view1 is not included as there is no
        #   corresponding pixels in the view5.
    
        [h,l] = gt_img.shape
        gt_img = gt_img[:, 250:l]
        pred_img = pred_img[:, 250:l]
        pred_img[gt_img==0]= 0
    
        peaksnr = psnr(pred_img, gt_img)
        print(f'The Peak-SNR value of {parent_dir} is %0.4f \n', peaksnr)
        return peaksnr


def main():
    test_dirs = ["Art", "Dolls", "Reindeer"]
    os.makedirs(RESULT_PATH, exist_ok=True)
    full_psnr_path = os.path.join(RESULT_PATH, PSNR_RESULT_FILENAME)
    
    for index in range(3):
        leftimg_path = f'./StereoMatchingTestings/{test_dirs[index]}/view1.png'
        rightimg_path = f'./StereoMatchingTestings/{test_dirs[index]}/view5.png'
        
        imgL, imgR, top_pad, right_pad = pre_process_disparity(leftimg_path, rightimg_path)

        start_time = time.time()
        print(f'Processing {test_dirs[index]} disparity ...')
        pred_disp = apply_model(imgL,imgR)
        running_time = time.time() - start_time
        print('time = %.2f' % running_time)

        parent_dir = test_dirs[index]
        post_process_disparity(pred_disp, top_pad, right_pad, parent_dir)
        
        psnr_value =calculate_psnr(parent_dir)
        with open(full_psnr_path, 'a', encoding='utf-8') as f:
            f.write(f'The Peak-SNR value of {parent_dir}: {psnr_value}. Running time: {running_time}\n')

if __name__ == '__main__':
   main()



