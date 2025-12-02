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

# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

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

def test(imgL,imgR):
    model.eval()

    imgL = imgL.to(device)
    imgR = imgR.to(device)

    with torch.no_grad():
        disp = model(imgL,imgR)

    disp = torch.squeeze(disp)
    pred_disp = disp.data.cpu().numpy()

    return pred_disp


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
        pred_disp = test(imgL,imgR)
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






