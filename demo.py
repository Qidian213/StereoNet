import argparse 
import os
import torch
import cv2 as cv
import numpy as np
import torchvision
from PIL import Image
from collections import OrderedDict
from models.StereoNet8Xmulti import StereoNet
from dataloader import preprocess

parser = argparse.ArgumentParser(description='StereoNet with Flyings3d')
parser.add_argument('--maxdisp', type=int, default=192, help='maxium disparity')
parser.add_argument('--datapath', default='/data/senseflow', help='datapath')
parser.add_argument('--save_path', type=str, default='results',help='the path of saving checkpoints and log')
parser.add_argument('--resume', type=str, default='results/checkpoint.pth', help='resume path')
parser.add_argument('--stages', type=int, default=4, help='the stage num of refinement')
parser.add_argument('--gpu', default='0', type=str, help='GPU ID')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    __normalize = {'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]}
    processed = preprocess.get_transform(augment=False, normalize=__normalize)

    model = StereoNet(k=args.stages-1, r=args.stages-1, maxdisp=args.maxdisp)
    checkpoint = torch.load('results/checkpoint.pth')
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    left_file = '/home/zzg/Datasets/flower_storm_x2/left/0000.png'
    right_file = '/home/zzg/Datasets/flower_storm_x2/right/0000.png'

    imgL = Image.open(left_file).convert('RGB')
    imgR = Image.open(right_file).convert('RGB')
    imgL = processed(imgL).unsqueeze(0)
    imgR = processed(imgR).unsqueeze(0)
    imgL = imgL.float().to(device)
    imgR = imgR.float().to(device)

    outputs = model(imgL, imgR)
    outputs = [torch.squeeze(output, 1) for output in outputs]
    
    #vis
    _, H, W = outputs[0].shape
    all_results = torch.zeros((len(outputs), 1, H, W))
    for j in range(len(outputs)):
        all_results[j, 0, :, :] = outputs[j][0, :, :]/255.0
    torchvision.utils.save_image(all_results, 'result.png')

if __name__ == '__main__':
    main()
