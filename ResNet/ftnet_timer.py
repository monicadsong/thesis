#----------------
# Timer on the validation for ResNet
#-----------------
import sys, os
import numpy as np
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import DataSet
from fine_tune_net import FineTuneNet

import argparse
import ast
from PIL import Image



def eval(loader, model, device):
    model.eval()
    for i, data_batch in enumerate(loader):
        x_batch = data_batch['image']
        y_batch = data_batch['label']
        if device is not None:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
        logit = model(x_batch) 
        prob = torch.nn.functional.softmax(logit, dim=1)
        pred_label = int(prob.data[:,1].cpu().numpy()> 0.5)
  
def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="data", help="directory that contains image files")
    parser.add_argument("--add_args", type=str ,default="", help = "additional CSV description | SHORT | SINGLE")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument("--output_dir", type=str, default="results", help="log result filename")
    parser.add_argument("--trials", type=int, default=100, help="num trials")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    from timeit import default_timer as timer
    args = _parse_args()

    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda:"+str(args.gpu_id))
    print ('loading model')
    model = torch.load("test_resnet.pth")  
    if device:
        model.to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    tran = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize,
            ])
    
    csv_file = "CSVs/test_50.csv"

    dataset = DataSet(csv_file, args.img_dir, transform=tran) 
    loader = DataLoader(dataset, 1, shuffle=False, drop_last=False, num_workers=0)
    start = timer()
    for i in range(args.trials):
        eval(loader, model, device)
    end = timer()
    #ogging.info('END ' + end)
    duration = end - start
    print("average ", duration/args.trials, "total", duration)