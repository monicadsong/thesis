# -------------
# create the feature vector dataset using BASIC NET
# -------------

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import pandas as pd 
import ast
from PIL import Image
import os
import logging
import numpy as np
import pickle


class DataSet(Dataset):
    def __init__(self, image_list, image_dir, model, device, transform=None):
        self.model = model
        self.transform = transform
        self.imgs = pickle.load( open(image_list, "rb" ) )
        self.image_dir = image_dir
        self.device = device

    def __len__(self):
        return len(self.imgs)
    
    def _get_features(self, images):
        self.model.eval()
        _, feature_vector = self.model(images)
        return feature_vector

    def __getitem__(self, index):
        frame =  self.imgs[index]
        filename = self.image_dir + "/" + frame
        img = Image.open(filename)
        if self.transform is not None:
            img = self.transform(img)
        if self.device is not None:
            img= img.to(self.device)
        features = self._get_features(torch.unsqueeze(img, 0))
        return {"frame": frame, "features": features}

if __name__ == '__main__':
    import argparse
    from torchvision.transforms import Compose
    import torchvision.transforms as transforms
    parser = argparse.ArgumentParser("Save feature vectors")
    parser.add_argument("--pickle", type=str, help="pickle of image names")
    parser.add_argument("--feature_pickle", type=str, default="features.p",  help="save features pickle")
    parser.add_argument("--image_dir", type=str, default="data", help="image dir")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")

    args = parser.parse_args()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    tran= transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize,
              ])
    print ('loading CNN')
    model = torch.load("ckpt_10_0.pth") 
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda:"+str(args.gpu_id))
        model.to(device)
    print ('making dataset ')
    dataset = DataSet(args.pickle, args.image_dir, model, device, tran)
    loader = DataLoader(dataset, 1, shuffle=False, drop_last=True, num_workers=0)
    feature_dict = {}
    for i,image in enumerate(loader):
        fn = image['frame']
        features =image['features']
        if i % 100 == 0:
            print (fn)
        feature_dict[fn[0]] = features.cpu().detach().numpy()[0][0]

    with open(args.feature_pickle, 'wb') as handle:
        pickle.dump(feature_dict, handle, protocol=2)





 