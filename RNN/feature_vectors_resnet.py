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
from fine_tune_net import FineTuneNet

# create the feature vector dataset
class DataSet(Dataset):
    def __init__(self, image_list, image_dir, net, device, transform=None):
        self.model = net
        self.transform = transform
        self.layer = net.model._modules.get('avgpool')
        self.imgs = pickle.load( open(image_list, "rb" ) )
        self.image_dir = image_dir
        self.device = device

    def __len__(self):
        return len(self.imgs)
    
    def _get_features(self, filename):
        self.model.eval()
        img = Image.open(filename)
        if self.transform is not None:
            img = self.transform(img)
        if self.device is not None:
            img= img.to(self.device)
        my_embedding = torch.zeros(512)
        def copy_data(m, i, o):
            my_embedding.copy_(o.data.squeeze())
        h = self.layer.register_forward_hook(copy_data)
        self.model(torch.unsqueeze(img, 0))
        h.remove()
        return my_embedding

    def __getitem__(self, index):
        frame =  self.imgs[index]
        filename = self.image_dir + "/" + frame
        features = self._get_features(filename)
        return {"frame": frame, "features": features}

if __name__ == '__main__':
    import argparse
    from torchvision.transforms import Compose
    import torchvision.transforms as transforms
    parser = argparse.ArgumentParser("Test dataset")
    parser.add_argument("--pickle", type=str, default="input.csv", help="csv filename")
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
    print ('loading resnet')
    net = torch.load("resnet.pth") 
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda:"+str(args.gpu_id))
        net.to(device)
    print ('making dataset ')
    dataset = DataSet(args.pickle, args.image_dir, net, device, tran)
    loader = DataLoader(dataset, 1, shuffle=False, drop_last=True, num_workers=0)
    feature_dict = {}
    for i,image in enumerate(loader):
        fn = image['frame']
        if i % 100 == 0:
            print (fn)
        features =image['features']
        feature_dict[fn[0]] = features.cpu().detach().numpy()[0]

    with open(args.feature_pickle, 'wb') as handle:
        pickle.dump(feature_dict, handle, protocol=2)





 