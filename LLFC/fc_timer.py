# -------------------------
# FEATURE CONCATENTTION
# -------------------------

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import pandas as pd 
import ast
import os
import logging
import pickle
import numpy as np


class DataSet(Dataset):
    def __init__(self, image_list, image_dir, feature_pickle, device, skip_frame=False):
        self.imgs = pd.read_csv(image_list) 
        self.features = pickle.load( open(feature_pickle, "rb" ))
        self.image_dir = image_dir
        self.device = device
        self.skip = skip_frame

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        row = self.imgs.iloc[index, :]
        video = row[0]
        frames =  ast.literal_eval(row[1])
        labels = ast.literal_eval(row[2])
        if self.skip:
            # KEEP EVERY OTHER FRAME
            frames = frames[::2]
            labels = labels[::2]
        files = [video + "/%03d.jpg"%x for x in frames]
        # concatenate features into one large vector
        features = torch.cat([torch.Tensor(self.features[f]) for f in files], 0)
        label = torch.tensor(float(1 in labels))
        if self.device:
            features = features.to(self.device)
            label = label.to(self.device)
        sample = {'features': features, 'filename': files, 'label': label}
        return sample

class LinearLayer(nn.Module):
    def __init__(self, feature_dim, num_frames):
        super(LinearLayer, self).__init__()
        self.device = device
        self.linear = nn.Linear(feature_dim * num_frames, 1)

    def forward(self, feature_vector):
        logic = self.linear(feature_vector)
        m = nn.Sigmoid()
        prob = m(logic)
        return prob 

def evaluate(model, device, loader):

    # set to eval mode
    model.eval()
    # save file names and predictions
  
    # enumerate over data loader
    #logging.info(output_dir)
    start = timer()
    #logging.info('START ' + start)
    #print('START ' + datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    for i in range(args.trials):
        for _, data_batch in enumerate(loader):
            x_batch = data_batch['features']
            y_batch = data_batch['label'] 
            prob = model(x_batch)
            pred_label = int(prob.data[0].cpu().numpy()> 0.5)
    end = timer()
    #ogging.info('END ' + end)
    duration = end - start
    logging.info("{}: {}".format(output_dir, duration/args.trials))
    print("average ", duration/args.trials)

if __name__ == '__main__':
    import argparse
    from timeit import default_timer as timer

    parser = argparse.ArgumentParser("Test dataset")
    parser.add_argument("--image_dir", type=str, default="data", help="image dir")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument("--seq_length", type=int, default=12, help="4 | 6 | 8 | 10 | 12")
    parser.add_argument("--type", type=str, help="resnet | basic")
    parser.add_argument("--checkpoint", type=int, default=49, help="checkpoint epoch")
    parser.add_argument("--skip_frame", type=int, default=0, help="1 = skip")
    parser.add_argument("--trials", type=int, default=100, help="1 = skip")
    parser.add_argument("--add_args", type=str, default="SINGLE", help="SINGLE | MEDIUM")

    args = parser.parse_args()
    #print (args)
    output_dir = os.path.join("models", "FC_{}_{}frame_{}".format(args.type, args.seq_length, args.skip_frame))
    if not os.path.exists(output_dir):
        print ("making", output_dir)
        os.makedirs(output_dir)
 
    log_fn = os.path.join("FC_timer.log")
    fmt = "%(message)s"
    logging.basicConfig(filename=log_fn, format=fmt, level=logging.INFO)

    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda:"+str(args.gpu_id))

    lengths = {
        "resnet": 512,
        "basic": 256
    }

    features = {
        "resnet": "FEATURES/test3_resnet.p",
        "basic": "FEATURES/test3_basic.p"
    }
    feature_pickle = features[args.type]
    print ('making dataset with', feature_pickle)

    csv_file = os.path.join("CSVs", "testfinal_{}{}.csv".format(args.seq_length, args.add_args))

    print ("using CSV", csv_file)
    #ogging.info("CSV: {}, PICKLE: {}".format(csv_file,feature_pickle ))
    dataset = DataSet(csv_file, args.image_dir, feature_pickle, device, skip_frame = bool(args.skip_frame))

    l = lengths[args.type]
    num_frames = args.seq_length / (2 ** args.skip_frame)
    #print ('initializing model with {}-length vectors and {} frames'.format(l, num_frames))
    model = LinearLayer(l, num_frames)
    if device:
        model.to(device)
    loader = DataLoader(dataset, 1, shuffle=False, drop_last=True, num_workers=0)


    #print("Starting eval ...")
    check_point_fn = os.path.join(output_dir, "ckpt_{}_0.pth".format(args.checkpoint))
    #logging.info("EVALUATING {}".format(check_point_fn))
    #print ("Loading", check_point_fn)
    ckpt = torch.load(check_point_fn)
    model.load_state_dict(ckpt)  
    evaluate(model, device, loader)
   
 