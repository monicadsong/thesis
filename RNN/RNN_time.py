#--------------
# Variable Length Frame Sequence RNN
#----------------

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
        frames = ast.literal_eval(row[1])
        labels = ast.literal_eval(row[2])
        if self.skip:
            # KEEP EVERY OTHER FRAME
            frames = frames[::2]
            labels = labels[::2]
        files = [video + "/%03d.jpg"%x for x in frames]
        features = torch.tensor([self.features[f] for f in files])
        label = torch.tensor(float(1 in labels))
        if self.device is not None:
            features = features.to(self.device)
            label = label.to(self.device)
        sample = {'features': features, 'filename': files, 'label': label}
        return sample

class LSTMSequence(nn.Module):
    def __init__(self, feature_dim, hidden_dim, target_size, layer_cnt=1, device=None):
        super(LSTMSequence, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_cnt = layer_cnt
        self.device = device
        self.lstm = nn.LSTM(input_size=feature_dim, 
                hidden_size=hidden_dim,
                num_layers=self.layer_cnt, batch_first=True)
        self.linear = nn.Linear(hidden_dim, target_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        h = torch.zeros(self.layer_cnt, 1, self.hidden_dim)
        c = torch.zeros(self.layer_cnt, 1, self.hidden_dim)
        if self.device is not None:
            h = h.to(self.device)
            c = c.to(self.device)
        return (h, c)

    def forward(self, frame_sequence):
    # only care the final stage
        lstm_out, self.hidden = self.lstm(frame_sequence, self.hidden)
        last_one = lstm_out[:, -1, :]
        logic = self.linear(last_one.view(-1))
        prob = torch.sigmoid(logic)
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
            model.hidden = model.init_hidden()
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
    output_dir = os.path.join("models", "RNN_{}_{}frame_{}".format(args.type, args.seq_length, args.skip_frame))
    if not os.path.exists(output_dir):
        print ("making", output_dir)
        os.makedirs(output_dir)
    #print (output_dir)

    log_fn = os.path.join("RNN_timer.log")
    fmt = "%(message)s"
    logging.basicConfig(filename=log_fn, format=fmt, level=logging.INFO)
    #logging.info(output_dir)

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
    #logging.info("CSV: {}, PICKLE: {}".format(csv_file,feature_pickle ))
    dataset = DataSet(csv_file, args.image_dir, feature_pickle, device, skip_frame = bool(args.skip_frame))

    feature_dim = lengths[args.type]
    hidden_size = 64
    #logging.info("using RNN with feature dim {} and {} hidden layers".format(feature_dim, hidden_size))
    RNN = LSTMSequence(feature_dim, hidden_size, 1, layer_cnt=2, device=device) 
    if device:
        RNN.to(device)
    loader = DataLoader(dataset, 1, shuffle=False, drop_last=True, num_workers=0)
    check_point_fn = os.path.join(output_dir, "ckpt_{}_0.pth".format(args.checkpoint))

    ckpt = torch.load(check_point_fn)
    RNN.load_state_dict(ckpt)  
    evaluate(RNN, device, loader)
 