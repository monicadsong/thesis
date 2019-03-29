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
    fns = [] 
    y_true = np.empty((0,)) 
    y_pred = np.empty((0,)) 
    correct = 0
    FN = 0
    FP = 0 
    TN = 0
    TP = 0
    # enumerate over data loader
    logging.info('START ' + datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    print('START ' + datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    for _, data_batch in enumerate(loader):
        x_batch = data_batch['features']
        y_batch = data_batch['label'] 
        model.hidden = model.init_hidden()
        prob = model(x_batch)
        y_true_n = y_batch.data.cpu().numpy()
        y_pred_n = prob.data.cpu().numpy()

        #print (y_pred_n.shape, y_pred.shape)
        y_true = np.concatenate((y_true, y_true_n))
        y_pred = np.concatenate((y_pred, y_pred_n))
        fns += data_batch['filename'][0]

        pred_label = int(prob.data[0].cpu().numpy()> 0.5)
        #print (prob, y_batch.data[0])
        if pred_label == y_batch:
            correct += 1
            if pred_label == 1:
                TP += 1
            else:
                TN += 1
        else:
            if pred_label == 1:
                FP += 1
            else:
                FN += 1
    logging.info('END ' + datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    print('END ' + datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    results = {}
    results["probs"] = y_pred.tolist() 
    results["ground_true"] = y_true.tolist()
    results["fns"] = fns 

   
    logging.info("TP/TN/FP/FN: {}/{}/{}/{}".format(TP, TN, FP, FN))
    a = float(correct)/len(loader)
    logging.info("Accuracy: {}".format(a))
    p = float(TP)/(TP + FP)
    r =  float(TP)/ (TP + FN)
    f = 2 * p * r / (p + r)
    logging.info("Precision: {}".format(p))
    logging.info("Recall: {}".format(r))
    logging.info("F1: {}".format(f))
    print("{}: TP/TN/FP/FN: {}/{}/{}/{}\n{}, {}, {}, {}".format(check_point_fn, TP, TN, FP, FN, a, p ,r,f))
    return results
 

if __name__ == '__main__':
    
    feature_dim = 512
    hidden_size = 64
    print ("using RNN with feature dim {} and {} hidden layers".format(feature_dim, hidden_size))
    RNN = LSTMSequence(feature_dim, hidden_size, 1, layer_cnt=2, device=device) 
    npd = sorted(dict(RNN.named_parameters()).items()) 
    param_cnt = 0
    for d in npd:
        print(d[0], d[1].shape, d[1].device) 
        param_cnt += np.prod(d[1].shape)
    print("Model param_cnt: ", param_cnt)
   
 