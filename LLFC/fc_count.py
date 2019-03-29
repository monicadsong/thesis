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
        self.linear = nn.Linear(feature_dim * num_frames, 1)

    def forward(self, feature_vector):
        logic = self.linear(feature_vector)
        m = nn.Sigmoid()
        prob = m(logic)
        return prob 

def train(model, dataloader, lr):
    model.train()
    #loss_func = F.binary_cross_entropy
    #logging.info("using binary cross entropy ")
    loss_func = nn.MSELoss(size_average=False)
    logging.info("using MSE")
    #logging.info("using SGD with learning rate {}".format(lr))
    #optimizer = optim.SGD(model.parameters(), lr=lr)
    logging.info("using Adam with learning rate {}".format(lr))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #logging.info("using RMS with learning rate {}".format(lr))
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=0.05)
    EPOCHS = 50
    SKIP = 2000
    for epoch in range(EPOCHS):
        print ('BEGINNING EPOCH', epoch)
        for i, data_batch in enumerate(loader):
            x_batch = data_batch['features']
            y_batch = data_batch['label'] 
            model.zero_grad()         
            prob = model(x_batch)
            loss = loss_func(prob, y_batch)
            #print (prob.data)
            if i %SKIP == 0:
                print (loss)
                logging.info("LOSS {} {}".format(epoch, loss.item()))
                check_point_fn = "ckpt_" + str(epoch) + "_" + str(i) + ".pth"
                model_fn = os.path.join(output_dir, check_point_fn)
                #logging.info("SAVING {}".format(model_fn))
                torch.save(model.state_dict(), model_fn)
            loss.backward()
            optimizer.step()

    
def evaluate(model, device, loader):
    # set to eval mode
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
    for _, data_batch in enumerate(loader):
        x_batch = data_batch['features']
        y_batch = data_batch['label'] 
        prob = model(x_batch)
        y_true_n = y_batch.data.cpu().numpy()
        y_pred_n =prob.data[0].cpu().numpy()
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


    
    logging.info("TP/TN/FP/FN: {}/{}/{}/{}".format(TP, TN, FP, FN))
    a = float(correct)/len(loader)
    logging.info("Accuracy: {}".format(a))
    p = float(TP)/(TP + FP)
    r =  float(TP)/ (TP + FN)
    f = 2 * p * r / (p + r)
    logging.info("Precision: {}".format(p))
    logging.info("Recall: {}".format(r))
    logging.info("F1: {}".format(f))
    print("{}:{}"
        .format(check_point_fn, 1 - a))


if __name__ == '__main__':
  
    l = 256
    num_frames = 10
    #print ('initializing model with {}-length vectors and {} frames'.format(l, num_frames))
    model = LinearLayer(l, num_frames)
    npd = sorted(dict(model.named_parameters()).items()) 
    param_cnt = 0
    for d in npd:
        print(d[0], d[1].shape, d[1].device) 
        param_cnt += np.prod(d[1].shape)
    print("Model param_cnt: ", param_cnt)
    
 