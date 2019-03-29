# -------------------------
# FOR USE WITH THE RESNET18
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
from PIL import Image
import os
import logging
import pickle
import numpy as np


class DataSet(Dataset):
    def __init__(self, image_list, image_dir, device):
        self.imgs = pd.read_csv(image_list) 
        name = image_list.split('/')[1].split('.')[0]
        print (name)
        self.features = pickle.load( open("FEATURES/test2_frames_RESNETFEATURES.p", "rb" ))
        self.image_dir = image_dir
        self.device = device

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        row = self.imgs.iloc[index, :]
        video = row[0]
        frames =  ast.literal_eval(row[1])
        files = [video + "/%03d.jpg"%x for x in frames]
        features = torch.tensor([self.features[f] for f in files])
        labels = ast.literal_eval(row[2])
        label = torch.tensor(float(1 in labels))
        if self.device:
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

def train(RNN, feature_dim, hidden_size, device, dataloader, lr):
    RNN.train()
    #loss_func = F.binary_cross_entropy
    #logging.info("using binary cross entropy ")
    loss_func = nn.MSELoss(size_average=False)
    logging.info("using MSE")
    #logging.info("using SGD with learning rate {}".format(lr))
    #optimizer = optim.SGD(RNN.parameters(), lr=lr)
    logging.info("using Adam with learning rate {}".format(lr))
    optimizer = torch.optim.Adam(RNN.parameters(), lr=lr)
    #logging.info("using RMS with learning rate {}".format(lr))
    #optimizer = torch.optim.RMSprop(RNN.parameters(), lr=0.05)
    EPOCHS = 40
    SKIP = 2000
    for epoch in range(EPOCHS):
        print ('BEGINNING EPOCH', epoch)
        for i, data_batch in enumerate(loader):
            x_batch = data_batch['features']
            y_batch = data_batch['label'] 
            RNN.zero_grad()
            RNN.hidden = RNN.init_hidden()
            
            prob = RNN(x_batch)
            loss = loss_func(prob, y_batch)
            if i %SKIP == 0:
                print (loss)
                logging.info("LOSS {} {}".format(epoch, loss.item()))
                check_point_fn = "ckpt_" + str(epoch) + "_" + str(i) + ".pth"
                model_fn = os.path.join(args.output_dir, check_point_fn)
                logging.info("SAVING {}".format(model_fn))
                torch.save(RNN.state_dict(), model_fn)
            loss.backward()
            optimizer.step()

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
    for _, data_batch in enumerate(loader):
        x_batch = data_batch['features']
        y_batch = data_batch['label'] 
        model.hidden = model.init_hidden()
        if device:
            model.hidden = [x.to(device) for x in model.hidden]
            x_batch = x_batch.to(device)
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
    results = {}
    results["probs"] = y_pred.tolist() 
    results["ground_true"] = y_true.tolist()
    results["fns"] = fns 

    
    print("true positives: ", TP)
    print("true negatives: ", TN)
    print("false positives: ", FP)
    print("false negatives: ", FN)
    print("accuracy: ", correct, len(loader), float(correct)/len(loader))
    p = float(TP)/(TP + FP)
    r =  float(TP)/ (TP + FN)
    print ("precision: ", p)
    print ("recall: ", r)
    print ("F1: ", 2 * p * r / (p + r))
    return results

if __name__ == '__main__':
    import argparse
    from torchvision.transforms import Compose
    import torchvision.transforms as transforms
    parser = argparse.ArgumentParser("Test dataset")
    parser.add_argument("--csv_file", type=str, default="input.csv", help="csv filename")
    parser.add_argument("--image_dir", type=str, default="data", help="image dir")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument("--log_file", type=str, default="train.log", help="save train log")
    parser.add_argument("--output_dir", type=str, default="models", help="save model dir")
    parser.add_argument("--learning_rate", type=float, default=0.005, help="learning rate")
    parser.add_argument("--check_point_fn", type=str, default=None, help="check point filename")

    args = parser.parse_args()
    fmt = "%(message)s"
    log_fn = os.path.join(args.log_file)
    logging.basicConfig(filename=log_fn, format=fmt, level=logging.INFO)
    logging.info(args)


    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda:"+str(args.gpu_id))
    print ('making dataset')
    dataset = DataSet(args.csv_file, args.image_dir, device)

    feature_dim = 512
    hidden_size = 64
    logging.info("using {} hidden layers".format(hidden_size))
    RNN = LSTMSequence(feature_dim, hidden_size, 1, layer_cnt=2, device=device) 
    if device:
        RNN.to(device)
    loader = DataLoader(dataset, 1, shuffle=True, drop_last=True, num_workers=0)

    if args.check_point_fn is None:
        train(RNN, feature_dim, hidden_size, device, loader, args.learning_rate)
        print ("EVALUATING MODEL AFTER LAST EPOCH")
        print ("---------------------------------")
        evaluate(RNN,device,loader)
        print ("SAVING FINISHED MODEL")
        print ("---------------------------------")
        torch.save(RNN.state_dict(), "{}/finished.pth".format(args.output_dir))
    else:
        print("Starting eval ...")
        ckpt = torch.load(args.check_point_fn)
        RNN.load_state_dict(ckpt)  
        evaluate(RNN, device, loader)
 