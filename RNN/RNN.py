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

def train(RNN, feature_dim, hidden_size, device, dataloader, lr):
    RNN.train()
    #loss_func = F.binary_cross_entropy
    #logging.info("using binary cross entropy ")
    loss_func = nn.MSELoss(size_average=False)
    logging.info("using MSE")
    #logging.info("using SGD with learning rate {}".format(lr))
    #print (RNN.parameters())
    #optimizer = optim.SGD(RNN.parameters(), lr=lr)
    logging.info("using Adam with learning rate {}".format(lr))
    optimizer = torch.optim.Adam(RNN.parameters(), lr=lr)
    #logging.info("using RMS with learning rate {}".format(lr))
    #optimizer = torch.optim.RMSprop(RNN.parameters(), lr=0.05)
    EPOCHS = 50
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
                model_fn = os.path.join(output_dir, check_point_fn)
                #logging.info("SAVING {}".format(model_fn))
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

    
   
    logging.info("TP/TN/FP/FN: {}/{}/{}/{}".format(TP, TN, FP, FN))
    a = float(correct)/len(loader)
    logging.info("Accuracy: {}".format(a))
    p = float(TP)/(TP + FP)
    r =  float(TP)/ (TP + FN)
    f = 2 * p * r / (p + r)
    logging.info("Precision: {}".format(p))
    logging.info("Recall: {}".format(r))
    logging.info("F1: {}".format(f))
    print("{}: TP/TN/FP/FN: {}/{}/{}/{}\n{}, {}, {}, {}"
        .format(check_point_fn, TP, TN, FP, FN, a, p ,r,f))
    return results
 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Test dataset")
    parser.add_argument("--mode", type=str, default="train", help="train | valid | test")
    parser.add_argument("--image_dir", type=str, default="data", help="image dir")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument("--seq_length", type=int, default=12, help="4 | 6 | 8 | 10 | 12")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="learning rate")
    parser.add_argument("--type", type=str, help="resnet | basic")
    parser.add_argument("--checkpoint", type=int, default=None, help="checkpoint epoch")
    parser.add_argument("--skip_frame", type=int, default=0, help="1 = skip")
    parser.add_argument("--add_args", type=str ,default="", help = "additional CSV description | SHORT")

    args = parser.parse_args()
    #print (args)
    output_dir = os.path.join("models", "RNN_{}_{}frame_{}".format(args.type, args.seq_length, args.skip_frame))
    if not os.path.exists(output_dir):
        print ("making", output_dir)
        os.makedirs(output_dir)

    if args.checkpoint:
        log_fn = os.path.join(output_dir, "eval.log")
    else:   
        log_fn = os.path.join(output_dir, "train.log")
    fmt = "%(message)s"
    logging.basicConfig(filename=log_fn, format=fmt, level=logging.INFO)
    logging.info(args)

    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda:"+str(args.gpu_id))

    lengths = {
        "resnet": 512,
        "basic": 256
    }

    features = {
        "resnet": "FEATURES/{}3_resnet.p".format(args.mode),
        "basic": "FEATURES/{}3_basic.p".format(args.mode)
    }
    feature_pickle = features[args.type]
    #print ('making dataset with', feature_pickle)

    csv_file = os.path.join("CSVs", "{}3_{}frame{}.csv".format(args.mode, args.seq_length, args.add_args))
    #print ("using CSV", csv_file)
    logging.info("CSV: {}, PICKLE: {}".format(csv_file,feature_pickle ))
    dataset = DataSet(csv_file, args.image_dir, feature_pickle, device, skip_frame = bool(args.skip_frame))

    feature_dim = lengths[args.type]
    hidden_size = 64
    logging.info("using RNN with feature dim {} and {} hidden layers".format(feature_dim, hidden_size))
    RNN = LSTMSequence(feature_dim, hidden_size, 1, layer_cnt=2, device=device) 
    if device:
        RNN.to(device)
    loader = DataLoader(dataset, 1, shuffle=True, drop_last=True, num_workers=0)

    if args.checkpoint is None:
        train(RNN, feature_dim, hidden_size, device, loader, args.learning_rate)
        print ("EVALUATING MODEL AFTER LAST EPOCH")
        print ("---------------------------------")
        evaluate(RNN,device,loader)
        print ("SAVING FINISHED MODEL")
        print ("---------------------------------")
        torch.save(RNN.state_dict(), os.path.join(output_dir,"finished.pth"))
    else:
        #print("Starting eval ...")
        check_point_fn = os.path.join(output_dir, "ckpt_{}_0.pth".format(args.checkpoint))
        #print("Loading ", check_point_fn)
        logging.info("EVALUATING {}".format(check_point_fn))
        ckpt = torch.load(check_point_fn)
        RNN.load_state_dict(ckpt)  
        result = evaluate(RNN, device, loader)
        result_df = pd.DataFrame(data=result)
        ppn = os.path.join(output_dir, "{}_{}_results.csv".format(args.mode, args.checkpoint)) 
        result_df.to_csv(ppn)

 