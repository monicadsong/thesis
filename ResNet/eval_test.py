#----------------
# Evaluate on the test sequences
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

import logging
import argparse
import ast
from PIL import Image

class DataSet(Dataset):
    def __init__(self, image_list, image_dir, device, transform=None):
        self.transform = transform
        self.imgs = pd.read_csv(image_list) 
        self.image_dir = image_dir
        self.device = device

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        row = self.imgs.iloc[index, :]
        video = row[0]
        frames = ast.literal_eval(row[1])
        labels = ast.literal_eval(row[2])
        label = torch.tensor(float(1 in labels))
        files = [os.path.join(self.image_dir, video, "%03d.jpg"%x) for x in frames]
        #print ("files", files)
        imgs = [Image.open(fn) for fn in files]
        if self.transform is not None:
            imgs = [self.transform(i) for i in imgs]
        imgs = torch.stack(imgs)
        #print (imgs.shape)
        if self.device is not None:
            imgs = imgs.to(self.device)
            label = label.to(self.device)
        return {"imgs": imgs, "label": label, "filename": video + "/%03d.jpg"%frames[0]}


def eval(loader, model):
    model.eval()
    fns = [] 
    y_true = []
    y_pred = []
    correct = 0
    FN = 0
    FP = 0 
    TN = 0
    TP = 0
# the size length of 
    for _, sequence in enumerate(loader):
        x_batch = torch.squeeze(sequence['imgs'])
        y_batch = sequence['label'] 
        logit = model(x_batch) 
        prob = torch.nn.functional.softmax(logit, dim=1)
        y_true_n = y_batch.data.cpu().numpy()
        y_pred_n = np.mean(prob.data.cpu().numpy()[:,1])
        y_true.append(y_true_n[0])
        #print (y_pred_n, y_pred)
        y_pred.append(y_pred_n)
        fns += sequence['filename'] 
        #print (prob, y_batch.data)
        pred_label = int(y_pred_n > 0.5)
        #print(pred_label, y_batch)
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
    results["probs"] = y_pred
    results["ground_true"] = y_true
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
    print("TP/TN/FP/FN: {}/{}/{}/{}\n{}, {}, {}, {}"
        .format(TP, TN, FP, FN, a, p ,r,f))
    return results
  
def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="data", help="directory that contains image files")
    parser.add_argument("--seq_length", type=int, default=12, help="4 | 6 | 8 | 10 | 12")
    parser.add_argument("--add_args", type=str ,default="", help = "additional CSV description | SHORT | SINGLE")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument("--output_dir", type=str, default="results", help="log result filename")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    import time
    args = _parse_args()
    fmt = "%(message)s"
    log_fn = os.path.join(args.output_dir, "valid_{}{}.log".format(args.seq_length, args.add_args))
    fmt = "%(message)s"
    logging.basicConfig(filename=log_fn, format=fmt, level=logging.INFO)
    logging.info(args)

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
            transforms.Resize((model.input_size, model.input_size)),
            transforms.ToTensor(),
            normalize,
            ])
    
    csv_file = os.path.join("testCSVs", "valid3_{}frame{}.csv".format(args.seq_length, args.add_args))

    print ("Using CSV", csv_file)
    dataset = DataSet(csv_file, args.img_dir, device, transform=tran) 
    logging.info("Total Sample count: {}".format(len(dataset)))


    logging.info('Starting evaluation at ' + time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()))

    loader = DataLoader(dataset, 1, shuffle=False, drop_last=False, num_workers=0)
    result = eval(loader, model)

    result_df = pd.DataFrame(data=result)
    ppn = os.path.join(args.output_dir, "valid_{}{}_result.csv".format(args.seq_length, args.add_args)) 
    result_df.to_csv(ppn)

    logging.info('Ending evaluation at ' + time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()))

