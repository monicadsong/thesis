import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import torchvision.transforms as transforms
import logging

from RNN import DataSet
class LSTMSequence(nn.Module):
    def __init__(self, feature_dim, hidden_dim, target_size, layer_cnt=1, device=None):
        super(LSTMSequence, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_cnt = layer_cnt
        self.device = device
        self.lstm = nn.LSTM(input_size=feature_dim, 
                hidden_size=hidden_dim,
                num_layers=self.layer_cnt)
        self.linear = nn.Linear(hidden_dim, target_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        h = torch.zeros(self.layer_cnt, 1, self.hidden_dim, device=device)
        c = torch.zeros(self.layer_cnt, 1, self.hidden_dim, device=device)
        return (h, c)

    def forward(self, frame_sequence):
    # only care the final stage
        #print ("frame_sequence.shape", frame_sequence.shape)
        for f in frame_sequence[0]:
            #print (f.view(1,1,-1).shape)
            lstm_out, self.hidden = self.lstm(f.view(1,1,-1), self.hidden)
        logic = self.linear(lstm_out.view(-1))
        prob = F.sigmoid(logic)
        #print (prob)
        return prob 

def evaluate(model, device, loader):
    model.eval()
    correct = 0
    # enumerate over data loader
    for _, data_batch in enumerate(loader):
        x_batch = data_batch['features']
        y_batch = data_batch['label'] 
        model.hidden = model.init_hidden()
        prob = model(x_batch)
        m = prob.data[0].cpu().numpy()
        if int(m > 0.5) == y_batch:
            correct += 1
    
    print("accuracy: ", correct, len(loader), float(correct)/len(loader))

import argparse
def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument("--csv_file", type=str, help="dataset to evaluate")
    parser.add_argument("--image_dir", type=str, help="dataset to evaluate")
    parser.add_argument("--model", type=str, help="model to evaluate")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = _parse_args()
    fmt = "%(message)s"
    # log_fn = os.path.join(args.log_file)
    # logging.basicConfig(filename=log_fn, format=fmt, level=logging.INFO)
    # logging.info(args)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    tran= transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize,
              ])
    print ('loading model')
    model = torch.load(args.model)
    print ("done loading") 
    
    print ('making dataset')
    print ('loading CNN')
    CNN = torch.load("ckpt_10_0.pth") 
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda:"+str(args.gpu_id))
        model.to(device)
        CNN.to(device)
    dataset = DataSet(args.csv_file, args.image_dir, CNN, device, tran)
    loader = DataLoader(dataset, 1, shuffle=True, drop_last=True, num_workers=0)
    print ("beginning evaluation")
    evaluate(model, device, loader)


