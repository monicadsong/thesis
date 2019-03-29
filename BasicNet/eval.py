import sys, os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import DataSet
from vanilla_net import BasicNet

import logging
import argparse
import pickle

def _eval_one_epoch(loader, model, device):
  model.eval()
  fns = [] 
  y_true = np.empty((0,)) 
  y_pred = np.empty((0,)) 
  features = np.empty((0,256)) 
  for i, data_batch in enumerate(loader):
    x_batch = data_batch['image']
    y_batch = data_batch['label'] 
    
    if device is not None:
      x_batch = x_batch.to(device)
      y_batch = y_batch.to(device)
 
    logit, feature_vector = model(x_batch)
    prob = torch.nn.functional.softmax(logit, dim=1)
    y_true_n = y_batch.data.cpu().numpy()
    y_pred_n = prob.data.cpu().numpy()[:,1]
    features_n = feature_vector.data.cpu().numpy()
    #print (features_n)
    print (features.shape, features_n.shape)
    y_true = np.concatenate((y_true, y_true_n))
    y_pred = np.concatenate((y_pred, y_pred_n))
    features = np.concatenate((features, features_n))

    fns += data_batch['filename'] 

  yy = (y_pred > 0.5).astype(np.int32)
  correct_cnt = (yy == y_true.astype(np.int32)).sum()   

  results = {}
  results["probs"] = y_pred.tolist() 
  results["ground_true"] = y_true.tolist()
  results["fns"] = fns 
  results["features"] = [f.tolist() for f in features]

  logging.info("correct count: {}, total count: {}, precision: {}".format(
		correct_cnt, len(yy), 
                float(correct_cnt)/len(yy))
               )

  return results
  
def _parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("img_list_fn", help="cvs file that has feature map filename and labels")
  parser.add_argument("img_dir", type=str, help="directory that contains image patch files")
  parser.add_argument("--check_point_fn", type=str, default=None, help="check point file name")
  parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
  parser.add_argument("--result_name", type=str, help="log result filename")
  parser.add_argument("--result_dir", type=str, default="evaluation", help="evaluation directory")
  parser.add_argument("--batch_size", type=int, default=16, help="batch for training")

  args = parser.parse_args()
  return args

if __name__ == "__main__":
  import time
  args = _parse_args()
  fmt = "%(message)s"
  #logging.basicConfig(stream=sys.stderr, format=fmt, level=logging.INFO)

  if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)
  logging.basicConfig(filename=os.path.join(args.result_dir, args.result_name +'_eval.log'), 
  			filemode='w', format=fmt, level=logging.INFO)

  logging.info("Input args")
  logging.info(args)
  logging.info(" ")

  model = torch.load(args.check_point_fn)  

  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  tran = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize,
            ])
  print ("creating dataset")
  dataset = DataSet(args.img_list_fn, args.img_dir, transform=tran) 
  logging.info("Total Sample count: {}".format(len(dataset)))

  device = None
  if torch.cuda.is_available():
    device = torch.device("cuda:"+str(args.gpu_id))
    model.to(device)

  logging.info('Starting evaluation at ' + time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()))
  logging.info("Filename, correct_count, total_count, precision")

  loader = DataLoader(dataset, args.batch_size, shuffle=False, drop_last=False, num_workers=4)
  result = _eval_one_epoch(loader, model, device)
  result_df = pd.DataFrame(data=result)
  ppn = os.path.join(args.result_dir, args.result_name + "_eval_result.pkl") 
  result_df.to_csv(ppn)

  logging.info('Ending evaluation at ' + time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()))

