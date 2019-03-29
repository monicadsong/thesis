import sys, os

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import torchvision.transforms as transforms
import torch.nn as nn

from dataset import DataSet
from vanilla_net import BasicNet

import logging
import argparse

class ComputeReguarizationLoss(object):
  # l2_norm should not be used. We implmented initially as a bug, but it works fine
  def _l2_sum(w): return torch.sum(w*w)
  def _l1_sum(w): return torch.sum(w.abs())
  def _l2_norm(w): return w.norm(2)
  sum_dict = {"l2": _l2_sum, "l1": _l1_sum, "l2_norm": _l2_norm}

  def __init__(self, params, lm=1.0, reg_type="l2"):
    # only regularize weight not bias
    self.params = []
    self.param_cnt = 0
    for p in params:
      if 'weight' in p[0]:        # only regularization
        self.params.append(p[1])
        self.param_cnt += p[1].numel()
      else:
        print("bias: {} is not in the regularzation list".format(p[0]))

    self.lm = lm

    if reg_type in ComputeReguarizationLoss.sum_dict:
      self.sum_func = ComputeReguarizationLoss.sum_dict[reg_type]
    else:
      print("reg_type: {} is not supported, use l2")
      self.sum_func = ComputeReguarizationLoss.sum_dict["l2"]

  def __call__(self):
    p_sum = None
    for w in self.params:
      sf =  self.sum_func(w)
      if p_sum is None:
        p_sum = sf
      else:
        p_sum += sf
    
    reg_loss = (self.lm/self.param_cnt)*p_sum
    return reg_loss
 


def _parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("img_list_fn", help="cvs file that has feature map filename and labels")
  parser.add_argument("img_dir", type=str, help="directory that contains image patch files")
  parser.add_argument("--check_point_fn", type=str, default=None, help="check point file name")
  parser.add_argument("--reg", type=float, help="regularization ratio", default=1e04)
  parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
  parser.add_argument("--output_dir", type=str, default="results", help="directory where the output are saved")
  parser.add_argument("--batch_size", type=int, default=64, help="batch for training")
  parser.add_argument("--num_class", type=int, default=2, help="number of class")
  parser.add_argument("--learning_rate", type=float, default=0.001,  help="learning rate for optimizer, you should reduce it after num_epoch is executed")
  parser.add_argument("--num_epoch", type=int, default=100,  help="epoch count, default to 4. If not enough, reduce lreaning_rate and run again with --check_point_fn")

  args = parser.parse_args()
  return args

class Train(object):
  def __init__(self, args):
    self.args = args
    self.max_epoch = args.num_epoch 
    if args.check_point_fn is not None:
      print("using checkpoint: {}".format(args.check_point_fn)) 
      self.model = torch.load(args.check_point_fn)  
    else:  
      self.model = BasicNet()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    tran = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
            ])

    dataset = DataSet(args.img_list_fn, self.args.img_dir, transform=tran) 
    logging.info("Samples Count: {}".format(len(dataset)))

    self.loader = DataLoader(dataset, args.batch_size, shuffle=True, drop_last=True, num_workers=4)

    self.device = None
    if torch.cuda.is_available():
      self.device = torch.device("cuda:"+str(args.gpu_id))
      self.model.to(self.device)

    self.criterion = nn.CrossEntropyLoss()

    momentum = 0.9
    weight_decay = 1e-4
    self.optimizer = torch.optim.SGD(self.model.parameters(), args.learning_rate,
                                momentum=momentum,
                                weight_decay=weight_decay)

    self.regularization = ComputeReguarizationLoss(self.model.named_parameters(), lm=args.reg)


  def _train_one_epoch(self, epoch):
    self.model.train()

    PRINT_SKIP = 100
    SAVE_SKIP = 5000

    for i, data_batch in enumerate(self.loader):
      x_batch = data_batch['image']
      y_batch = data_batch['label'] 
    
      if self.device is not None:
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
 
      y_hat_batch = self.model(x_batch) 
      #print (y_hat_batch)
      if isinstance(y_hat_batch, (tuple)):
        print ('yhatbatch is a tuple')
        # in this cases, axusilary loss
        print ('FIRST element', y_hat_batch[0])
        print ('SECOND element', y_hat_batch[1])
        total_loss = []
        for y in y_hat_batch: 
          l = self.criterion(y, y_batch)
          total_loss.append(l)
        loss_ce = sum(total_loss)
      else:
        print ('yhatbatch is NOT a tuple')
        loss_ce = self.criterion(y_hat_batch, y_batch)

      loss_re = self.regularization()
      loss = loss_ce + loss_re

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      if i%PRINT_SKIP is 0:
        logging.info("{}, {}, {}, {}, {}".format(epoch, i,
                float(loss.data), float(loss_ce.data),
                float(loss_re.data)))
		
      if (i%SAVE_SKIP is 0):
        check_point_fn = "ckpt_" + str(epoch) + "_" + str(i) + ".pth"
        logging.info("save: {}".format(check_point_fn))
        torch.save(self.model, os.path.join(self.args.output_dir, check_point_fn))

  def train(self):
    starting_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime())
    logging.info('Starting evaluation at ' + starting_time )
    s_step = 4
    s_size = self.args.num_epoch//s_step
    scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=s_size, gamma=0.5)
    for epoch in xrange(self.max_epoch): 
      scheduler.step()
      for pg in self.optimizer.param_groups:
        print("Current lr: {}".format(pg['lr']))
      self._train_one_epoch(epoch)

    ending_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime())
    logging.info('Ending evaluation at ' + ending_time)

if __name__ == "__main__":
  import time
  args = _parse_args()

  if args.check_point_fn is not None:
    if not os.path.exists(args.check_point_fn):
      print("checkpoint: {} does not exist".format(args.check_point_fn)) 
      sys.exit(1)

  if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

  fmt = "%(message)s"
  logging.basicConfig(stream=sys.stderr, format=fmt, level=logging.INFO)
  log_fn = os.path.join(args.output_dir, 'train.log')
  #logging.basicConfig(filename=log_fn, format=fmt, level=logging.INFO)
  logging.info(args)

  trainer = Train(args)
  trainer.train()

