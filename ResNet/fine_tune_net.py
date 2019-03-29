import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np

import logging, sys

def pytorch_initialize_weight(m):
  if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    nn.init.kaiming_normal_(m.weight)
    if m.bias is not None:
      m.bias.data.zero_()
  elif isinstance(m, nn.BatchNorm2d):
    m.weight.data.fill_(1)
    m.bias.data.zero_()

def _config_resnet152(num_class):
  model = models.resnet152(pretrained=True)
  feature_size = model.fc.in_features 
  model.fc = nn.Linear(feature_size, num_class)
  pytorch_initialize_weight(model.fc)
  return model
_config_resnet152.input_size = 224

def _config_resnet18(num_class):
  model = models.resnet18(pretrained=True)
  feature_size = model.fc.in_features 
  model.fc = nn.Linear(feature_size, num_class)
  pytorch_initialize_weight(model.fc)
  return model
_config_resnet18.input_size = 224

def _config_resnet34(num_class):
  model = models.resnet34(pretrained=True)
  feature_size = model.fc.in_features 
  model.fc = nn.Linear(feature_size, num_class)
  pytorch_initialize_weight(model.fc)
  return model
_config_resnet34.input_size = 224

def _config_densenet169(num_class):
  model = models.densenet169(pretrained=True)
  feature_size = model.classifier.in_features 
  model.classifier = nn.Linear(feature_size, num_class)
  pytorch_initialize_weight(model.classifier)
  return model
_config_densenet169.input_size = 224

def _config_densenet201(num_class):
  model = models.densenet201(pretrained=True)
  feature_size = model.classifier.in_features 
  model.classifier = nn.Linear(feature_size, num_class)
  pytorch_initialize_weight(model.classifier)
  return model
_config_densenet201.input_size = 224

def _config_inception_v3(num_class):
  model = models.inception_v3(pretrained=True, transform_input=True)
  feature_size = model.fc.in_features  
  model.fc = nn.Linear(feature_size, num_class)
  pytorch_initialize_weight(model.fc)
  return model
_config_inception_v3.input_size = 299

_arch_map = { 
      "resnet18": _config_resnet18,
      "resnet34": _config_resnet34,
      "resnet152": _config_resnet152,
      "densenet169": _config_densenet169,
      "densenet201": _config_densenet201,
      "inception_v3": _config_inception_v3
}

class FineTuneNet(nn.Module):
  def __init__(self, arch="resnet152", num_class=2):
    super(FineTuneNet, self).__init__()

    if arch in _arch_map:
      arch_config_fn = _arch_map[arch]
      self.input_size = arch_config_fn.input_size 
      self.model = arch_config_fn(num_class)
    else:
      assert False, "arch {} is not supported".format(arch)
    self.arch = arch

  def forward(self, x):
    logging.debug("x: {}".format(x.size()))
    output = self.model(x)

    #logging.debug("output: {}".format(output.size()))
    return output

if __name__ == '__main__':

  import argparse
  parser = argparse.ArgumentParser("Fine Tune Net")
  parser.add_argument("--gpu_ids", type=str, default="0", help="gpu id")
  parser.add_argument("--batch_size", type=int, default=2, help="batch for training")
  parser.add_argument("--class_cnt", type=int, default=2, help="number of class")
  parser.add_argument("--arch", type=str, default="resnet18", help="resnet18 | resnet34 | resnet152 | inception_v3 ")
  args = parser.parse_args()
  logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

  net = FineTuneNet(args.arch, args.class_cnt)

  image_width = net.input_size
  image_height = net.input_size

  imgs = torch.randn(args.batch_size, 3, image_width, image_height)

  if torch.cuda.is_available():
    device = torch.device("cuda:"+args.gpu_ids)
    net.to(device)
    imgs = imgs.to(device)
    print("imgs: ", imgs.is_cuda, type(imgs))

  net.eval()
  
  npd = sorted(dict(net.named_parameters()).items()) 
  param_cnt = 0
  for d in npd:
    print(d[0], d[1].shape, d[1].device) 
    param_cnt += np.prod(d[1].shape)

  print("Model param_cnt: ", param_cnt)

  y = net(imgs) 

