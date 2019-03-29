# extends the Dataset 
from PIL import Image
from torch.utils.data.dataset import Dataset
import pickle
import pandas as pd
import os

# for 15 fps, 

# def find_classes():
#     DIR = "/data/vision/oliva/scratch/moments/moments_nov17_frames"
#     classes = os.listdir(DIR)
#     classes.sort()
#     class_to_idx = {classes[i]: i for i in range(len(classes))}
#     return classes, class_to_idx

class DataSet(Dataset):
    def __init__(self, image_list, image_dir, transform=None):
        self.transform = transform
        self.imgs = pd.read_csv(image_list) 
        self.image_dir = image_dir

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        row = self.imgs.iloc[index, :]
        fn = row[0]
        target = row[1]
        pn = self.image_dir + "/" + fn
        img = Image.open(pn)
        if self.transform is not None:
            img = self.transform(img)
        sample = {'image': img, 'label': target, 'filename': fn}
        return sample

if __name__ == '__main__':
  import argparse
  from torchvision.transforms import Compose
  import torchvision.transforms as transforms
  parser = argparse.ArgumentParser("Test dataset")
  parser.add_argument("--csv_file", type=str, default="input.csv", help="csv filename")
  parser.add_argument("--image_dir", type=str, help="image dir")
  args = parser.parse_args()

  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  tran= transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize,
              ])

  dataset = DataSet(args.csv_file, args.image_dir, tran)

  for s in dataset:
    print("name: {}, shape: {}, label: {}".format(
         s['filename'], s['image'].shape, s['label']))


