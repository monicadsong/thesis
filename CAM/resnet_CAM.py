from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import torch
import numpy as np
import cv2
import os
import argparse
import pickle

CAM_DIR = '/data/vision/oliva/scratch/monica/CAM'
FRAME_DIR ='/data/vision/oliva/scratch/monica/prediction_30fps'

num_classes = 2
alpha = 0.75


features_blobs = []
def load_model(model_path):
    print ('loading net')
    net = torch.load(model_path)
    feature_names = ["layer4", "avgpool"]
    print ('setting net to eval')
    net.eval()
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())
    print ('registering forward hook')
    model = net._modules['model']
    for name in feature_names:
        model._modules.get(name).register_forward_hook(hook_feature)
    #print ('features blobs', features_blobs)
    return model

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    print ('returning CAM')
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def extract_frames(video_name):
    print ('extracting frames')
    with open(os.path.join(CAM_DIR,'test_jan13_2_dict.p'), 'rb') as f:
        data = pickle.load(f)
    frames = data[video_name]
    frame_paths  = [os.path.join(FRAME_DIR, video_name, f) for f in frames]
    images = [Image.open(frame).convert('RGB') for frame in frame_paths]
    return images

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test CAM on a set of videos")
    parser.add_argument('--video_name', type=str, default=None)
    args = parser.parse_args()
    
    frames = extract_frames(args.video_name)
    OUTPUT_DIR = os.path.join(CAM_DIR, args.video_name)
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    model = load_model("incident_model.pth")
    
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
    weight_softmax[weight_softmax<0] = 0
    buffer = torch.FloatTensor(num_classes, 5).fill_(0)
    cam_buffer = torch.FloatTensor(5, 256, 256).fill_(0)

    for i in range(len(frames)):
        features_blobs = []
        print ("processing image")
        img_tensor = preprocess(frames[i])
        img_variable = Variable(img_tensor.unsqueeze(0).cuda(), volatile=True)
        print ("running the image")
        logit = model(img_variable)
        probs = F.softmax(logit).data.squeeze()
        frames[i] = np.array(frames[i])
        height, width, _ = frames[i].shape
        buffer[:,i%5] = probs
        probs = buffer.mean(1)
        #print (features_blobs)
        cam_buffer[i%5] = torch.FloatTensor(returnCAM(features_blobs[0], weight_softmax, [1])[0])
        CAM = cam_buffer.mean(0)
        CAM.add_(-CAM.min()).div_(CAM.max()).mul_(255)
        CAMs = np.uint8(CAM)
        heatmap = cv2.applyColorMap(cv2.resize(CAMs, (width, height)), cv2.COLORMAP_JET)
        frames[i] = frames[i]*0.6 + heatmap*0.4
        overlay = frames[i].copy()

        if probs[1] > 0.5:
            cv2.putText(overlay,"incident", (1, int(height/8)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        cv2.addWeighted(overlay, alpha, frames[i], 1 - alpha, 0, frames[i])
        # frames[i] = np.flip(frames[i], 2)
        output_jpg = os.path.join(OUTPUT_DIR, "%03d.jpg" % i)
        print ("writing", output_jpg)
        cv2.imwrite(output_jpg, frames[i])
