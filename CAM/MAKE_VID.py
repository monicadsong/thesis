import cv2
import csv
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle
import sys


IM_DIR = '/data/vision/oliva/scratch/monica/prediction_30fps/'

def visualize(sequence):
    for (jpg, label) in sequence:
        image_name = os.path.join(IM_DIR, jpg)
    #image_name = IM_DIR + video_name  + '/' + '%03d' % x + ".jpg"
        #print ('reading', image_name)
        img = cv2.imread(image_name)
        if label == 1:
            #print ('drawing border')
            #print ("making border")
            #img = cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_CONSTANT,value=COL)
            height, width, _ = img.shape
            #print ("putting text", height, width)
            cv2.putText(img, "INCIDENT",
                    (10, int(height/4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4, COL, 12)
        #print ("creating output")
        out_jpg = os.path.join(args.output_dir, jpg)
        #print ('writing', out_jpg)
        cv2.imwrite(out_jpg, img) 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Make Video")
    parser.add_argument("--pickle", type=str, help="result pickle")
    parser.add_argument("--output_dir", type=str, help="output dir of frames")
    parser.add_argument("--type", type=str, help="RNN | LLFC | BASELINE")
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        #print ("making", args.output_dir)
        os.makedirs(args.output_dir)

    colors = {
        "RNN": (0,0,255),
        "LLFC": (255, 0,0),
        "BASELINE":(0,255,0)
    }
    COL = colors[args.type]

    with open(args.pickle, 'rb') as handle:
        d = pickle.load(handle)
    for video in d:
        print (video)
        vid_dir = os.path.join(args.output_dir, video)
        if not os.path.exists(vid_dir):
            print ("making",vid_dir)
            os.makedirs(vid_dir)
        visualize(d[video])
        
        
 

    


