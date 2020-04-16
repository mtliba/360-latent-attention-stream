import torch
import os
import datetime
import numpy as np
import cv2
from torch.utils import data
from torchvision import utils


# The DataLoader for our specific datataset with extracted frames
class Static_dataset(data.Dataset):

  def __init__(self, split, number_of_frames, root_path, load_gt, resolution=None, val_perc = 0.1):
        # augmented frames
        self.frames_path = os.path.join(root_path, "frames") 
        self.load_gt = load_gt

        if load_gt:
          #ground truth        
          self.gt_path = os.path.join(root_path, "maps") 

        self.resolution = resolution
        self.frames_list = []
        # A list to keep all the dictionaries of ground truth - saliency map pairings for each video
        self.frame_gts_dict = {}

        # Gives accurate human readable time, rounded down not to include too many decimals

        start = datetime.datetime.now().replace(microsecond=0) 
        for i in range(number_of_frames+1): 

            # The way the folder structure is organized allows to iterate simply  
            ext = 4-len(str(i))
            ext = '0'*ext + str(i)  
            ext = ext + '.png' 

            frame_files = os.listdir(os.path.join(self.frames_path,ext))
            
            self.frames_list.append(frame_files)

            if load_gt:
              gt_files = os.listdir(os.path.join(self.gt_path , ext))

              # Make dictionary where keys are the frames and values are the ground truths


              self.frame_gts_list[frame_files]= gt_files

            if i%10==0:
              print("frame {} finished.".format(i))
              print("Time elapsed so far: {}".format(datetime.datetime.now().replace(microsecond=0)-start))


        limit = int(round(val_perc*len(self.frames_list)))
        if split == "validation":
          self.frames_list = self.frames_list[:limit]
          
        elif split == "train":
          self.frames_list = self.frames_list[limit:]
          





  def __len__(self):
        'Denotes the total number of samples'
        return len(self.frames_list)

  def __getitem__(self, frame_index):

        'Generates one sample of data'
        
        frames = self.frames_list[frame_index]
        if self.load_gt:
          gts = self.frame_gts_list[frames]

        packed = []

        normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                         std = [0.229, 0.224, 0.225])
        path_to_frame = os.path.join(self.frames_path, frame)

        X = cv2.imread(path_to_frame)

        if self.resolution!=None:

            X = cv2.resize(X, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_AREA)
            
        X = X.astype(np.float32)
        X = torch.FloatTensor(X)
        X = normalize(X)
        # swap channel dimensions
        X = X.permute(2,0,1) 
        # add batch dim
        data = X.unsqueeze(0)


        # Load and preprocess ground truth (saliency maps)
        if self.load_gt:

            path_to_gt = os.path.join(self.gt_path , gts)
        
        # Load as grayscale
        
        y = cv2.imread(path_to_gt, 0) 
        
        if self.resolution!=None:
            y = cv2.resize(y, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_AREA)
        
        y = (y-np.min(y))/(np.max(y)-np.min(y))

        y = torch.FloatTensor(y)

        gt = y.unsqueeze(0)

        if self.load_gt:

            packed.append((data,gt)) # pack a list of data with the corresponding list of ground truths
        else:
            packed.append((data, "_"))


    return packed
