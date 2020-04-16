import cv2
import numpy as np
import os

slid = 6
images = 'path to images'
gt = 'path to gt'
fx = 'path to fixatio'

all_paths ={images:'./augmented_frames/',gt:'./augmented_gt',fx:'./augmented_fx'}

for path  in all_paths :

  stored_path = all_paths[path]

  os.mkdir(stored_path)
  
  for pa in os.listdir(path):
    if pa.endswith(".png"):
      img = cv2.imread(path+pa)
      step =int(img.shape[0]/slid)
      j=1

      high = int(img.shape[0]*0.04)
      print(high)
      upper =img[:high,:,:]
      upper = cv2.resize(upper,(int(img.shape[1]),int(img.shape[0]*0.05)))
      midle = img[high:-high,:,:]
      down =img[-high:,:,:]
      down = cv2.resize(down,(int(img.shape[1]),int(img.shape[0]*0.03)))
      im = np.concatenate((upper,midle,down),axis=0)
      im = cv2.resize(im,(int(img.shape[1]),int(img.shape[0])))
      cv2.imwrite(stored_path+'up_shifted'+pa,im)

      upper =img[:high,:,:]
      upper = cv2.resize(upper,(int(img.shape[1]),int(img.shape[0]*0.03)))
      midle = img[high:-high,:,:]
      down =img[-high:,:,:]
      down = cv2.resize(down,(int(img.shape[1]),int(img.shape[0]*0.05)))
      im = np.concatenate((upper,midle,down),axis=0)
      im = cv2.resize(im,(int(img.shape[1]),int(img.shape[0])))
      cv2.imwrite(stored_path+'down_shifted'+pa,im)
      for i in range(step,img.shape[0],step):

        first_part = img[:,:i,:]
        second_part = img[:,i:,:]
        inversed_part = img[:,-i:,:]
        second_inversed = img[:,:-i,:]
        full = np.concatenate((second_part,first_part),axis=1)
        fulliv = np.concatenate((inversed_part,second_inversed),axis=1)
        fulliv2 = cv2.flip(fulliv, 1)
        full2 = cv2.flip(full, 1)
        cv2.imwrite(stored_path+'direct'+pa,full)
        cv2.imwrite(stored_path+'direct_fliped'+pa,full2)
        cv2.imwrite(stored_path+'invers'+pa,fulliv)
        cv2.imwrite(stored_path+'invers_fliped'+pa,fulliv2)
     



