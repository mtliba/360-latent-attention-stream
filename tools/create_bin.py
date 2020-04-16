import cv2
import os
import numpy as np

listvid=['12_TeatroRegioTorino','14_Warship','15_Cockpit','18_Bar','1_PortoRiverside','3_PlanEnergyBioLab','4_Ocean']
my_dict={'12_TeatroRegioTorino':600,'14_Warship':500,'15_Cockpit':500,'16_Turtle':600,'18_Bar':501,'1_PortoRiverside':500,'2_Diner':600,'3_PlanEnergyBioLab':500,'4_Ocean':600,'5_Waterpark':601}
os.mkdir('./output')
os.chdir('./output')
for f in os.listdir('./output'):
  
  image_folder ='./output/'+f
  images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
  images.sort()
  for i,image in enumerate(images):
      image_in = cv2.imread(os.path.join(image_folder, image))

      if image_in is None:
        image_in = cv2.imread(os.path.join(image_folder, images[i-1]))

      if image_in.shape != (1024,2048) :
        image_in = cv2.resize(image_in, (2048,1024), interpolation = cv2.INTER_AREA)

      image_in = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
      
      with open(f+'_2048x1024x'+str(len(images))+'_32b.bin', 'ab') as the_file:
          for i in range(image_in.shape[0]):
              line = np.float32(np.array(image_in[i]))
              the_file.write(bytes(line))
      print(image)
  print('complete :',f)