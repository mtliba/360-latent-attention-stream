import numpy as np
from PIL import Image
import projection_methods
import os
import cv2

stored_path= './final_equi'
os.mkdir(stored_path)
for pa in os.listdir('./projected'):
    images = [img for img in os.listdir('./projected/'+pa+'/0') if img.endswith(".png")]
    images.sort()
    face={'0':'F', '1':'R', '2':'B', '3':'L', '4':'U', '5':'D'}
    img={}
    print('start inverse project of :',pa)
    for i,m in enumerate(images):
        for f in face :
          if not os.path.exists(os.path.join(stored_path, pa,m)):
            a = np.array((Image.open(os.path.join('./projected',pa, f,m))).resize((256,256)))
            img[face[f]]=np.array(a)
        if not os.path.exists(os.path.join(stored_path, pa)):
                os.mkdir(os.path.join(stored_path, pa))
        if not os.path.exists(os.path.join(stored_path, pa,m)):
                out = projection_methods.c2e(img, h=1024, w=2048, mode='bilinear',cube_format='dict')
                Image.fromarray(out.astype(np.uint8)).save(os.path.join(stored_path, pa,m))
                print('complete '+pa+'/'+m)
        else : 
                print('done',pa+'/'+m)
