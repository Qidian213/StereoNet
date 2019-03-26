import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename): 
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath): # /media/hugonie/Hhome/dataset/SceneFlowData/

 # classes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
 # print(classes)
 # image = [img for img in classes if img.find('frames_cleanpass') > -1]
 # print(image)
 # disp  = [dsp for dsp in classes if dsp.find('disparity') > -1]
 # print(disp)
 # monkaa
 
 # monkaa_path = filepath + [x for x in image if 'monkaa' in x][0]
 # monkaa_disp = filepath + [x for x in disp if 'monkaa' in x][0]
 monkaa_path = filepath + '/frames_cleanpass/monkaa'
 monkaa_disp = filepath + '/disparity/monkaa'
 monkaa_dir  = os.listdir(monkaa_path)

 all_left_img=[]
 all_right_img=[]
 all_left_disp = []
 test_left_img=[]
 test_right_img=[]
 test_left_disp = []

 for dd in monkaa_dir:
   for im in os.listdir(monkaa_path+'/'+dd+'/left/'):
    if is_image_file(monkaa_path+'/'+dd+'/left/'+im):
     all_left_img.append(monkaa_path+'/'+dd+'/left/'+im)
     all_left_disp.append(monkaa_disp+'/'+dd+'/left/'+im.split(".")[0]+'.pfm')

   for im in os.listdir(monkaa_path+'/'+dd+'/right/'):
    if is_image_file(monkaa_path+'/'+dd+'/right/'+im):
     all_right_img.append(monkaa_path+'/'+dd+'/right/'+im)

 # flyingthings
 # flying_path = filepath + [x for x in image if x == 'flyingthings3D'][0]
 # flying_disp = filepath + [x for x in disp if x == 'flyingthings3D'][0]
 flying_path = filepath + '/frames_cleanpass/test'
 flying_disp = filepath + '/disparity/test'
 flying_dir  = os.listdir(flying_path)
 
 for dd in flying_dir:
   for im in os.listdir(flying_path+'/'+dd+'/left/'):
    if is_image_file(flying_path+'/'+dd+'/left/'+im):
     test_left_img.append(flying_path+'/'+dd+'/left/'+im)
     test_left_disp.append(flying_disp+'/'+dd+'/left/'+im.split(".")[0]+'.pfm')

   for im in os.listdir(flying_path+'/'+dd+'/right/'):
    if is_image_file(flying_path+'/'+dd+'/right/'+im):
     test_right_img.append(flying_path+'/'+dd+'/right/'+im)

 return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp
