import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torchvision
from torchvision.models import resnet50, vgg16, alexnet, inception_v3
import os
import json
import time
from tqdm import tqdm, tqdm_notebook
import csv
from PIL import Image
from pytorch_grad_cam import GradCAM 
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# --------------------获取Label-------------------------------------
def load_ground_truth(csv_filename):
    image_id_list = []
    label_tar_list = []

    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            image_id_list.append( row['ImageId'] )
            label_tar_list.append( int(row['TrueLabel'])-1 )

    return image_id_list,label_tar_list

#load image list
image_id_list,label_tar_list=load_ground_truth('./dataset/images.csv')

#------------------数据预处理-----------------------------------------

# trn = transforms.Compose([transforms.Resize((224, 224)),
#                         transforms.ToTensor(),
#                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

trn = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
# trn = transforms.Compose([transforms.ToTensor()])


# fix the random seed of pytorch and make cudnn deterministic for reproducing the same results
# torch.manual_seed(42)
# torch.backends.cudnn.deterministic = True

#-------------------模型选择-------------------------------------------
# load the pre-trained model
model = inception_v3(pretrained=True).eval().to(device)
# model = resnet50(pretrained=True).eval()

#-------------------载入图片--------------------------------------------
imgs = torch.zeros(len(image_id_list),3,299,299)
for i in tqdm_notebook(range(len(image_id_list))):
    imgs[i] = trn(Image.open(os.path.join('./dataset/images/'+image_id_list[i])+'.png'))

#------------------使用Grad-CAM得到热力图并进行二值化--------------------
suc_imgs1 = imgs.to(device)
suc_labels1 = label_tar_list
target_layers = [model.Mixed_7c]
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

cam_map = [[None]]*len(suc_imgs1)
cam_map_binary = [[None]]*len(suc_imgs1)
for i in range(len(suc_imgs1)):
    suc_imgs1_single = suc_imgs1[i].unsqueeze(0)
    suc_labels1_single = [ClassifierOutputTarget(suc_labels1[i])]
    cam_map[i] = cam(input_tensor=suc_imgs1_single, targets=suc_labels1_single)[0] # 不加平滑

    # 热力图的二值化
    threshold = cam_map[i].mean() #+ (cam_map[i].max()-cam_map[i].min())/4
    h,w = cam_map[i].shape
    thresh_img = cam_map[i].copy().reshape(h*w)
    for j in range(h*w):
        if thresh_img[j] < threshold:
            thresh_img[j] = 0

        
    thresh_img = thresh_img.reshape(h,w)
    cam_map_binary[i] = thresh_img




#-------------------evaluation on Image_1000----------------------------
batch_size = 32
num_batches = np.int(np.ceil(len(image_id_list)/batch_size))
cnt=0
for k in tqdm_notebook(range(0,num_batches)):
    batch_size_cur=min(batch_size,len(image_id_list)-k*batch_size)
    X_ori = torch.zeros(batch_size_cur,3,299,299).to(device) 
    for i in range(batch_size_cur): 
        X_ori[i]=trn(Image.open(os.path.join('./dataset/images/'+image_id_list[k*batch_size+i])+'.png'))
    labels=torch.argmax(model(X_ori),dim=1)
    label_tar=torch.tensor(label_tar_list[k*batch_size:k*batch_size+batch_size_cur]).to(device)
    cnt += (labels == label_tar).sum().item()
acc = cnt/len(image_id_list)
print(f'Top 1 Acc: {acc * 100:4.2f}%')




