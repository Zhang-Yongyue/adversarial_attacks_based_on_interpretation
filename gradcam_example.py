from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.models import resnet50, vgg16, alexnet, inception_v3, resnet18
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
import torch
import torchcam
from torchcam.utils import overlay_mask

# 设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)


# 数据预处理
# test_transform = transforms.Compose([transforms.Resize(512),
#                                      # transforms.CenterCrop(512),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize(
#                                          mean=[0.485, 0.456, 0.406], 
#                                          std=[0.229, 0.224, 0.225])
#                                     ])
test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
                                    ])


# 数据的加载
img_path ='./dataset/images/0c7ac4a8c9dfa802.png' #'test_img/cat_dog_new.jpg'
img_pil = Image.open(img_path)
input_tensor = test_transform(img_pil).unsqueeze(0).to(device) # 预处理
print(input_tensor.shape)


# 指定分析类别
# 如果 targets 为 None，则默认为最高置信度类别
targets = [ClassifierOutputTarget(305)]#281


# 模型的加载
# model = resnet50(pretrained=True).eval().to(device)
# model = vgg16(pretrained=True).eval().to(device)
model = inception_v3(pretrained=True).eval().to(device)
# model = resnet50(pretrained=True).eval().to(device)
# print(model.layer4[-1].conv3)
# print(model.features[6])
print(model.Mixed_7c)
# print(model.layer4[1])
# target_layers = [model.layer4[-1].conv3]
# target_layers = [model.features[6]]
target_layers = [model.Mixed_7c]
# target_layers = [model.layer4[1]]

# 计算热力图
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
cam_map = cam(input_tensor=input_tensor, targets=targets)[0] # 不加平滑
# cam_map = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)[0] # 加平滑
# plt.imshow(cam_map)
# plt.show()

# 热力图的二值化
threshold = cam_map.mean()
print(threshold)

h,w = cam_map.shape
thresh_img = cam_map.copy().reshape(h*w)
for i in range(h*w):
    if thresh_img[i] < threshold:
        thresh_img[i] = 0
    else:
        thresh_img[i] = 1
    
thresh_img = thresh_img.reshape(h,w)
# plt.imshow(thresh_img)
# plt.show()

# 对热力图进行双线性插值并与原图进行叠加显示
result = overlay_mask(img_pil, Image.fromarray(cam_map), alpha=0.6) # alpha越小，原图越淡
result1 = overlay_mask(img_pil, Image.fromarray(thresh_img), alpha=0.6)
result.save('output/B.jpg')
result1.save('output/B1.jpg')