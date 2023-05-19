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


#如果有可使用的Gpu就默认使用第一块GPU，如果没有，就用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


'''#数据预处理
data_transform = {      
    "train": transforms.Compose([transforms.RandomResizedCrop(224),  #随机裁剪成224*224大小
                                 transforms.RandomHorizontalFlip(),   #随机水平翻转
                                 transforms.ToTensor(),             #转为张量形式
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),  #标准化处理
    "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

# 导入训练集并进行预处理
train_dataset = torchvision.datasets.CIFAR10(root='./data', 
                                              train=True,  # 下载训练集的数据
                                              download=True,    
                                              transform=data_transform["train"])
train_num = len(train_dataset)

# 按batch_size分批次加载训练集
train_loader = torch.utils.data.DataLoader(train_dataset,	# 导入的训练集
                                           batch_size=32, 	# 每批训练的样本数
                                           shuffle=True,	# 是否打乱训练集
                                           num_workers=0)	# 使用线程数，在windows下设置为0


# 导入验证集并进行预处理
validate_dataset = torchvision.datasets.CIFAR10(root='./data', 
                                                train=False,  # 下载训练集的数据
                                                download=True, 
                                                transform=data_transform["val"])

val_num = len(validate_dataset)

# 加载验证集
validate_loader = torch.utils.data.DataLoader(validate_dataset,	# 导入的验证集
                                              batch_size=32, 
                                              shuffle=False,
                                              num_workers=0)



class_list = train_dataset.class_to_idx    #获取类对应的索引
cla_dict = dict((val, key) for key, val in class_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)    #编码成json格式
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

'''
# --------------------获取Image和Label------------------------
# get the list of images along with the specified target labels
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

trn = transforms.Compose([transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# trn = transforms.Compose([
#      transforms.ToTensor(),])


'''class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]

norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])'''


# fix the random seed of pytorch and make cudnn deterministic for reproducing the same results
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

#-------------------模型选择-------------------------------------------
# load the pre-trained model
# model = inception_v3(pretrained=True,transform_input=False).eval()
model = resnet50(pretrained=True).eval()
for param in model.parameters():
    param.requires_grad=False
model.to(device)


#-------------------evaluation----------------------------------------
batch_size = 32
num_batches = np.int(np.ceil(len(image_id_list)/batch_size))
cnt=0
for k in tqdm_notebook(range(0,num_batches)):
    batch_size_cur=min(batch_size,len(image_id_list)-k*batch_size)
    X_ori = torch.zeros(batch_size_cur,3,224,224).to(device) 
    for i in range(batch_size_cur): 
        X_ori[i]=trn(Image.open(os.path.join('./dataset/images/'+image_id_list[k*batch_size+i])+'.png'))
    # labels=torch.argmax(model(X_ori),dim=1)
    _, labels = torch.max(model(X_ori), 1)
    label_tar=torch.tensor(label_tar_list[k*batch_size:k*batch_size+batch_size_cur]).to(device)
    # cnt=cnt+torch.sum(labels==label_tar)
    cnt += (labels == label_tar).sum().item()
print(cnt/len(image_id_list))




