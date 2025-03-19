import time
import argparse
import datetime
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils   
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
import numpy as np
import numpy.ma as ma
import torch.nn.functional as F
from model_res import *
#from ConvNeXt import *
from utils import AverageMeter
from extract_points_c import ExtractData
from sklearn.metrics import r2_score
from image_functions import normalise,simple_image_generator,normalise_pt
import os

#load dataset
image_list= r'/data/volume_2/look_space/Deeplearning_class/test_vs/'  
fie_vec=r'/data/volume_2/look_space/Deeplearning_class/test_shp/'
#image_list= r'D:/Utrecht_macrozoobenthos_data_feb2021/Deeplearning_class/image_set1/'  
#fie_vec=r'D:/Utrecht_macrozoobenthos_data_feb2021/Deeplearning_class/shp_bound/'
load=r'/data/volume_2/look_space/Deeplearning_class/mod_augmed192senti_visu/'
X,y=ExtractData(image_list,fie_vec,192,5)
bands=4#6
X=np.float32(X)
y=np.float32(y)
print(len(X))
#Normalise images
X=normalise(X,bands,'senti')
y=normalise_pt(y)
#y=np.log10(y)
#X,y=simple_image_generator(X, y,rotation_range=180, horizontal_flip=True,vertical_flip=True)
print(len(X))
print(len(y))
# Load data
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01)
# train_dataset = []
# test_dataset = []
# for k in range(len(X_train)):
	# train_dataset.append([X_train[k], y_train[k]])
	# print(y_train[k])
	# c=X_train[k]
	# #plt.imshow(c[0,:,:])
	# #plt.show()
# for k1 in range(len(X_val)):
	# test_dataset.append([X_val[k1], y_val[k1]])

# test_loader = DataLoader(test_dataset, batch_size=len(X_val))

# Load data
test_dataset = []   
for k in range(len(X)):
	test_dataset.append([X[k], y[k]])

#test_loader = DataLoader(test_dataset, shuffle=True,batch_size=len(X))
test_loader = DataLoader(test_dataset, shuffle=True,batch_size=100)#64
# Load model

load = os.path.expanduser(load)

assert os.path.isdir(load), \
		"Cannot find folder {}".format(load)
print("loading model from folder {}".format(load))

path = os.path.join(load, "model_250.pt")
pretrained_dict = torch.load(path,map_location=torch.device( "cpu"))
resnet50 = Encoder(50,False,4)
#resnet50=convnext_small(pretrained=False,in_22k=False,in_chans=4,num_classes=1)#50
resnet50.to(torch.device( "cpu"))
resnet50.load_state_dict(pretrained_dict)
optimizer_load_path = os.path.join(load, "adam.pt")
# if os.path.isfile(optimizer_load_path):
	# print("Loading Adam weights")
	# optimizer = torch.optim.Adam( resnet50.parameters(), 0.00005 )
	# optimizer_dict = torch.load(optimizer_load_path,map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
	# optimizer.load_state_dict(optimizer_dict)
test_loss=0.0
resnet50.eval()
input_1=[]
out=[]

with torch.no_grad():
	for da, la in test_loader:
		test_in,test_lab = da,la
		test_out=resnet50(test_in)
		input_1.extend(test_lab.detach().numpy())
		#print(input)
		out.extend(test_out.detach().numpy())
		#print(out)
		l_mse=nn.MSELoss()
		test_loss += l_mse(test_out.squeeze(),test_lab).item()*test_in.size(0)  
# sequential = test_loader
# test_in,test_lab = next(iter(sequential))
# test_out=resnet50(test_in)
# l_mse=nn.MSELoss()
# test_loss += l_mse(test_out.squeeze(),test_lab).item()*test_in.size(0)  
print(test_loss)
# true=test_lab.detach().numpy()
# pred=test_out.detach().numpy()
true=input_1
print(len(input_1))
pred=out
score=r2_score(true,pred)
print(score)
plt.scatter(true,pred)
plt.title("Patch size-192;R_score:"+str(score))
plt.xlabel("True value")
plt.ylabel("Predicted value")
plt.show()
