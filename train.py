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
from model_res import Encoder
from utils import AverageMeter
from extract_points_128sil import ExtractData
import os

def main():
    # Arguments
        parser = argparse.ArgumentParser(description='Tidal flats regression model')
        parser.add_argument('--epochs', default=90, type=int, help='number of total epochs to run')
        parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
        parser.add_argument('--bs', default=8, type=int, help='batch size')
        parser.add_argument('--load', default=None, type=str, help='model')
        parser.add_argument('--band', default=4, type=int, help='band')
        parser.add_argument('--layer', default=50, type=int, help='layer')
        args = parser.parse_args()
	#load dataset
        image_list= r'/data/volume_2/look_space/Deeplearning_class/image/'  
        fie_vec=r'/data/volume_2/look_space/Deeplearning_class/shp_bound/'
        X,y=ExtractData(image_list,fie_vec)
        X=np.float32(X)
        y=np.float32(y)
        print(len(X))
	#Normalise images
        for i in range(len(X)):
	        for b in range(len(X[i])):
		        min_im=0
		        max_im=10000
		        X[i][b] = (X[i][b]-min_im)/(max_im-min_im)
	# Load data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
        train_dataset = []
        val_dataset = []
        for k in range(len(X_train)):
	        train_dataset.append([X_train[k], y_train[k]])
        for k1 in range(len(X_val)):
	        val_dataset.append([X_val[k1], y_val[k1]])
        train_loader = DataLoader(train_dataset, shuffle=True,batch_size=args.bs)
        val_loader = DataLoader(val_dataset, batch_size=100)	
	
    # Create model
        print('Model created.')
        resnet50 = Encoder(args.layer,False,args.band)
        resnet50.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    # Training parameters
        optimizer = torch.optim.Adam( resnet50.parameters(), args.lr )
        batch_size = args.bs
 
    # Logging
        writer = SummaryWriter(comment='{}-lr{}-e{}-bs{}'.format("Resnet50", args.lr, args.epochs, args.bs), flush_secs=30)


    # Start training...
        for epoch in range(args.epochs):
	        train_loss = 0.0
	        batch_time = AverageMeter()
	        N = len(train_loader)
        # Switch to train mode
	        resnet50.train()

	        end = time.time()

	        for i, sample_batched in enumerate(train_loader):
		        optimizer.zero_grad()

            # Prepare sample and target
		        inputs,targets=sample_batched
		        output = resnet50(inputs.cuda())
		        l_mse=nn.MSELoss()
		        l_mse1=l_mse(output.squeeze(), targets.cuda())
		        print(l_mse1)
		        print(output[1])
		        print(targets[1])
		        losses = l_mse1
            # Update step
		        losses.backward()
		        optimizer.step()

            # Measure elapsed time
		        batch_time.update(time.time() - end)
		        end = time.time()
		        eta = str(datetime.timedelta(seconds=int(batch_time.val*(N - i))))
		        train_loss += losses.item()*inputs.size(0)    
            # Log progress
		        niter = epoch*N+i
		        if i % 5 == 0:
                # Print to console
		        	print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                'ETA {eta}\t'
                'Loss {loss:.4f} ({loss:.4f})'
                .format(epoch, i, N, batch_time=batch_time, loss=train_loss, eta=eta))

                # Log to tensorboard
		        	writer.add_scalar('Train/Loss', train_loss, niter)
                

		        if i % 300 == 0:
		        	LogProgress(resnet50, writer, val_loader, niter)
		        if epoch % 2 == 0:
		        	save_model(resnet50,epoch,optimizer)
		        if epoch == 10:
		        	save_model(resnet50,epoch,optimizer)
        # Record epoch's intermediate results
	        LogProgress(resnet50, writer, val_loader, niter)
        #train_loss1 = train_loss/len(train_loader.sampler)
        #val_loss1 = val_loss/len(val_loader.sampler)
        #writer.add_scalar('Training vs. Validation Loss/Loss',
                    #{ 'Training' : train_loss1, 'Validation' : val_loss1 },
                    #epoch + 1)
def save_model(resnet50,epoch,optimizer):
    """Save model weights to disk
        """
    torch.save(resnet50.state_dict(),"/data/volume_2/look_space/Deeplearning_class/model_rs_jul29_sil2/model_{}.pt".format(epoch))
    torch.save(optimizer.state_dict(),"/data/volume_2/look_space/Deeplearning_class/model_rs_jul29_sil2/adam.pt")

def LogProgress(resnet50, writer, val_loader, epoch):
        val_loss = 0.0
        resnet50.eval()
        sequential = val_loader
        val_in,val_lab = next(iter(sequential))
        val_out=resnet50(val_in.cuda())
        l_mse=nn.MSELoss()
        val_loss += l_mse(val_out.squeeze(),val_lab.cuda()).item()*val_in.size(0)  
    #depth[depth<=0.2]=0
        writer.add_scalar('Val/Loss', val_loss, epoch)
        #return val_loss

def load_model(load,lr):
    """Load model(s) from disk
        """
    load = os.path.expanduser(load)

    assert os.path.isdir(load), \
            "Cannot find folder {}".format(load)
    print("loading model from folder {}".format(load))

    path = os.path.join(load, "model_19.pt")
    pretrained_dict = torch.load(path)
    model = Model().cuda()
    model.load_state_dict(pretrained_dict)
    optimizer_load_path = os.path.join(load, "adam.pt")
    if os.path.isfile(optimizer_load_path):
        print("Loading Adam weights")
        optimizer = torch.optim.Adam( model.parameters(), lr )
        optimizer_dict = torch.load(optimizer_load_path)
        optimizer.load_state_dict(optimizer_dict)


if __name__ == '__main__':
	main()
