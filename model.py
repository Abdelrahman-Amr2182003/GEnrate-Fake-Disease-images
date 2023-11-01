import torch
import torch.nn as nn
import torch.nn.functional as F
class Des(nn.Module):
    def __init__(self,image_size):
        super(Des,self).__init__()
        self.image_size = image_size

        self.conv1=nn.Conv2d(4,64,4,2,1,bias=False)#input,n_feautre_maps,kernel_size,stride,padding,biase state
        self.conv2=nn.Conv2d(64,128,4,2,1,bias=False)#input,n_feautre_maps,kernel_size,stride,padding,biase state
        self.batch2=nn.BatchNorm2d(128)#input
        self.conv3=nn.Conv2d(128,256,4,2,1,bias=False)#input,n_feautre_maps,kernel_size,stride,padding,biase state
        self.batch3=nn.BatchNorm2d(256)
        self.conv4=nn.Conv2d(256,512,4,2,1,bias=False)#input,n_feautre_maps,kernel_size,stride,padding,biase state
        self.batch4=nn.BatchNorm2d(512)
        self.conv5=nn.Conv2d(512, 1024, 4, 2, 1, bias = False)#input,n_feautre_maps,kernel_size,stride,padding,biase state
        self.batch5=nn.BatchNorm2d(1024)
        self.conv6=nn.Conv2d(1024, 1, 4, 1, 0, bias = False)#input,n_feautre_maps,kernel_size,stride,padding,biase state

        self.embed=nn.Embedding(4,self.image_size*self.image_size)
    def disc(self,x):
        x=F.leaky_relu(self.conv1(x),negative_slope=0.2,inplace=True)# move the input through all the layers
        x=F.leaky_relu(self.batch2(self.conv2(x)),negative_slope=0.2,inplace=True)
        x=F.dropout2d(x,0.3)
        x=F.leaky_relu(self.batch3(self.conv3(x)),negative_slope=0.2,inplace=True)
        x=F.leaky_relu(self.batch4(self.conv4(x)),negative_slope=0.2,inplace=True)
        x = F.leaky_relu(self.batch5(self.conv5(x)), negative_slope=0.2, inplace=True)
        x=F.sigmoid(self.conv6(x))
        return x.view(-1)
    def forward(self,x,labels):
        embedding=self.embed(labels).view(labels.shape[0],1,self.image_size,self.image_size)
        x=torch.cat([x,embedding],dim=1)
        x=self.disc(x)

        return x.view(-1)


class Gen(nn.Module):
    def __init__(self):
        super(Gen,self).__init__()
        # the order of parameters are writen in the comments beside each line

        self.conv1=nn.ConvTranspose2d(200,2048,4,1,0,bias=False)# input,number of output feautre maps,kernel size,stride,padding,biase=Flase
        self.batch1=nn.BatchNorm2d(2048,momentum=0.1,eps=0.8)

        self.conv2=nn.ConvTranspose2d(2048,1024,4,2,1,bias=False)# input,number of output feautre maps,kernel size,stride,padding,biase=Flase
        self.batch2=nn.BatchNorm2d(1024,momentum=0.1,eps=0.8)

        self.conv3=nn.ConvTranspose2d(1024,512,4,2,1,bias=False)# input,number of output feautre maps,kernel size,stride,padding,biase=Flase
        self.batch3=nn.BatchNorm2d(512,momentum=0.1,eps=0.8)

        self.conv4=nn.ConvTranspose2d(512,256,4,2,1,bias=False)# input,number of output feautre maps,kernel size,stride,padding,biase=Flase
        self.batch4=nn.BatchNorm2d(256,momentum=0.1,eps=0.8)

        self.conv5=nn.ConvTranspose2d(256,128,4,2,1,bias=False)# input,number of output feautre maps,kernel size,stride,padding,biase=Flase
        self.batch5=nn.BatchNorm2d(128,momentum=0.1,eps=0.8)

        self.conv6=nn.ConvTranspose2d(128,3,4,2,1,bias=False)# input,number of output feautre maps,kernel size,stride,padding,biase=Flase

        self.embed=nn.Embedding(4,100)
    def gen(self,x):
        x=F.relu(self.batch1(self.conv1(x)),inplace=True)#means that it will modify the input directly, without allocating any additional output.
        #It can sometimes slightly decrease the memory usage
        x=F.relu(self.batch2(self.conv2(x)),inplace=True)
        x=F.dropout2d(x,0.25)
        x=F.relu(self.batch3(self.conv3(x)),inplace=True)
        x=F.relu(self.batch4(self.conv4(x)),inplace=True)
        x = F.relu(self.batch5(self.conv5(x)), inplace=True)
        x=F.tanh(self.conv6(x))
        return x
    def forward(self,x,labels):
        embedding=self.embed(labels).unsqueeze(2).unsqueeze(3)
        x=torch.cat([x,embedding],dim=1)
        x=self.gen(x)
        return x
