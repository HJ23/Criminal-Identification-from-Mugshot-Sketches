import torch.nn as nn
import torch.nn.functional as F
import Utils.misc as misc
import torch

class BasicNetwork(nn.Module):
	def __init__(self):
		super(BasicNetwork,self).__init__()
	def forward(self,input):
		pass


class GENERATOR(BasicNetwork):
#   general structure of U-Net based model
    def __init__(self, d=64):
        super(GENERATOR, self).__init__()
        # donwsampling layers
        self.convolution1 = nn.Conv2d(3, d, 4, 2, 1)
        self.convolution2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.convolution2_batch = nn.BatchNorm2d(d * 2)
        self.convolution3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.convolution3_batch = nn.BatchNorm2d(d * 4)
        self.convolution4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.convolution4_batch = nn.BatchNorm2d(d * 8)
        self.convolution5 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.convolution5_batch = nn.BatchNorm2d(d * 8)
        self.convolution6= nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.convolution6_batch = nn.BatchNorm2d(d * 8)
        self.convolution7= nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.convolution7_batch= nn.BatchNorm2d(d * 8)
        self.convolution8= nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        # self.conv8_bn = nn.BatchNorm2d(d * 8)

        # transpose convolution for upsampling 
        self.transpose_convolution1 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1)
        self.transpose_convolution1_batch = nn.BatchNorm2d(d * 8)
        self.transpose_convolution2= nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.transpose_convolution2_batch = nn.BatchNorm2d(d * 8)
        self.transpose_convolution3= nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.transpose_convolution3_batch= nn.BatchNorm2d(d * 8)
        self.transpose_convolution4= nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.transpose_convolution4_batch= nn.BatchNorm2d(d * 8)
        self.transpose_convolution5= nn.ConvTranspose2d(d * 8 * 2, d * 4, 4, 2, 1)
        self.transpose_convolution5_batch= nn.BatchNorm2d(d * 4)
        self.transpose_convolution6= nn.ConvTranspose2d(d * 4 * 2, d * 2, 4, 2, 1)
        self.transpose_convolution6_batch= nn.BatchNorm2d(d * 2)
        self.transpose_convolution7= nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1)
        self.transpose_convolution7_batch= nn.BatchNorm2d(d)
        self.transpose_convolution8= nn.ConvTranspose2d(d * 2, 3, 4, 2, 1)

    def forward(self, input):
      # preparing net
        e1 = self.convolution1(input)
        e2 = self.convolution2_batch(self.convolution2(F.leaky_relu(e1, 0.2)))
        e3 = self.convolution3_batch(self.convolution3(F.leaky_relu(e2, 0.2)))
        e4 = self.convolution4_batch(self.convolution4(F.leaky_relu(e3, 0.2)))
        e5 = self.convolution5_batch(self.convolution5(F.leaky_relu(e4, 0.2)))
        e6 = self.convolution6_batch(self.convolution6(F.leaky_relu(e5, 0.2)))
        e7 = self.convolution7_batch(self.convolution7(F.leaky_relu(e6, 0.2)))
        e8 = self.convolution8(F.leaky_relu(e7, 0.2))

        
        d1 = F.dropout(self.transpose_convolution1_batch(self.transpose_convolution1(F.relu(e8))), 0.5, training=True)
        d1 = torch.cat([d1, e7], 1)
        d2 = F.dropout(self.transpose_convolution2_batch(self.transpose_convolution2(F.relu(d1))), 0.5, training=True)
        d2 = torch.cat([d2, e6], 1)
        d3 = F.dropout(self.transpose_convolution3_batch(self.transpose_convolution3(F.relu(d2))), 0.5, training=True)
        d3 = torch.cat([d3, e5], 1)
        d4 = self.transpose_convolution4_batch(self.transpose_convolution4(F.relu(d3)))
        d4 = torch.cat([d4, e4], 1)
        d5 = self.transpose_convolution5_batch(self.transpose_convolution5(F.relu(d4)))
        d5 = torch.cat([d5, e3], 1)
        d6 = self.transpose_convolution6_batch(self.transpose_convolution6(F.relu(d5)))
        d6 = torch.cat([d6, e2], 1)
        d7 = self.transpose_convolution7_batch(self.transpose_convolution7(F.relu(d6)))
        d7 = torch.cat([d7, e1], 1)
        d8 = self.transpose_convolution8(F.relu(d7))
        output = F.tanh(d8)
        return output

    def weight_initializer(self, mean, std):
        for m in self._modules:
            misc.winit(self._modules[m], mean, std)



class DISCRIMINATOR(BasicNetwork):
    def __init__(self, d=64):
        super(DISCRIMINATOR, self).__init__()
        self.convolution1 = nn.Conv2d(6, d, 4, 2, 1)
        self.convolution2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.convolution2_batch = nn.BatchNorm2d(d * 2)
        self.convolution3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.convolution3_batch = nn.BatchNorm2d(d * 4)
        self.convolution4= nn.Conv2d(d * 4, d * 8, 4, 1, 1)
        self.convolution4_batch = nn.BatchNorm2d(d * 8)
        self.convolution5 = nn.Conv2d(d * 8, 1, 4, 1, 1)

    # agin katsayilarinin ayarlanmasi 
    def weight_initializer(self, mean, std):
        for m in self._modules:
            misc.winit(self._modules[m], mean, std)

    # input-cikti methodu (forward-pass)
    def forward(self, input, real_out):
        x = torch.cat([input, real_out], 1)
        x = F.leaky_relu(self.convolution1(x), 0.2)
        x = F.leaky_relu(self.convolution2_batch(self.convolution2(x)), 0.2)
        x = F.leaky_relu(self.convolution3_batch(self.convolution3(x)), 0.2)
        x = F.leaky_relu(self.convolution4_batch(self.convolution4(x)), 0.2)
        x = F.sigmoid(self.convolution5(x))
        return x



















    
