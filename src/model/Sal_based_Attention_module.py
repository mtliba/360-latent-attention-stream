import torch
from torch import nn, sigmoid
from torch.nn.modules.upsampling import Upsample
from torch.nn.functional import interpolate
from torch.autograd import Variable
from troch.nn import MaxPool2d
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import Sigmoid, ReLU

from Encoders import  based_AM

# create pooling layer
class Downsample(nn.Module):
    # specify the kernel_size for downsampling 
    def __init__(self, kernel_size, stride):
        super(Downsample, self).__init__()
        self.pool = MaxPool2d
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.pool(x, kernel_size= self.kernel_size, stride= self.stride)
        return x

# create unpooling layer 
class Upsample(nn.Module):
    # specify the scale_factor for upsampling 
    def __init__(self, scale_factor, mode):
        super(Upsample, self).__init__()
        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

# create Add layer , support backprop
class Multiply(nn.Module):
    def __init__(self):
        super(Multiply, self).__init__()

    def forward(self, tensors):
        result = torch.ones(tensors[0].size())
        for t in tensors:
            result *= t
        return result

# create Multiply layer , supprot backprop
class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, tensors):
        result = torch.zeros(tensors[0].size())
        for t in tensors:
            result += t
        return result
# reshape vectors layer
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Sal_based_Attention_module(nn.Module):
    """
    In this model, we take salgan architecture and erase its last maxpooling layer and add attention module above its botellneck than mltiply attention module and salgan bottelneck results together 
    
    """
    def  __init__(self, use_gpu=True):
        super(Sal_based_Attention_module,self).__init__()

        self.use_gpu = use_gpu

        # Create encoder based on VGG16 architecture as pointed on salgan architecture and apply aforementionned changes
        Based_Attention_Module = based_AM
        
        # select only first 5 conv blocks , here we keep same receptive field of VGG 212*212 
        # each neuron on bottelneck will see just (244,244) viewport during sliding  ,
        # input (640,320) , features numbers on bottelneck 40*20*512, exclude last maxpooling of salgan ,receptive
        # features number on AM boottlneck 10*5*128 
        # attentioin moduels receptive field enlarged (676,676)
        self.encoder = torch.nn.Sequential(*Based_Attention_Module)
        self.attention_module = torch.nn.Sequential(*[
            Downsample(kernel_size= 2, stride=2),
            Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Downsample(kernel_size= 2, stride=2),
            Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)),
            Sigmoid(),
            Upsample(scale_factor=4 , mode='nearest' )

        ])
        
        self.reshape = Reshape(-1,512,72,36)
        self.hadmard = Multiply()
        self.residual = Add()

        # define decoder based on VGG16 (inverse order and Upsampling layers , chose nearest mode)
        decoder_list=[
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='nearest'),

            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='nearest'),

            Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='nearest'),

            Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Upsample(scale_factor=2, mode='nearest'),

            Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=0),
            Sigmoid(),
        ]

        self.decoder = torch.nn.Sequential(*decoder_list)

        print("Model initialized, Sal_based_Attention_module")
        print("architecture len :",str(len(self.Sal_based_Attention_module)))

    def forward(self, input):
        x = self.encoder[:](input)
        y = self.attention_module(x)
        repeted  = x.repeat(1,512,1,1)
        reshaped = self.reshape(repeted)
        product  = self.hadmard([x, reshaped])
        added    = self.residual([x, product])
        x = self.decoder(added)

        batch_size_x = x.data.size()[0]
        print(batch_size_x)
        spatial_size_x = x.data.size()[2:]
        print(spatial_size_x)

        batch_size_y = y.data.size()[0]
        print(batch_size_y)
        spatial_size_y = y.data.size()[2:]
        print(spatial_size_y)
        return x ,y # x is a saliency map at this point,y is the fixation map


'''
encoder logs description 
encoder :
image size for  encoder layer 1 :(640,320)
  layer 3x3 conv 64   : [3, 1] [  3, 1]
  receptive field (3,3)
image size for  encoder layer 2 :(640,320)
  layer 3x3 conv 64   : [3, 1] [  5, 1]
  receptive field (5,5)
image size for  encoder layer 3 :(320,160)
  layer pool1         : [2, 2] [  6, 2]
  receptive field (6,6)
image size for  encoder layer 4 :(320,160)
  layer 3x3 conv 128  : [3, 1] [ 10, 2]
  receptive field (10,10)
image size for  encoder layer 5 :(320,160)
  layer 3x3 conv 128  : [3, 1] [ 14, 2]
  receptive field (14,14)
image size for  encoder layer 6 :(160,80)
  layer pool2         : [2, 2] [ 16, 4]
  receptive field (16,16)
image size for  encoder layer 7 :(160,80)
  layer 3x3 conv 256  : [3, 1] [ 24, 4]
  receptive field (24,24)
image size for  encoder layer 8 :(160,80)
  layer 3x3 conv 256  : [3, 1] [ 32, 4]
  receptive field (32,32)
image size for  encoder layer 9 :(160,80)
  layer 3x3 conv 256  : [3, 1] [ 40, 4]
  receptive field (40,40)
image size for  encoder layer 10 :(40,20)
  layer pool3         : [4, 4] [ 52,16]
  receptive field (52,52)
image size for  encoder layer 11 :(40,20)
  layer 3x3 conv 512  : [3, 1] [ 84,16]
  receptive field (84,84)
image size for  encoder layer 12 :(40,20)
  layer 3x3 conv 512  : [3, 1] [116,16]
  receptive field (116,116)
image size for  encoder layer 13 :(40,20)
  layer 3x3 conv 512  : [3, 1] [148,16]
  receptive field (148,148)
image size for  encoder layer 14 :(40,20)
  layer 3x3 conv 512  : [3, 1] [180,16]
  receptive field (180,180)
image size for  encoder layer 15 :(40,20)
  layer 3x3 conv 512  : [3, 1] [212,16]
  receptive field (212,212)
image size for  encoder layer 16 :(40,20)
  layer 3x3 conv 512  : [3, 1] [244,16]
  receptive field (244,244)
image size for  encoder layer 17 :(20,10)
  layer pool4         : [2, 2] [260,32]
  receptive field (260,260)
image size for  encoder layer 18 :(20,10)
  layer 3x3 conv 64   : [3, 1] [324,32]
  receptive field (324,324)
image size for  encoder layer 19 :(20,10)
  layer 3x3 conv 128  : [3, 1] [388,32]
  receptive field (388,388)
image size for  encoder layer 20 :(10,5)
  layer pool5         : [2, 2] [420,64]
  receptive field (420,420)
image size for  encoder layer 21 :(10,5)
  layer 3x3 conv 64  : [3, 1] [548,64]
  receptive field (548,548)
image size for  encoder layer 22 :(10,5)
  layer 3x3 conv 128  : [3, 1] [676,64]
  receptive field (676,676)
image size for  encoder layer 23 :(10,5)
  layer 1x1 conv 1    : [1, 1] [676,64]
  receptive field (676,676)
'''