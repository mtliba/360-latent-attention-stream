import torch
from torch import nn, sigmoid
from torch.nn.modules.upsampling import Upsample
from torch.nn.functional import interpolate
from torch.autograd import Variable
from troch.nn import MaxPool2d
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import Sigmoid, ReLU
from Encoders import global_attention 

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

class Sal_global_Attention(nn.Module):
    """
    In this model, we take salgan architecture and chnge just maxpooling layer on 4 th and 5 th conv block to kernel size == 4 , downscal it by factor of 4 in order to have receptive field which cover the whole frame
    """
    def  __init__(self, use_gpu=True):
        super(Sal_global_Attention,self).__init__()

        self.use_gpu = use_gpu
        # Create encoder based on VGG16 architecture as pointed on salgan architecture 
        # Change just 4,5 th maxpooling lyer to 4 scale instead of 2 
        Global_Attention_Encoder = global_attention

        # select only convolutional layers first 5 conv blocks ,cahnge maxpooling=> enlarge receptive field
        # each neuron on bottelneck will see (580,580) all viewports  ,
        # input (576,288) , features numbers on bottelneck (9*4)*512, exclude last maxpooling
        encoder = torch.nn.Sequential(*Global_Attention_Encoder)

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

        decoder = torch.nn.Sequential(*decoder_list)

        # aggreegate the full architecture encoder-decoder of Sal_global_Attention
        self.Sal_global_Attention = torch.nn.Sequential(*(list(encoder.children())+list(decoder.children())))

        print("Model initialized, Sal_global_Attention")
        print("architecture len :",str(len(self.Sal_global_Attention)))

    def forward(self, input):
        x = self.Sal_global_Attention[:](input)
        
        #batch_size = x.data.size()[0]
        #print(batch_size)

        #spatial_size = x.data.size()[2:]
        #print(spatial_size)

        return x #x is a saliency map at this point