import torch
from torch import nn, optim
class Discriminator(nn.Module):
    """
    Multi-task discriminator D

    #    """

    def __init__(self, Nd, channel_num):
        super(Discriminator, self).__init__()
        convLayers = [
            nn.Conv2d(channel_num, 32, 3, 1, 1, bias=False), # Bxchx96x96 -> Bx32x96x96
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False), # Bx32x96x96 -> Bx64x96x96
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx64x96x96 -> Bx64x97x97
            nn.Conv2d(64, 64, 3, 2, 0, bias=False), # Bx64x97x97 -> Bx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False), # Bx64x48x48 -> Bx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), # Bx64x48x48 -> Bx128x48x48
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx128x48x48 -> Bx128x49x49
            nn.Conv2d(128, 128, 3, 2, 0, bias=False), #  Bx128x49x49 -> Bx128x24x24
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 96, 3, 1, 1, bias=False), #  Bx128x24x24 -> Bx96x24x24
            nn.BatchNorm2d(96),
            nn.ELU(),
            nn.Conv2d(96, 192, 3, 1, 1, bias=False), #  Bx96x24x24 -> Bx192x24x24
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx192x24x24 -> Bx192x25x25
            nn.Conv2d(192, 192, 3, 2, 0, bias=False), # Bx192x25x25 -> Bx192x12x12
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.Conv2d(192, 128, 3, 1, 1, bias=False), # Bx192x12x12 -> Bx128x12x12
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), # Bx128x12x12 -> Bx256x12x12
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx256x12x12 -> Bx256x13x13
            nn.Conv2d(256, 256, 3, 2, 0, bias=False),  # Bx256x13x13 -> Bx256x6x6
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 160, 3, 1, 1, bias=False), # Bx256x6x6 -> Bx160x6x6
            nn.BatchNorm2d(160),
            nn.ELU(),
            nn.Conv2d(160, 256, 3, 1, 1, bias=False), # Bx160x6x6 -> Bx320x6x6
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.AvgPool2d(6, stride=1), #  Bx320x6x6 -> Bx320x1x1
        ]

        self.convLayers = nn.Sequential(*convLayers)
        self.fc = nn.Linear(256, Nd+1)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)

    def forward(self, input):

        x = self.convLayers(input)

        x = x.view(-1, 256)

        x = self.fc(x) 

        return x


class DiscriminatorV(nn.Module):
    """
    Identity discriminator for variation

    """

    def __init__(self, Nd, channel_num):
        super(DiscriminatorV, self).__init__()
        convLayers = [
            nn.Conv2d(channel_num, 32, 3, 1, 1, bias=False), # Bxchx96x96 -> Bx32x96x96
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False), # Bx32x96x96 -> Bx64x96x96
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx64x96x96 -> Bx64x97x97
            nn.Conv2d(64, 64, 3, 2, 0, bias=False), # Bx64x97x97 -> Bx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False), # Bx64x48x48 -> Bx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), # Bx64x48x48 -> Bx128x48x48
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx128x48x48 -> Bx128x49x49
            nn.Conv2d(128, 128, 3, 2, 0, bias=False), #  Bx128x49x49 -> Bx128x24x24
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 96, 3, 1, 1, bias=False), #  Bx128x24x24 -> Bx96x24x24
            nn.BatchNorm2d(96),
            nn.ELU(),
            nn.Conv2d(96, 192, 3, 1, 1, bias=False), #  Bx96x24x24 -> Bx192x24x24
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx192x24x24 -> Bx192x25x25
            nn.Conv2d(192, 192, 3, 2, 0, bias=False), # Bx192x25x25 -> Bx192x12x12
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.Conv2d(192, 128, 3, 1, 1, bias=False), # Bx192x12x12 -> Bx128x12x12
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), # Bx128x12x12 -> Bx256x12x12
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx256x12x12 -> Bx256x13x13
            nn.Conv2d(256, 256, 3, 2, 0, bias=False),  # Bx256x13x13 -> Bx256x6x6
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 160, 3, 1, 1, bias=False), # Bx256x6x6 -> Bx160x6x6
            nn.BatchNorm2d(160),
            nn.ELU(),
            nn.Conv2d(160, 256, 3, 1, 1, bias=False), # Bx160x6x6 -> Bx320x6x6
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.AvgPool2d(6, stride=1), #  Bx320x6x6 -> Bx320x1x1
        ]

        self.convLayers = nn.Sequential(*convLayers)
        self.fc = nn.Linear(256, Nd)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)

    def forward(self, input):

        x = self.convLayers(input)

        x = x.view(-1, 256)

        x = self.fc(x) 

        return x
    
    
class Crop(nn.Module):

    def __init__(self, crop_list):
        super(Crop, self).__init__()

        # crop_lsit = [crop_top, crop_bottom, crop_left, crop_right]
        self.crop_list = crop_list

    def forward(self, x):
        B,C,H,W = x.size()
        x = x[:,:, self.crop_list[0] : H - self.crop_list[1] , self.crop_list[2] : W - self.crop_list[3]]

        return x


class Generator(nn.Module):
    """
    Encoder/Decoder generator

    """

    def __init__(self, channel_num):
        super(Generator, self).__init__()
        self.features = []

        G_enc_convLayers = [
            nn.Conv2d(channel_num, 32, 3, 1, 1, bias=False), # Bx3x96x96 -> Bx32x96x96
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False), # Bx32x96x96 -> Bx64x96x96
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx64x96x96 -> Bx64x97x97
            nn.Conv2d(64, 64, 3, 2, 0, bias=False), # Bx64x97x97 -> Bx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False), # Bx64x48x48 -> Bx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), # Bx64x48x48 -> Bx128x48x48
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx128x48x48 -> Bx128x49x49
            nn.Conv2d(128, 128, 3, 2, 0, bias=False), #  Bx128x49x49 -> Bx128x24x24
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 96, 3, 1, 1, bias=False), #  Bx128x24x24 -> Bx96x24x24
            nn.BatchNorm2d(96),
            nn.ELU(),
            nn.Conv2d(96, 192, 3, 1, 1, bias=False), #  Bx96x24x24 -> Bx192x24x24
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx192x24x24 -> Bx192x25x25
            nn.Conv2d(192, 192, 3, 2, 0, bias=False), # Bx192x25x25 -> Bx192x12x12
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.Conv2d(192, 128, 3, 1, 1, bias=False), # Bx192x12x12 -> Bx128x12x12
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), # Bx128x12x12 -> Bx256x12x12
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx256x12x12 -> Bx256x13x13
            nn.Conv2d(256, 256, 3, 2, 0, bias=False),  # Bx256x13x13 -> Bx256x6x6
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 160, 3, 1, 1, bias=False), # Bx256x6x6 -> Bx160x6x6
            nn.BatchNorm2d(160),
            nn.ELU(),
            nn.Conv2d(160, 320, 3, 1, 1, bias=False), # Bx160x6x6 -> Bx320x6x6
            nn.BatchNorm2d(320),
            nn.ELU(),
        ]
        
        G_prototype_convLayers = [
            nn.Conv2d(320, 320, 3, 1, 1, bias=False), # Bx320x6x6 -> Bx320x6x6
            nn.BatchNorm2d(320),
            nn.ELU(),  
            nn.Conv2d(320, 320, 3, 1, 1, bias=False), # Bx320x6x6 -> Bx320x6x6
            nn.BatchNorm2d(320),
            nn.ELU(),  
            nn.Conv2d(320, 256, 3, 1, 1, bias=False), # Bx320x6x6 -> Bx256x6x6
            nn.BatchNorm2d(256),
            nn.ELU(), 
            nn.AvgPool2d(6, stride=1), #  Bx256x6x6 -> Bx256x1x1
        ]
        
        G_variation_convLayers = [
            nn.Conv2d(320, 320, 3, 1, 1, bias=False), # Bx320x6x6 -> Bx320x6x6
            nn.BatchNorm2d(320),
            nn.ELU(),  
            nn.Conv2d(320, 320, 3, 1, 1, bias=False), # Bx320x6x6 -> Bx320x6x6
            nn.BatchNorm2d(320),
            nn.ELU(),  
            nn.Conv2d(320, 256, 3, 1, 1, bias=False), # Bx320x6x6 -> Bx256x6x6
            nn.BatchNorm2d(256),
            nn.ELU(), 
            nn.AvgPool2d(6, stride=1), #  Bx256x6x6 -> Bx256x1x1
        ]
                
        self.G_enc_convLayers = nn.Sequential(*G_enc_convLayers)
        self.G_prototype_convLayers = nn.Sequential(*G_prototype_convLayers)
        self.G_variation_convLayers = nn.Sequential(*G_variation_convLayers)

        G_dec_convLayers = [
            nn.ConvTranspose2d(256,160, 3,1,1, bias=False), # Bx256x6x6 -> Bx160x6x6
            nn.BatchNorm2d(160),
            nn.ELU(),
            nn.ConvTranspose2d(160, 256, 3,1,1, bias=False), # Bx160x6x6 -> Bx256x6x6
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.ConvTranspose2d(256, 256, 3,2,0, bias=False), # Bx256x6x6 -> Bx256x13x13
            nn.BatchNorm2d(256),
            nn.ELU(),
            Crop([0, 1, 0, 1]),
            nn.ConvTranspose2d(256, 128, 3,1,1, bias=False), # Bx256x12x12 -> Bx128x12x12
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ConvTranspose2d(128, 192,  3,1,1, bias=False), # Bx128x12x12 -> Bx192x12x12
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.ConvTranspose2d(192, 192,  3,2,0, bias=False), # Bx128x12x12 -> Bx192x25x25
            nn.BatchNorm2d(192),
            nn.ELU(),
            Crop([0, 1, 0, 1]),
            nn.ConvTranspose2d(192, 96,  3,1,1, bias=False), # Bx192x24x24 -> Bx96x24x24
            nn.BatchNorm2d(96),
            nn.ELU(),
            nn.ConvTranspose2d(96, 128,  3,1,1, bias=False), # Bx96x24x24 -> Bx128x24x24
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ConvTranspose2d(128, 128,  3,2,0, bias=False), # Bx128x24x24 -> Bx128x49x49
            nn.BatchNorm2d(128),
            nn.ELU(),
            Crop([0, 1, 0, 1]),
            nn.ConvTranspose2d(128, 64,  3,1,1, bias=False), # Bx128x48x48 -> Bx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ConvTranspose2d(64, 64,  3,1,1, bias=False), # Bx64x48x48 -> Bx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ConvTranspose2d(64, 64,  3,2,0, bias=False), # Bx64x48x48 -> Bx64x97x97
            nn.BatchNorm2d(64),
            nn.ELU(),
            Crop([0, 1, 0, 1]),
            nn.ConvTranspose2d(64, 32,  3,1,1, bias=False), # Bx64x96x96 -> Bx32x96x96
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.ConvTranspose2d(32, channel_num,  3,1,1, bias=False), # Bx32x96x96 -> Bxchx96x96
            nn.Tanh(),
        ]

        self.G_dec_convLayers = nn.Sequential(*G_dec_convLayers)

        self.G_dec_fc = nn.Linear(256, 256*6*6)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)

            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)



    def forward(self, input, batch_size):

        x = self.G_enc_convLayers(input) # Bxchx96x96 -> Bx320x1x1
        profea = self.G_prototype_convLayers(x)
        varfea = self.G_variation_convLayers(x)
        
        profea = profea.view(-1,256)
        varfea = varfea.view(-1,256)
        xpv = profea + varfea

        self.profea = profea
        self.varfea = varfea
        self.features = xpv
        
        profea_fc = self.G_dec_fc(profea) 
        varfea_fc = self.G_dec_fc(varfea)
        xpv_fc = self.G_dec_fc(xpv)
        
        profea_fc = profea_fc.view(-1, 256, 6, 6)
        varfea_fc = varfea_fc.view(-1, 256, 6, 6)
        xpv_fc = xpv_fc.view(-1, 256, 6, 6) 
        
        pro = self.G_dec_convLayers(profea_fc)
        var = self.G_dec_convLayers(varfea_fc)
        xhat = self.G_dec_convLayers(xpv_fc) 

        return pro, var, xhat