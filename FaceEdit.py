# -*- coding: utf-8 -*-
from modelPV import Discriminator,Generator
from readerDisguise import get_batch, get_batch1
import torch as t
from torch import nn, optim
from configPV_disguise import get_config
import numpy as np
from tensorboardX import SummaryWriter
from torchvision import transforms
import torchvision.utils as vutils
import visdom
import os
from scipy import misc
import imageio

def one_hot(label,depth):
    ones = t.sparse.torch.eye(depth)
    return ones.index_select(0,label)

def generateNewFace(conf):
#    vis = visdom.Visdom()
    G = Generator(3).cuda()
    G1 = Generator(3).cuda()
    
    T=t.load('D:\PytorchProject\EncodedPV\saved_modelDisguise\E%3d.pth'%340)  
    T1=t.load('D:\PytorchProject\EncodedPV\saved_modelDisguise\E%3d.pth'%340)
    
    
    
    G.load_state_dict(T['g_net_list'])
    G.eval()
    
    G1.load_state_dict(T1['g_net_list'])
    G1.eval()

    train_loader = get_batch(conf.root, conf.file, conf.batch_size) #xxxxxx
    train_loader1 = get_batch1(conf.root, conf.file1, conf.batch_size)
    a=1
    for epoch in range(1,conf.epochs+1):
        print('%d epoch ...'%(epoch))
        for i, batch_data in enumerate(train_loader):
            for j, batch_data1 in enumerate(train_loader1):
            
                batch_image = batch_data[0]
                batch_pro = batch_data[3]
            
                batch_image1 = batch_data1[0]
                batch_pro1 = batch_data1[3]



            #cuda
                batch_image, batch_image1, batch_pro, batch_pro1 = batch_image.cuda(), batch_image1.cuda(), batch_pro.cuda(), batch_pro1.cuda()

                G(batch_image, conf.batch_size)
                G1(batch_image1,conf.batch_size)
                
                pro_features = G.profea
                var_features1 = G1.varfea
                
                features = pro_features+var_features1
                xpv_fc = G.G_dec_fc(features)
                xpv_fc = xpv_fc.view(-1, 256, 6, 6) 
                
                gen_pro, gen_var, gen_ori = G(batch_image, conf.batch_size)
                gen_pro1, gen_var1, gen_ori1 = G1(batch_image1, conf.batch_size)
                gen_new = G.G_dec_convLayers(xpv_fc)
                    
                
                gen_pro = gen_pro.cpu().data.numpy()/2+0.5
                gen_var1 = gen_var1.cpu().data.numpy()/2+0.5
                gen_new = gen_new.cpu().data.numpy()/2+0.5
                batch_image = batch_image.cpu().data.numpy()/2+0.5
                batch_image1 = batch_image1.cpu().data.numpy()/2+0.5
                
                
                if j % 1 == 0:
#                    vis.images(batch_image,nrow=1, win='batch_image', opts=dict(caption='Input face'))
#                    vis.images(batch_image1,nrow=1,win='batch_image1', opts=dict(caption='Target disguise'))
#                    vis.images(gen_new,nrow=1,win='gen_new', opts=dict(caption='Interpolated face'))
                   
                    
                    batch_image = np.squeeze(batch_image)
                    batch_image = batch_image.transpose(1, 2, 0)
                    batch_image1 = np.squeeze(batch_image1)
                    batch_image1 = batch_image1.transpose(1, 2, 0)
                    gen_new = np.squeeze(gen_new)
                    gen_new = gen_new.transpose(1, 2, 0) 
                    save_ori = '{}_gennew_test'.format(conf.savefig)
                    filename_a = os.path.join(save_ori, 'PersonA', 'A{}.png'.format(str(i+1)))
                    filename_b = os.path.join(save_ori, 'PersonB', 'B{}.png'.format(str(j+1)))
                    filename_c = os.path.join(save_ori, 'GenNew', 'A{}B{}.png'.format(str(i+1),(str(j+1))))
                    imageio.imwrite(filename_a, batch_image)
                    imageio.imwrite(filename_b, batch_image1)
                    imageio.imwrite(filename_c, gen_new) 
                
                                
if __name__=='__main__':
    conf = get_config()
    print(conf)
    generateNewFace(conf) 
    
