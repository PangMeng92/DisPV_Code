# -*- coding: utf-8 -*-
#from modeldrop3 import Discriminator,Generator
from modelPV import Discriminator,Generator
from readerDisguise import get_batch
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

def generateImg(conf):
#    vis = visdom.Visdom()
    G = Generator(3).cuda()  
    
    T=t.load('D:\PytorchProject\EncodedPV\saved_modelDisguise\E%3d.pth'%340)  
    
    G.load_state_dict(T['g_net_list'])
    G.eval()
       
#    train_loader = get_batch(conf.root,conf.batch_size) #xxxxxx
    train_loader = get_batch(conf.root, conf.file, conf.batch_size) #xxxxxx
    steps = 0
    for epoch in range(1,conf.epochs+1):
        print('%d epoch ...'%(epoch))
        for i, batch_data in enumerate(train_loader):
            batch_image = batch_data[0]
            batch_id_label = batch_data[1]-1
            batch_var_label = batch_data[2]
            batch_pro = batch_data[3]
            
            batch_ones_label = t.ones(conf.batch_size)  
            batch_zeros_label = t.zeros(conf.batch_size)



            #cuda
            batch_image, batch_id_label, batch_var_label, batch_ones_label, batch_zeros_label = \
                batch_image.cuda(), batch_id_label.cuda(), batch_var_label.cuda(), batch_ones_label.cuda(), batch_zeros_label.cuda()



            gen_pro, gen_var, gen_ori = G(batch_image, conf.batch_size)
            

            steps += 1
            if i % 1 == 0:
                
                batch_image = batch_image.cpu().data.numpy()/2+0.5
                batch_image = np.squeeze(batch_image)
                batch_image = batch_image.transpose(1, 2, 0) 
                save_ori = '{}_ori_test'.format(conf.savefig)
                filename_ori = os.path.join(save_ori, '{}.png'.format(str(i+1)))
                imageio.imwrite(filename_ori, batch_image)
##                                
                batch_pro = batch_pro.cpu().data.numpy()/2+0.5
                batch_pro = np.squeeze(batch_pro)
                batch_pro = batch_pro.transpose(1, 2, 0) 
                save_pro = '{}_pro_test'.format(conf.savefig)
                filename_pro = os.path.join(save_pro, '{}.png'.format(str(i+1)))
                imageio.imwrite(filename_pro, batch_pro)
#        
                gen_pro = gen_pro.cpu().data.numpy()/2+0.5
                gen_pro = np.squeeze(gen_pro)
                gen_pro = gen_pro.transpose(1, 2, 0) 
                save_gen = '{}_genpro_test'.format(conf.savefig)
                filename_gen = os.path.join(save_gen, '{}.png'.format(str(i+1)))
                imageio.imwrite(filename_gen, gen_pro)
                
                gen_var = gen_var.cpu().data.numpy()/2+0.5
                gen_var = np.squeeze(gen_var)
                gen_var = gen_var.transpose(1, 2, 0) 
                save_gen = '{}_genvar_test'.format(conf.savefig)
                filename_gen = os.path.join(save_gen, '{}.png'.format(str(i+1)))
                imageio.imwrite(filename_gen, gen_var)
                
                gen_ori = gen_ori.cpu().data.numpy()/2+0.5
                gen_ori = np.squeeze(gen_ori)
                gen_ori = gen_ori.transpose(1, 2, 0) 
                save_gen = '{}_genori_test'.format(conf.savefig)
                filename_gen = os.path.join(save_gen, '{}.png'.format(str(i+1)))
                imageio.imwrite(filename_gen, gen_ori)
                
                
                                
if __name__=='__main__':
    conf = get_config()
    print(conf)
    generateImg(conf) 
    
