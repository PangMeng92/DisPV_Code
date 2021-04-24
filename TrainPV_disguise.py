# -*- coding: utf-8 -*-
from modelPV import Discriminator, DiscriminatorV, Generator
from readerDisguise import get_batch
import torch as t
from torch import nn, optim
from configPV_disguise import get_config
import numpy as np
from tensorboardX import SummaryWriter
from torchvision import transforms
import torchvision.utils as vutils
import cv2 as cv
import visdom
import os
import torch.nn.functional as F

def one_hot(label,depth):
    ones = t.sparse.torch.eye(depth)
    return ones.index_select(0,label)


def trainPV(conf):
    vis = visdom.Visdom()
    train_loader = get_batch(conf.root, conf.file, conf.batch_size)    #xxxxxx
    D = Discriminator(conf.nd, 3).cuda()
    Dv = DiscriminatorV(conf.nd, 3).cuda()
    G = Generator(3).cuda()
    D.train()
    Dv.train()
    G.train()

    optimizer_D = optim.Adam(D.parameters(),
                             lr=conf.lr,betas=(conf.beta1,conf.beta2))
    optimizer_Dv = optim.Adam(Dv.parameters(),
                             lr=conf.lr,betas=(conf.beta1,conf.beta2))
    optimizer_G = optim.Adam(G.parameters(), lr=conf.lr,
                                           betas=(conf.beta1, conf.beta2))
    loss_criterion = nn.CrossEntropyLoss()
    loss_criterion_gan = nn.BCEWithLogitsLoss()

    steps = 0
    # writer = SummaryWriter()
    flag_D_strong = False
    for epoch in range(1,conf.epochs+1):
        print('%d epoch ...'%(epoch))
        g_loss = 0
        for i, batch_data in enumerate(train_loader):
            D.zero_grad()
            G.zero_grad()
            batch_image = batch_data[0]
            batch_id_label = batch_data[1]-1
            batch_var_label = batch_data[2]
            batch_pro = batch_data[3]
            for j in range(conf.batch_size):
                if batch_var_label[j]==0:
                    batch_pro[j]=batch_image[j]
            batch_ones_label = t.ones(conf.batch_size)  
            batch_zeros_label = t.zeros(conf.batch_size)
            
            #cuda
            batch_image, batch_id_label, batch_var_label, batch_pro, batch_ones_label, batch_zeros_label = \
                batch_image.cuda(), batch_id_label.cuda(), batch_var_label.cuda(), batch_pro.cuda(), batch_ones_label.cuda(), batch_zeros_label.cuda()


            gen_pro, gen_var, gen_ori = G(batch_image, conf.batch_size)

            steps += 1

            if flag_D_strong:

                if i%5 == 0:
                    # Discriminator 
                    flag_D_strong, real_output,  syn_output = Learn_D(D, loss_criterion, loss_criterion_gan, optimizer_D, batch_image, batch_pro, gen_pro, \
                                            batch_id_label, batch_ones_label, batch_zeros_label, epoch, steps, conf.nd, conf)
                    # Discriminator
                    var_output = Learn_Dv(Dv, loss_criterion, optimizer_Dv, gen_var, batch_id_label, epoch, steps, conf.nd, conf)

                else:
                    # Generator
                    g_loss = Learn_G(D, Dv, loss_criterion, loss_criterion_gan, optimizer_G , batch_image, gen_pro, gen_var, gen_ori,\
                            batch_id_label, batch_ones_label, epoch, steps, conf.nd, conf)
            else:

                if i%2==0:
                    # Discriminator 
                    flag_D_strong, real_output,  syn_output = Learn_D(D, loss_criterion, loss_criterion_gan, optimizer_D, batch_image, batch_pro, gen_pro, \
                                            batch_id_label, batch_ones_label, batch_zeros_label, epoch, steps, conf.nd, conf)
                    # Discriminator
                    var_output = Learn_Dv(Dv, loss_criterion, optimizer_Dv, gen_var, batch_id_label, epoch, steps, conf.nd, conf)

                else:
                    # Generator
                    g_loss = Learn_G(D, Dv, loss_criterion, loss_criterion_gan, optimizer_G , batch_image, gen_pro, gen_var, gen_ori,\
                            batch_id_label, batch_ones_label, epoch, steps, conf.nd, conf)

            if i % 10 == 0:
                gen_pro = gen_pro.cpu().data.numpy()/2+0.5
                gen_var = gen_var.cpu().data.numpy()/2+0.5
                gen_ori = gen_ori.cpu().data.numpy()/2+0.5
                batch_image = batch_image.cpu().data.numpy()/2+0.5
                vis.images(gen_pro,nrow=4,win='gen_pro')
                vis.images(gen_var,nrow=4,win='gen_var')
                vis.images(gen_ori,nrow=4,win='gen_ori')
                vis.images(batch_image,nrow=4,win='original')
                print('%d steps loss is  %f'%(steps,g_loss))
                
        if epoch%20 ==0:
            msg = 'Saving checkpoint :{}'.format(epoch)    #restore from epoch+1
            print(msg)
            G_state_list = G.state_dict()
            D_state_list = D.state_dict()
            t.save({
                'epoch':epoch,
                'g_net_list':G_state_list,
                'd_net_list' :D_state_list
            },
            os.path.join(conf.save_dir,'%04d.pth'% epoch))

    # writer.close()


def Learn_D(D_model, loss_criterion, loss_criterion_gan, optimizer_D, batch_image, batch_pro, gen_pro, \
            batch_id_label, batch_ones_label, batch_zeros_label, epoch, steps, Nd, args):

    real_output = D_model(batch_image)
    pro_output = D_model(batch_pro)
    syn_output = D_model(gen_pro.detach()) # .detach() 
    
    
    L_id    = loss_criterion(real_output[:, :Nd], batch_id_label)
    L_gan   = loss_criterion_gan(pro_output[:, Nd], batch_ones_label) + loss_criterion_gan(syn_output[:, Nd], batch_zeros_label)
    
    
    d_loss = L_gan + 5*L_id    # lighting 1,5,  pose  1,5 （1，10）, Fuse 1,5

    d_loss.backward()
    optimizer_D.step()

    # Discriminator 
    flag_D_strong = Is_D_strong(real_output, syn_output, batch_id_label, Nd)

    return flag_D_strong,  real_output,  syn_output



def Learn_Dv(Dv_model, loss_criterion, optimizer_Dv, gen_var, batch_id_label, epoch, steps, Nd, args):

    var_output = Dv_model(gen_var.detach()) # .detach() 
    
    
    L_id    = loss_criterion(var_output[:, :Nd], batch_id_label)
    
    
    d_loss = L_id

    d_loss.backward()
    optimizer_Dv.step()


    return var_output



def Learn_G(D_model, Dv_model, loss_criterion, loss_criterion_gan, optimizer_G, batch_image, gen_pro, gen_var, gen_ori,\
            batch_id_label, batch_ones_label, epoch, steps, Nd, args):

    pro_output = D_model(gen_pro)
    var_output = Dv_model(gen_var)
    
    L_id    = loss_criterion(pro_output[:, :Nd], batch_id_label)
    L_gan   = loss_criterion_gan(pro_output[:, Nd], batch_ones_label)
    
    y = var_output[:, :Nd]
    y = F.softmax(y,dim=1)
    y1 = y*t.log(y)
    L_var = 1.0/args.batch_size*y1.sum()
    
    L_rec = (batch_image - gen_ori).pow(2).sum()/args.batch_size
    
    g_loss = 1*L_gan + 5*L_id + 0.5*L_var + 0.1*L_rec
    g_loss.backward()
    optimizer_G.step()
    a = g_loss.cpu().data.item()
    return a


def Is_D_strong(real_output, syn_output, id_label_tensor, Nd, thresh=0.9):
    """
    # Discriminator 

    """
    _, id_real_ans = t.max(real_output[:, :Nd], 1)
    _, id_syn_ans = t.max(syn_output[:, :Nd], 1)

    id_real_precision = (id_real_ans==id_label_tensor).type(t.FloatTensor).sum() / real_output.size()[0]
    gan_real_precision = (real_output[:,Nd].sigmoid()>=0.5).type(t.FloatTensor).sum() / real_output.size()[0]
    gan_syn_precision = (syn_output[:,Nd].sigmoid()<0.5).type(t.FloatTensor).sum() / syn_output.size()[0]

    total_precision = (id_real_precision+gan_real_precision+gan_syn_precision)/4

    total_precision = total_precision.data.item()
    if total_precision>=thresh:
        flag_D_strong = True
    else:
        flag_D_strong = False

    return flag_D_strong


def generatePV(conf):
    vis = visdom.Visdom()
    G = Generator(3).cuda()
    T=t.load('D:\PytorchProject\EncodedPV\saved_modelDisguise\E%3d.pth'%340)  
    
    G.load_state_dict(T['g_net_list'])
    G.eval()
    
    train_loader = get_batch(conf.root, conf.file, conf.batch_size) 
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
                gen_pro = gen_pro.cpu().data.numpy()/2+0.5
                gen_var = gen_var.cpu().data.numpy()/2+0.5
                gen_ori = gen_ori.cpu().data.numpy()/2+0.5
                batch_image = batch_image.cpu().data.numpy()/2+0.5
                               
                vis.images(gen_pro,nrow=1,win='gen_pro')
                vis.images(gen_var,nrow=1,win='gen_var')
                vis.images(gen_ori,nrow=1,win='gen_ori')
                vis.images(batch_image,nrow=1,win='original')
                
                
if __name__=='__main__':
    conf = get_config()
    
    print(conf)
    if conf.TrainTag:
        trainPV(conf)
    else:
        generatePV(conf) 
    

