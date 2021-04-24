# -*- coding: utf-8 -*-
from easydict import EasyDict as edict

def get_config():
    
    conf = edict()
    conf.batch_size = 1
    conf.lr = 0.0002
    conf.beta1 = 0.5
    conf.beta2 = 0.999
    conf.epochs = 1
    conf.save_dir = './saved_modelDisguise'
    conf.root='./PEAL_data'
    conf.savefig='./PEAL'
    conf.file='./dataset/LoadPEALA1.txt'
    conf.file1='./dataset/LoadPEALB.txt'
    conf.nd = 100
    conf.TrainTag=False
    return conf