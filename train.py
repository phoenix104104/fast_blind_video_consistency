#!/usr/bin/python
from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, copy, pickle
from datetime import datetime
import numpy as np

### torch lib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

### custom lib
from networks.resample2d_package.modules.resample2d import Resample2d
import networks
import datasets
import utils




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fast Blind Video Temporal Consistency")

    ### model options
    parser.add_argument('-model',           type=str,     default="TransformNet",   help='TransformNet') 
    parser.add_argument('-nf',              type=int,     default=32,               help='#Channels in conv layer')
    parser.add_argument('-blocks',          type=int,     default=5,                help='#ResBlocks') 
    parser.add_argument('-norm',            type=str,     default='IN',             choices=["BN", "IN", "none"],   help='normalization layer')
    parser.add_argument('-model_name',      type=str,     default='none',           help='path to save model')

    ### dataset options
    parser.add_argument('-datasets_tasks',  type=str,     default='W3_D1_C1_I1',    help='dataset-task pairs list')
    parser.add_argument('-data_dir',        type=str,     default='data',           help='path to data folder')
    parser.add_argument('-list_dir',        type=str,     default='lists',          help='path to lists folder')
    parser.add_argument('-checkpoint_dir',  type=str,     default='checkpoints',    help='path to checkpoint folder')
    parser.add_argument('-crop_size',       type=int,     default=192,              help='patch size')
    parser.add_argument('-geometry_aug',    type=int,     default=1,                help='geometry augmentation (rotation, scaling, flipping)')
    parser.add_argument('-order_aug',       type=int,     default=1,                help='temporal ordering augmentation')
    parser.add_argument('-scale_min',       type=float,   default=0.5,              help='min scaling factor')
    parser.add_argument('-scale_max',       type=float,   default=2.0,              help='max scaling factor')
    parser.add_argument('-sample_frames',   type=int,     default=11,               help='#frames for training')
        
    ### loss optinos
    parser.add_argument('-alpha',           type=float,   default=50.0,             help='alpha for computing visibility mask')
    parser.add_argument('-loss',            type=str,     default="L1",             help="optimizer [Options: SGD, ADAM]")
    parser.add_argument('-w_ST',            type=float,   default=100,              help='weight for short-term temporal loss')
    parser.add_argument('-w_LT',            type=float,   default=100,              help='weight for long-term temporal loss')
    parser.add_argument('-w_VGG',           type=float,   default=10,               help='weight for VGG perceptual loss')
    parser.add_argument('-VGGLayers',       type=str,     default="4",              help="VGG layers for perceptual loss, combinations of 1, 2, 3, 4")
                

    ### training options
    parser.add_argument('-solver',          type=str,     default="ADAM",           choices=["SGD", "ADAIM"],   help="optimizer")
    parser.add_argument('-momentum',        type=float,   default=0.9,              help='momentum for SGD')
    parser.add_argument('-beta1',           type=float,   default=0.9,              help='beta1 for ADAM')
    parser.add_argument('-beta2',           type=float,   default=0.999,            help='beta2 for ADAM')
    parser.add_argument('-weight_decay',    type=float,   default=0,                help='weight decay')
    parser.add_argument('-batch_size',      type=int,     default=4,                help='training batch size')
    parser.add_argument('-train_epoch_size',type=int,     default=1000,             help='train epoch size')
    parser.add_argument('-valid_epoch_size',type=int,     default=100,              help='valid epoch size')
    parser.add_argument('-epoch_max',       type=int,     default=100,              help='max #epochs')


    ### learning rate options
    parser.add_argument('-lr_init',         type=float,   default=1e-4,             help='initial learning Rate')
    parser.add_argument('-lr_offset',       type=int,     default=20,               help='epoch to start learning rate drop [-1 = no drop]')
    parser.add_argument('-lr_step',         type=int,     default=20,               help='step size (epoch) to drop learning rate')
    parser.add_argument('-lr_drop',         type=float,   default=0.5,              help='learning rate drop ratio')
    parser.add_argument('-lr_min_m',        type=float,   default=0.1,              help='minimal learning Rate multiplier (lr >= lr_init * lr_min)')
    

    ### other options
    parser.add_argument('-seed',            type=int,     default=9487,             help='random seed to use')
    parser.add_argument('-threads',         type=int,     default=8,                help='number of threads for data loader to use')
    parser.add_argument('-suffix',          type=str,     default='',               help='name suffix')
    parser.add_argument('-gpu',             type=int,     default=0,                help='gpu device id')
    parser.add_argument('-cpu',             action='store_true',                    help='use cpu?')
    
    opts = parser.parse_args()

    ### adjust options
    opts.cuda = (opts.cpu != True)
    opts.lr_min = opts.lr_init * opts.lr_min_m
    
    ### default model name
    if opts.model_name == 'none':
        
        opts.model_name = "%s_B%d_nf%d_%s" %(opts.model, opts.blocks, opts.nf, opts.norm)

        opts.model_name = "%s_T%d_%s_pw%d_%sLoss_a%s_wST%s_wHT%s_wVGG%s_L%s_%s_lr%s_off%d_step%d_drop%s_min%s_es%d_bs%d" \
                %(opts.model_name, opts.sample_frames, \
                  opts.datasets_tasks, opts.crop_size, opts.loss, str(opts.alpha), \
                  str(opts.w_ST), str(opts.w_LT), str(opts.w_VGG), opts.VGGLayers, \
                  opts.solver, str(opts.lr_init), opts.lr_offset, opts.lr_step, str(opts.lr_drop), str(opts.lr_min), \
                  opts.train_epoch_size, opts.batch_size)
    
    ### check VGG layers
    opts.VGGLayers = [int(layer) for layer in list(opts.VGGLayers)]
    opts.VGGLayers.sort()

    if opts.VGGLayers[0] < 1 or opts.VGGLayers[-1] > 4:
        raise Exception("Only support VGG Loss on Layers 1 ~ 4")

    opts.VGGLayers = [layer - 1 for layer in list(opts.VGGLayers)] ## shift index to 0 ~ 3

    if opts.suffix != "":
        opts.model_name += "_%s" %opts.suffix


    opts.size_multiplier = 2 ** 6 ## Inputs to FlowNet need to be divided by 64
    
    print(opts)


    torch.manual_seed(opts.seed)
    if opts.cuda:
        torch.cuda.manual_seed(opts.seed)


    ### model saving directory
    opts.model_dir = os.path.join(opts.checkpoint_dir, opts.model_name)
    print("========================================================")
    print("===> Save model to %s" %opts.model_dir)
    print("========================================================")
    if not os.path.isdir(opts.model_dir):
        os.makedirs(opts.model_dir)

    
    ### initialize model
    print('===> Initializing model from %s...' %opts.model)
    model = networks.__dict__[opts.model](opts, nc_in=12, nc_out=3)


    ### initialize optimizer
    if opts.solver == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=opts.lr_init, momentum=opts.momentum, weight_decay=opts.weight_decay)
    elif opts.solver == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=opts.lr_init, weight_decay=opts.weight_decay, betas=(opts.beta1, opts.beta2))
    else:
        raise Exception("Not supported solver (%s)" %opts.solver)


    ### resume latest model
    name_list = glob.glob(os.path.join(opts.model_dir, "model_epoch_*.pth"))
    epoch_st = 0
    if len(name_list) > 0:
        epoch_list = []
        for name in name_list:
            s = re.findall(r'\d+', os.path.basename(name))[0]
            epoch_list.append(int(s))

        epoch_list.sort()
        epoch_st = epoch_list[-1]


    if epoch_st > 0:

        print('=====================================================================')
        print('===> Resuming model from epoch %d' %epoch_st)
        print('=====================================================================')

        ### resume latest model and solver
        model, optimizer = utils.load_model(model, optimizer, opts, epoch_st)

    else:
        ### save epoch 0
        utils.save_model(model, optimizer, opts)


    print(model)

    num_params = utils.count_network_parameters(model)

    print('\n=====================================================================')
    print("===> Model has %d parameters" %num_params)
    print('=====================================================================')


    ### initialize loss writer
    loss_dir = os.path.join(opts.model_dir, 'loss')
    loss_writer = SummaryWriter(loss_dir)
    
    

    ### Load pretrained FlowNet2
    opts.rgb_max = 1.0
    opts.fp16 = False

    FlowNet = networks.FlowNet2(opts, requires_grad=False)
    model_filename = os.path.join("pretrained_models", "FlowNet2_checkpoint.pth.tar")
    print("===> Load %s" %model_filename)
    checkpoint = torch.load(model_filename)
    FlowNet.load_state_dict(checkpoint['state_dict'])


    ### Load pretrained VGG
    VGG = networks.Vgg16(requires_grad=False)

    ### convert to GPU
    device = torch.device("cuda" if opts.cuda else "cpu")

    model = model.to(device)
    FlowNet = FlowNet.to(device)
    VGG = VGG.to(device)
    
    model.train()

    
    ### create dataset
    train_dataset = datasets.MultiFramesDataset(opts, "train")

    
    ### start training
    while model.epoch < opts.epoch_max:

        model.epoch += 1

        ### re-generate train data loader for every epoch
        data_loader = utils.create_data_loader(train_dataset, opts, "train")

        ### update learning rate
        current_lr = utils.learning_rate_decay(opts, model.epoch)

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        ## submodule
        flow_warping = Resample2d().to(device)
        downsampler = nn.AvgPool2d((2, 2), stride=2).to(device)


        ### criterion and loss recorder
        if opts.loss == 'L2':
            criterion = nn.MSELoss(size_average=True)
        elif opts.loss == 'L1':
            criterion = nn.L1Loss(size_average=True)
        else:
            raise Exception("Unsupported criterion %s" %opts.loss)
        
        
        ### start epoch
        ts = datetime.now()
        
        for iteration, batch in enumerate(data_loader, 1):
            
            total_iter = (model.epoch - 1) * opts.train_epoch_size + iteration

            ### convert data to cuda
            frame_i = []
            frame_p = []
            for t in range(opts.sample_frames):
                frame_i.append(batch[t * 2].to(device))
                frame_p.append(batch[t * 2 + 1].to(device))

            frame_o = []
            frame_o.append(frame_p[0]) ## first frame

            ### get batch time
            data_time = datetime.now() - ts

            ts = datetime.now()

            ### clear gradients
            optimizer.zero_grad()

            lstm_state = None
            ST_loss = 0
            LT_loss = 0
            VGG_loss = 0

            ### forward
            for t in range(1, opts.sample_frames):

                frame_i1 = frame_i[t - 1]
                frame_i2 = frame_i[t]
                frame_p2 = frame_p[t]

                if t == 1:
                    frame_o1 = frame_p[t - 1]
                else: 
                    frame_o1 = frame_o2.detach()    ## previous output frame

                frame_o1.requires_grad = False 

                ### model input        
                inputs = torch.cat((frame_p2, frame_o1, frame_i2, frame_i1), dim=1)
                
                ### forward model
                output, lstm_state = model(inputs, lstm_state)

                ### residual learning
                frame_o2 = output + frame_p2

                ## detach from graph and avoid memory accumulation
                lstm_state = utils.repackage_hidden(lstm_state)

                frame_o.append(frame_o2)


                ### short-term temporal loss
                if opts.w_ST > 0:
                
                    ### compute flow (from I2 to I1)
                    flow_i21 = FlowNet(frame_i2, frame_i1)

                    ### warp I1 and O1
                    warp_i1 = flow_warping(frame_i1, flow_i21)
                    warp_o1 = flow_warping(frame_o1, flow_i21)

                    ### compute non-occlusion mask: exp(-alpha * || F_i2 - Warp(F_i1) ||^2 )
                    noc_mask2 = torch.exp( -opts.alpha * torch.sum(frame_i2 - warp_i1, dim=1).pow(2) ).unsqueeze(1)
                    
                    ST_loss += opts.w_ST * criterion(frame_o2 * noc_mask2, warp_o1 * noc_mask2)
                

                ### perceptual loss
                if opts.w_VGG > 0:

                    ### normalize
                    frame_o2_n = utils.normalize_ImageNet_stats(frame_o2)
                    frame_p2_n = utils.normalize_ImageNet_stats(frame_p2)
                    
                    ### extract VGG features
                    features_p2 = VGG(frame_p2_n, opts.VGGLayers[-1])
                    features_o2 = VGG(frame_o2_n, opts.VGGLayers[-1])

                    VGG_loss_all = []
                    for l in opts.VGGLayers:
                        VGG_loss_all.append( criterion(features_o2[l], features_p2[l]) )
                        
                    VGG_loss += opts.w_VGG * sum(VGG_loss_all)

            ## end of forward


            ### long-term temporal loss
            if opts.w_LT > 0:

                t1 = 0
                for t2 in range(t1 + 2, opts.sample_frames):

                    frame_i1 = frame_i[t1]
                    frame_i2 = frame_i[t2]

                    frame_o1 = frame_o[t1].detach() ## make a new Variable to avoid backwarding gradient
                    frame_o1.requires_grad = False

                    frame_o2 = frame_o[t2]

                    ### compute flow (from I2 to I1)
                    flow_i21 = FlowNet(frame_i2, frame_i1)
                    
                    ### warp I1 and O1
                    warp_i1 = flow_warping(frame_i1, flow_i21)
                    warp_o1 = flow_warping(frame_o1, flow_i21)

                    ### compute non-occlusion mask: exp(-alpha * || F_i2 - Warp(F_i1) ||^2 )
                    noc_mask2 = torch.exp( -opts.alpha * torch.sum(frame_i2 - warp_i1, dim=1).pow(2) ).unsqueeze(1)

                    LT_loss += opts.w_LT * criterion(frame_o2 * noc_mask2, warp_o1 * noc_mask2)

                ### end of t2
            ### end of w_LT
            

            ### overall loss
            overall_loss = ST_loss + LT_loss + VGG_loss

            ### backward loss
            overall_loss.backward()

            ### update parameters
            optimizer.step()
                
            network_time = datetime.now() - ts


            ### print training info
            info = "[GPU %d]: " %(opts.gpu)
            info += "Epoch %d; Batch %d / %d; " %(model.epoch, iteration, len(data_loader))
            info += "lr = %s; " %(str(current_lr))

            ## number of samples per second
            batch_freq = opts.batch_size / (data_time.total_seconds() + network_time.total_seconds())
            info += "data loading = %.3f sec, network = %.3f sec, batch = %.3f Hz\n" %(data_time.total_seconds(), network_time.total_seconds(), batch_freq)
            
            info += "\tmodel = %s\n" %opts.model_name

            ### print and record loss
            if opts.w_ST > 0:
                loss_writer.add_scalar('ST_loss', ST_loss.item(), total_iter)
                info += "\t\t%25s = %f\n" %("ST_loss", ST_loss.item())

            if opts.w_LT > 0:
                loss_writer.add_scalar('LT_loss', LT_loss.item(), total_iter)
                info += "\t\t%25s = %f\n" %("LT_loss", LT_loss.item())

            if opts.w_VGG > 0:
                loss_writer.add_scalar('VGG_loss', VGG_loss.item(), total_iter)
                info += "\t\t%25s = %f\n" %("VGG_loss", VGG_loss.item())

            loss_writer.add_scalar('Overall_loss', overall_loss.item(), total_iter)
            info += "\t\t%25s = %f\n" %("Overall_loss", overall_loss.item())

            print(info)

        ### end of epoch

        ### save model
        utils.save_model(model, optimizer, opts)


            
