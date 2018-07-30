#!/usr/bin/python
from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, pickle, cv2
from datetime import datetime
import numpy as np

### torch lib
import torch
import torch.nn as nn

### custom lib
from networks.resample2d_package.modules.resample2d import Resample2d
import networks
import utils


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fast Blind Video Temporal Consistency')
    

    ### testing options
    parser.add_argument('-task',            type=str,     required=True,            help='evaluated task')
    parser.add_argument('-method',          type=str,     required=True,            help='test model name')
    parser.add_argument('-dataset',         type=str,     required=True,            help='test datasets')
    parser.add_argument('-phase',           type=str,     default="test",           choices=["train", "test"])
    parser.add_argument('-data_dir',        type=str,     default='data',           help='path to data folder')
    parser.add_argument('-list_dir',        type=str,     default='lists',          help='path to list folder')
    parser.add_argument('-redo',            action="store_true",                    help='redo evaluation')

    opts = parser.parse_args()
    opts.cuda = True

    print(opts)

    output_dir = os.path.join(opts.data_dir, opts.phase, opts.method, opts.task, opts.dataset)

    ## print average if result already exists
    metric_filename = os.path.join(output_dir, "WarpError.txt")
    if os.path.exists(metric_filename) and not opts.redo:
        print("Output %s exists, skip..." %metric_filename)

        cmd = 'tail -n1 %s' %metric_filename
        utils.run_cmd(cmd)
        sys.exit()
    

    ## flow warping layer
    device = torch.device("cuda" if opts.cuda else "cpu")
    flow_warping = Resample2d().to(device)

    ### load video list
    list_filename = os.path.join(opts.list_dir, "%s_%s.txt" %(opts.dataset, opts.phase))
    with open(list_filename) as f:
        video_list = [line.rstrip() for line in f.readlines()]

    ### start evaluation
    err_all = np.zeros(len(video_list))

    for v in range(len(video_list)):

        video = video_list[v]

        frame_dir = os.path.join(opts.data_dir, opts.phase, opts.method, opts.task, opts.dataset, video)
        occ_dir = os.path.join(opts.data_dir, opts.phase, "fw_occlusion", opts.dataset, video)
        flow_dir = os.path.join(opts.data_dir, opts.phase, "fw_flow", opts.dataset, video)
        
        frame_list = glob.glob(os.path.join(frame_dir, "*.jpg"))

        err = 0
        for t in range(1, len(frame_list)):
            
            
            ### load input images
            filename = os.path.join(frame_dir, "%05d.jpg" %(t - 1)) 
            img1 = utils.read_img(filename)
            filename = os.path.join(frame_dir, "%05d.jpg" %(t))
            img2 = utils.read_img(filename)

            print("Evaluate Warping Error on %s-%s: video %d / %d, %s" %(opts.dataset, opts.phase, v + 1, len(video_list), filename))


            ### load flow
            filename = os.path.join(flow_dir, "%05d.flo" %(t-1))
            flow = utils.read_flo(filename)

            ### load occlusion mask
            filename = os.path.join(occ_dir, "%05d.png" %(t-1))
            occ_mask = utils.read_img(filename)
            noc_mask = 1 - occ_mask

            with torch.no_grad():

                ## convert to tensor
                img2 = utils.img2tensor(img2).to(device)
                flow = utils.img2tensor(flow).to(device)

                ## warp img2
                warp_img2 = flow_warping(img2, flow)

                ## convert to numpy array
                warp_img2 = utils.tensor2img(warp_img2)


            ## compute warping error
            diff = np.multiply(warp_img2 - img1, noc_mask)
            
            N = np.sum(noc_mask)
            if N == 0:
                N = diff.shape[0] * diff.shape[1] * diff.shape[2]

            err += np.sum(np.square(diff)) / N
            
        err_all[v] = err / (len(frame_list) - 1)


    print("\nAverage Warping Error = %f\n" %(err_all.mean()))

    err_all = np.append(err_all, err_all.mean())
    print("Save %s" %metric_filename)
    np.savetxt(metric_filename, err_all, fmt="%f")
