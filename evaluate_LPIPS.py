#!/usr/bin/python
from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, pickle, cv2
from datetime import datetime
import numpy as np

### torch lib
import torch

### custom lib
import utils



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='optical flow estimation')

    ### testing options
    parser.add_argument('-task',            type=str,     required=True,            help='evaluated task')
    parser.add_argument('-method',          type=str,     required=True,            help='test model name')
    parser.add_argument('-dataset',         type=str,     required=True,            help='test datasets')
    parser.add_argument('-phase',           type=str,     default="test",           choices=["train", "test"])
    parser.add_argument('-data_dir',        type=str,     default='data',           help='path to data folder')
    parser.add_argument('-list_dir',        type=str,     default='lists',          help='path to list folder')
    parser.add_argument('-LPIPS_dir',       type=str,     default='../PerceptualSimilarity',    help='path to LPIPS folder')
    parser.add_argument('-net',             type=str,     default="squeeze",        choices=["alex", "vgg", "squeeze"], help='LPIPS model')
    parser.add_argument('-redo',            action="store_true",                    help='redo evaluation')

    opts = parser.parse_args()
    print(opts)

    output_dir = os.path.join(opts.data_dir, opts.phase, opts.method, opts.task, opts.dataset)
    
    ## print average if result already exists
    metric_filename = os.path.join(output_dir, "LPIPS.txt")
    if os.path.exists(metric_filename) and not opts.redo:
        print("Output %s exists...skip" %metric_filename)

        cmd = 'tail -n1 %s' %metric_filename
        utils.run_cmd(cmd)
        sys.exit()
    

    ## import LPIPS
    sys.path.append(opts.LPIPS_dir)
    from models import dist_model as dm

    ## Initializing LPIPS model
    print("Initialize Distance model from %s" %opts.net)
    model = dm.DistModel()
    model.initialize(model='net-lin',net=opts.net, use_gpu=True, model_path=os.path.join(opts.LPIPS_dir, 'weights/%s.pth' %opts.net))

    ### load video list
    list_filename = os.path.join(opts.list_dir, "%s_%s.txt" %(opts.dataset, opts.phase))
    with open(list_filename) as f:
        video_list = [line.rstrip() for line in f.readlines()]

    ### start evaluation
    dist_all = np.zeros(len(video_list))

    for v in range(len(video_list)):

        video = video_list[v]

        input_dir = os.path.join(opts.data_dir, opts.phase, "input", opts.dataset, video)
        process_dir = os.path.join(opts.data_dir, opts.phase, "processed", opts.task, opts.dataset, video)
        output_dir = os.path.join(opts.data_dir, opts.phase, opts.method, opts.task, opts.dataset, video)

        frame_list = glob.glob(os.path.join(input_dir, "*.jpg"))

        dist = 0
        for t in range(1, len(frame_list)):
            
            ### load processed images
            filename = os.path.join(process_dir, "%05d.jpg" %(t))
            P = utils.read_img(filename)

            ### load output images
            filename = os.path.join(output_dir, "%05d.jpg" %(t))
            O = utils.read_img(filename)

            print("Evaluate LPIPS on %s-%s: video %d / %d, %s" %(opts.dataset, opts.phase, v + 1, len(video_list), filename))

            ### convert to tensor
            P = utils.img2tensor(P)
            O = utils.img2tensor(O)

            ### scale to [-1, 1]
            P = P * 2.0 - 1
            O = O * 2.0 - 1

            dist += model.forward(P, O)[0]

        dist_all[v] = dist / (len(frame_list) - 1)


    print("\nAverage perceptual distance = %f\n" %(dist_all.mean()))

    dist_all = np.append(dist_all, dist_all.mean())
    print("Save %s" %metric_filename)
    np.savetxt(metric_filename, dist_all, fmt="%f")

