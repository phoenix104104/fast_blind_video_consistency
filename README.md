# Learning Blind Video Temporal Consistency

[Wei-Sheng Lai](http://graduatestudents.ucmerced.edu/wlai24/), 
[Jia-Bin Huang](https://filebox.ece.vt.edu/~jbhuang/), 
[Oliver Wang](http://www.oliverwang.info/), 
[Eli Shechtman](https://research.adobe.com/person/eli-shechtman/), 
[Ersin Yumer](http://www.meyumer.com/), 
and [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/)

[European Conference on Computer Vision (ECCV), 2018](https://eccv2018.org/)

[[Project page]](http://vllab.ucmerced.edu/wlai24/video_consistency/)[[Paper]](http://arxiv.org/abs/1808.00449)

<img src="teaser_small.gif" width="1000">

### Table of Contents
1. [Introduction](#introduction)
1. [Requirements and Dependencies](#requirements-and-dependencies)
1. [Installation](#installation)
1. [Dataset](#dataset)
1. [Apply Pre-trained Models](#apply-pre-trained-models)
1. [Training and Testing](#training-and-testing)
1. [Evaluation](#evaluation)
1. [Image Processing Algorithms](#image-processing-algorithms)
1. [Citation](#citation)


### Introduction
Our method takes the original unprocessed and per-frame processed videos as inputs to produce a temporally consistent video. Our approach is agnostic to specific image processing algorithms applied on the original video.


### Requirements and dependencies
- [Pytorch 0.4.0](https://pytorch.org/) (FlowNet2 dependencies cannot be compiled on Pytorch 0.4.1)
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [FlowNet2-Pytorch](https://github.com/NVIDIA/flownet2-pytorch) (Code and model are already included in this repository)
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity) (for evaluation)

Our code is tested on Ubuntu 16.04 with cuda 9.0 and cudnn 7.0.


### Installation
Download repository:

    git clone https://github.com/phoenix104104/fast_blind_video_consistency.git

Compile FlowNet2 dependencies (correlation, resample, and channel norm layers):

    ./install.sh


### Dataset
Download our training and testing datasets:

    cd data
    ./download_data.sh [train | test | all]
    cd ..
    
For example, download training data only:

    ./download_data.sh train
    
Download both training and testing data:

    ./download_data.sh all
    
You can also download the results of [[Bonneel et al. 2015]](https://perso.liris.cnrs.fr/nicolas.bonneel/consistency/) and our approach:

    ./download_data.sh results
    
    
### Apply pre-trained models
Download pretrained models (including FlowNet2 and our model):
    
    cd pretrained_models
    ./download_models.sh
    cd ..

Test pre-trained model:

    python test_pretrained.py -dataset DAVIS -task WCT/wave
    
The output frames are saved in `data/test/ECCV18/WCT/wave/DAVIS`.


### Training and testing
Train a new model:

    python train.py -datasets_tasks W3_D1_C1_I1

We have specified all the default parameters in train.py. `lists/train_tasks_W3_D1_C1_I1.txt` specifies the dataset-task pairs for training.

Test a model:

    python test.py -method MODEL_NAME -epoch N -dataset DAVIS -task WCT/wave
    
Check the checkpoint folder for the `MODEL_NAME`.
The output frames are saved in `data/test/MODEL_NAME/epoch_N/WCT/wave/DAVIS`.


You can also generate results for multiple tasks using the following script:

    python batch_test.py -method output/MODEL_NAME/epoch_N

which will test all the tasks in `lists/test_tasks.txt`.


### Evaluation
**Temporal Warping Error**

To compute the temporal warping error, we first need to generate optical flow and occlusion masks:

    python compute_flow_occlusion.py -dataset DAVIS -phase test

The flow will be stored in `data/test/fw_flow/DAVIS`. The occlusion masks will be stored in `data/test/fw_occlusion/DAVIS`.

Then, run the evaluation script:

    python evaluate_WarpError.py -method output/MODEL_NAME/epoch_N -task WCT/wave
    
**LPIPS**

Download [LPIPS repository](https://github.com/richzhang/PerceptualSimilarity) and change `LPIPS_dir` in evalate_LPIPS.py if necesary (default path is `../LPIPS`).

Run the evaluation script:

    python evaluate_LPIPS.py -method output/MODEL_NAME/epoch_N -task WCT/wave

**Batch evaluation**

You can evaluate multiple tasks using the following script:

    python batch_evaluate.py -method output/MODEL_NAME/epoch_N -metric LPIPS
    python batch_evaluate.py -method output/MODEL_NAME/epoch_N -metric WarpError
    
which will evaluate all the tasks in `lists/test_tasks.txt`.

   
### Test on new videos
To test our model on new videos or applications, please follow the folder structure in `./data`.

Given a video, we extract frames named as `%05d.jpg` and save frames in `data/test/input/DATASET/VIDEO`.

The per-frame processed video is stored in `data/test/processed/TASK/DATASET/VIDEO`, where `TASK` is the image processing algorithm applied on the original video.


### Image Processing Algorithms
We use the following algorithms to obtain per-frame processed results:

**Style transfer**
- [WCT: Universal Style Transfer via Feature Transforms, NIPS 2017](https://github.com/Yijunmaverick/UniversalStyleTransfer)
- [Fast Neural Style Transfer: Perceptual Losses for Real-Time Style Transfer and Super-Resolution, ECCV 2016](https://github.com/jcjohnson/fast-neural-style)

**Image Enhancement**
- [DBL: Deep Bilateral Learning for Real-Time Image Enhancement, Siggraph 2017](https://groups.csail.mit.edu/graphics/hdrnet/)

**Intrinsic Image Decomposition**
- [Intrinsic Images in the Wild, Siggraph 2014](http://opensurfaces.cs.cornell.edu/publications/intrinsic/)

**Image-to-Image Translation**
- [CycleGAN: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks, ICCV 2017](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

**Colorization**
- [Colorful Image Colorization, ECCV 2016](https://github.com/richzhang/colorization)
- [Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization with Simultaneous Classification, Siggraph 2016](http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/en/)


### Citation

If you find the code and datasets useful in your research, please cite:
    
    @inproceedings{Lai-ECCV-2018,
        author    = {Lai, Wei-Sheng and Huang, Jia-Bin and Wang, Oliver and Shechtman, Eli and Yumer, Ersin and Yang, Ming-Hsuan}, 
        title     = {Learning Blind Video Temporal Consistency}, 
        booktitle = {European Conference on Computer Vision},
        year      = {2018}
    }
    
