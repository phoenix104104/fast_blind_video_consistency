# Learning Blind Video Temporal Consistency (ECCV 2018) 

[Wei-Sheng Lai](http://graduatestudents.ucmerced.edu/wlai24/), 
[Jia-Bin Huang](https://filebox.ece.vt.edu/~jbhuang/), 
[Oliver Wang](http://www.oliverwang.info/), 
[Eli Shechtman](https://research.adobe.com/person/eli-shechtman/), 
[Ersin Yumer](http://www.meyumer.com/), 
and [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/)

[European Conference on Computer Vision (ECCV), 2018](https://eccv2018.org/)

[[Project page]](http://vllab.ucmerced.edu/wlai24/video_consistency/)[Paper]

<img src="teaser_small.gif" width="1000">

### Table of Contents
1. [Introduction](#introduction)
1. [Citation](#citation)
1. [Requirements and Dependencies](#requirements-and-dependencies)
1. [Installation](#installation)
1. [Dataset](#dataset)
1. [Apply Pre-trained Models](#apply-pre-trained-models)
1. [Training and Testing](#training-and-testing)
1. [Image processing algorithms](#image-processing-algorithms)

### Introduction
Our method takes the original unprocessed and per-frame processed videos as inputs to produce a temporally consistent video. Our approach is agnostic to specific image processing algorithms applied on the original video.




### Citation

If you find the code and datasets useful in your research, please cite:
    
    @inproceedings{Lai-ECCV-2018,
        author    = {Lai, Wei-Sheng and Huang, Jia-Bin and Wang, Oliver and Shechtman, Eli and Yumer, Ersin and Yang, Ming-Hsuan}, 
        title     = {Learning Blind Video Temporal Consistency}, 
        booktitle = {European Conference on Computer Vision},
        year      = {2018}
    }
    

### Requirements and dependencies
- [Pytorch 0.4](https://pytorch.org/)
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity) (for evaluation)


### Dataset


### Installation
Download repository:

    $ git clone https://github.com/phoenix104104/fast_blind_video_consistency.git


### Apply pre-trained models


### Training and testing

### Image processing algorithms
We use the following algorithms to obtain per-frame processed results:

**Style transfer**

