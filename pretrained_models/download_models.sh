#!/bin/bash

URL=http://vllab.ucmerced.edu/wlai24/video_consistency/models

wget -N $URL/FlowNet2_checkpoint.pth.tar
wget -N $URL/ECCV18_blind_consistency.pth
wget -N $URL/ECCV18_blind_consistency_opts.pth
