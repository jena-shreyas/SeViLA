#!/bin/bash
# environment setup
module purge
module load python/3.9
module load scipy-stack/2021a
module load StdEnv/2020 gcc/9.3.0 cuda/11.4
module load opencv/4.8.0
module load arrow/13.0.0    
python -c "import cv2"
python -c "import pyarrow"
source $HOME/envs/sevila/bin/activate
