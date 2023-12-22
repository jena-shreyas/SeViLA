#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100l:2
#SBATCH --time=8:0:0
#SBATCH --account=def-egranger
#SBATCH --job-name=sevila-causalvidqa
#SBATCH --output=sevila-causalvidqa_%j.out
#SBATCH --error=sevila-causalvidqa_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shreyas.jena.1@etsmtl.net

module purge
module load python/3.9
module load scipy-stack/2021a
module load StdEnv/2020 gcc/9.3.0 cuda/11.4
module load opencv/4.8.0
module load arrow/13.0.0    
python -c "import cv2"
python -c "import pyarrow"
source $HOME/envs/sevila/bin/activate
cd $SCRATCH/BTP/SeViLA
export TORCH_CPP_LOG_LEVEL=INFO NCCL_DEBUG=INFO # debug

result_dir="expts"
exp_name='causalvidqa_infer'
date=$(date +"%d_%m_%Y_%h_%M_%S")
ckpt='sevila_checkpoints/sevila_pretrained.pth'

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 evaluate.py \
--cfg-path lavis/projects/sevila/eval/causalvidqa_eval.yaml \
--options run.output_dir=${result_dir}/${exp_name}/${date} \
model.frame_num=4 \
datasets.causalvidqa.vis_processor.eval.n_frms=32 \
run.batch_size_eval=1 \
model.task='qvh_freeze_loc_freeze_qa_vid_rand_init' \
model.finetuned=${ckpt} \
run.task='videoqa' \
run.num_workers=0
