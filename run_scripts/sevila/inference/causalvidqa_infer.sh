#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100l:4
#SBATCH --time=8:00:0
#SBATCH --account=def-egranger
#SBATCH --job-name=sevila-causalvidqa
#SBATCH --output=sevila-causalvidqa_%j.out
#SBATCH --error=sevila-causalvidqa_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shreyas.jena.1@etsmtl.net

result_dir="expts"
qn_type=$1
exp_name='causalvidqa_infer'
ckpt='sevila_checkpoints/sevila_pretrained.pth'

module purge
module load python/3.9
module load StdEnv/2020 gcc/9.3.0 cuda/11.4
module load opencv/4.8.0
module load arrow/13.0.0
python -c "import cv2"
python -c "import pyarrow"
source $HOME/sevila/bin/activate
cd $SCRATCH/BTP/SeViLA
export TORCH_CPP_LOG_LEVEL=INFO NCCL_DEBUG=INFO # debug

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 evaluate.py \
--cfg-path lavis/projects/sevila/eval/causalvidqa_eval.yaml \
--options run.output_dir=${result_dir}/${exp_name}/${qn_type} \
model.frame_num=4 \
datasets.causalvidqa.vis_processor.eval.n_frms=32 \
run.batch_size_eval=1 \
model.task='qvh_freeze_loc_freeze_qa_vid' \
model.finetuned=${ckpt} \
run.task='videoqa' \
run.num_workers=0
