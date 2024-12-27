# # Single GPU
# python train.py \
# -c configs/pa_po_nuscenes_trainval_r50.yaml \
# -l test_nusc_20230101_pix_4k2bs_val_a6000.log

## DDP
export CUDA_VISIBLE_DEVICES=1

# nohup \
python -m torch.distributed.launch --nproc_per_node=1 train.py \
-c configs/fps_nuscenes_mini_r50_single.yaml \
-l nusc_fps_single_20240529_HPC3.log \
# -r
# --local-rank 0
# > nohup_nusc_20230101_pix_4k2bs_a6000.log 2>&1 &