# # Single GPU
# python train.py \
# -c configs/pa_po_nuscenes_trainval_r50.yaml \
# -l test_nusc_20230101_pix_4k2bs_val_a6000.log

## DDP
export CUDA_VISIBLE_DEVICES=0,1,2,3

# nohup \
python -m torch.distributed.launch --nproc_per_node=4 train.py \
-c configs/pa_po_nuscenes_trainval_r50.yaml \
-l nusc_fps_20241227_HPC3.log \
# -r
# --local-rank 0
# > nohup_nusc_20230101_pix_4k2bs_a6000.log 2>&1 &