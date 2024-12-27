# Single GPU
export CUDA_VISIBLE_DEVICES=1
python val.py \
-c configs/fps_nuscenes_mini_val_r50_mm.yaml \
-l test_single_fps_nusc_20240603.log

#TODO-YINING: remain bug
# ## DDP
# export CUDA_VISIBLE_DEVICES=4

# # nohup \
# python -m torch.distributed.launch --nproc_per_node=1 val.py \
# -c configs/pa_po_nuscenes_val_r50.yaml \
# -l nusc_20240528.log 
# # --local-rank 0

# # > nohup_nusc_20230101_pix_4k2bs_val_a6000.log 2>&1 &