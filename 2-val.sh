# # Single GPU
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python val.py \
# -c configs/pa_po_nuscenes_val_r50.yaml \
# -l infer_full_20250317.log \
# -r 

#for debug only

#TODO-YINING: remain bug
# ## DDP
export CUDA_VISIBLE_DEVICES=0,1,2,3

# # nohup \
python -m torch.distributed.launch --nproc_per_node=4 val.py \
-c configs/pa_po_nuscenes_val_r50.yaml \
-l infer_full_20250317.log \
-r
# --local-rank 0

# # > nohup_nusc_20230101_pix_4k2bs_val_a6000.log 2>&1 &