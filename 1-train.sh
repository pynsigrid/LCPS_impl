# # Single GPU
# python train.py \
# -c configs/pa_po_nuscenes_trainval_r50.yaml \
# -l test_nusc_20230101_pix_4k2bs_val_a6000.log

## DDP
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9

# nohup \
python -m torch.distributed.launch --nproc_per_node=10 train.py \
-c configs/pa_po_nuscenes_trainval_r50_bs1.yaml \
-l nusc_train_20241228_10xbs1_ASTAR.log \
# -r
# --local-rank 0
# > nohup_nusc_20230101_pix_4k2bs_a6000.log 2>&1 &