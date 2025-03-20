# Single GPU
python test.py \
-c configs/pa_po_nuscenes_test.yaml \
-l test_nusc_202501019_test_HPC3.log \
--resume

# ## DDP
# export CUDA_VISIBLE_DEVICES=4,5,6,7

# nohup \
# python -m torch.distributed.launch --nproc_per_node=4 test.py \
# -c configs/nusc_test.yaml \
# -l test_nusc_20230101_pix_4k2bs_test_a6000.log \
# > nohup_nusc_20230101_pix_4k2bs_test_a6000.log 2>&1 &