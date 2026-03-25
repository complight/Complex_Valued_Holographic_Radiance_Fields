python train.py \
    --lr 0.01 \
    --load_point \
    --dataset_name "colmap_chair" \
    --dataset_type "colmap" \
    --split_ratio 2.2 \
    --extra_scale 550 \
    --densify_every 300 \
    --generate_dense_point 3 \
    --densepoint_scatter 1