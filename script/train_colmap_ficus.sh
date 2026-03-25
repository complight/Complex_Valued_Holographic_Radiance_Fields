python train.py \
    --lr 0.01 \
    --load_point \
    --dataset_name "colmap_ficus" \
    --dataset_type "colmap" \
    --split_ratio 1.0 \
    --extra_scale 1020 \
    --densify_every 300 \
    --generate_dense_point 5 \
    --densepoint_scatter 1