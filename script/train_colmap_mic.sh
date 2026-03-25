python train.py \
    --lr 0.01 \
    --load_point \
    --dataset_name "colmap_mic" \
    --dataset_type "colmap" \
    --split_ratio 2.2 \
    --extra_scale 1100 \
    --densify_every 300 \
    --generate_dense_point 4 \
    --densepoint_scatter 1