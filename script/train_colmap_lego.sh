python train.py \
    --lr 0.01 \
    --load_point \
    --dataset_name "colmap_lego" \
    --dataset_type "colmap" \
    --split_ratio 2.2 \
    --extra_scale 4290 \
    --densify_every 300 \
    --generate_dense_point 5 \
    --densepoint_scatter 10