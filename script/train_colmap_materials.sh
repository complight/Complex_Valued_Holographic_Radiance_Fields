python train.py \
    --lr 0.01 \
    --load_point \
    --dataset_name "colmap_materials" \
    --dataset_type "colmap" \
    --split_ratio 2.5 \
    --extra_scale 2300 \
    --densify_every 300 \
    --generate_dense_point 3 \
    --densepoint_scatter 1