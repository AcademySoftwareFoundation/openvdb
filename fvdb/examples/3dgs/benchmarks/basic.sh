SCENE_DIR="/workspace/data/360_v2/"
RESULT_DIR="results/benchmark"
SCENE="garden"
DATA_FACTOR=4

# train without eval
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --eval_steps -1 --disable_viewer --data_factor $DATA_FACTOR \
    --data_dir data/360_v2/$SCENE/ \
    --result_dir $RESULT_DIR/$SCENE/

# # run eval and render
# for CKPT in $RESULT_DIR/$SCENE/ckpts/*;
# do
#     CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --disable_viewer --data_factor $DATA_FACTOR \
#         --data_dir data/360_v2/$SCENE/ \
#         --result_dir $RESULT_DIR/$SCENE/ \
#         --ckpt $CKPT
# done
