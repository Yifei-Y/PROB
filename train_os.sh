CUDA_VISIBLE_DEVICES=3 python main_open_set.py \
    --output_dir "exps/openset/COCO/PROB" --dataset coco --dataset_file os_coco\
    --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 80\
    --batch_size 1 --lr 2e-5 --lr_backbone 4e-6 --obj_temp 1.3 --epochs 41\
    --model_type 'prob' --obj_loss_coef 8e-4\
    --wandb_project 'PROB' --wandb_name "OS_COCO_PROB"\
    --exemplar_replay_selection --exemplar_replay_max_length 850
