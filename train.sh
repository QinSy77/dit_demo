#!/bin/bash
source activate pixartdemo
torchrun --nnodes=1 --standalone --nproc_per_node=2 \
    /mnt/data/qinshiyang/code/PixArt-alpha/train_scripts/train.py \
    /mnt/data/qinshiyang/code/PixArt-alpha/test/PixArt_xl2_img512_internal_for_pokemon_sample_training.py \
    --work-dir /mnt/data/qinshiyang/dlctest_dit_fine_train_outputs/trained_model \
    --report_to="all" \
    --loss_report_name="train_loss" 